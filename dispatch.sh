#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/scripts/pixi_env_check.sh"

usage() {
    echo "Usage: $0 env1 [env2 ...] --config PATH [--runner RUNNER] [--branch BRANCH]" >&2
    echo "  Triggers the 'Run Benchmarks' workflow (.github/workflows/run-benchmarks.yml)" >&2
    echo "  via 'gh workflow run', passing env1 [env2 ...] as its 'env' input." >&2
    echo "  Environments are the leading arguments, same as run.sh: space-separated," >&2
    echo "  may include '[all]' / '[builds]' tokens, and may use the env@owner:ref" >&2
    echo "  form (see run.sh --help)." >&2
    echo "" >&2
    echo "  --config PATH    benchmark config script (required)" >&2
    echo "  --runner RUNNER  intel-laptop (default) | intel-xeon | both |" >&2
    echo "                   aws-nvidia-l4. The AWS runner must be dispatched" >&2
    echo "                   alone (no 'all' choice); use the 16 GiB config," >&2
    echo "                   e.g. configs/all_models_16gb.py." >&2
    echo "  --branch BRANCH  dispatch off BRANCH instead of main. Requires you to be" >&2
    echo "                   checked out on BRANCH and for it to be up to date with" >&2
    echo "                   its remote tracking branch." >&2
    echo "" >&2
    echo "  Example: $0 sklearn-pypi intel --config configs/smoke_check_test.py" >&2
    echo "  Example: $0 '[all]' --config configs/all_models.py --runner both" >&2
    echo "  Example: $0 '[all]' --config configs/all_models_16gb.py --runner aws-nvidia-l4" >&2
    echo "  Example: $0 '[all]' --config configs/all_models.py --branch my-feature" >&2
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

envs=()
runner="intel-laptop"
config=""
branch="main"
parsing_envs=true

while [ "$#" -gt 0 ]; do
    arg="$1"
    if [ "$parsing_envs" = true ] && [[ "$arg" != -* ]]; then
        envs+=("$arg")
        shift
        continue
    fi
    parsing_envs=false

    case "$arg" in
        --runner)
            runner="$2"
            shift 2
            ;;
        --config)
            config="$2"
            shift 2
            ;;
        --branch)
            branch="$2"
            shift 2
            ;;
        *)
            echo "error: unrecognized argument '$arg'" >&2
            usage
            exit 2
            ;;
    esac
done

if [ "${#envs[@]}" -eq 0 ]; then
    usage
    exit 2
fi

if [ -z "$config" ]; then
    echo "error: --config is required" >&2
    usage
    exit 2
fi

case "$runner" in
    intel-laptop|intel-xeon|both|aws-nvidia-l4) ;;
    *)
        echo "error: unknown --runner '$runner' (expected intel-laptop, intel-xeon, both or aws-nvidia-l4)" >&2
        exit 2
        ;;
esac

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "$branch" ]; then
    echo "error: current branch is '$current_branch', not '$branch'; switch to '$branch' before dispatching (pass --branch to dispatch off a branch other than main)" >&2
    exit 1
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)" || upstream=""
if [ -z "$upstream" ]; then
    echo "error: '$branch' has no upstream tracking branch; push it before dispatching" >&2
    exit 1
fi
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "$upstream")"
if [ "$local_head" != "$remote_head" ]; then
    echo "error: '$branch' is not pushed to '$upstream' (the workflow runs off the remote ref, not your local checkout); push before dispatching" >&2
    exit 1
fi

env_input="${envs[*]}"

real_envs=()
status=0
for env_spec in $env_input; do
    # "[all]" / "[builds]" are special tokens resolved server-side, per
    # runner, by run-benchmarks.yml - nothing to check locally.
    if [[ "$env_spec" == \[*\] ]]; then
        continue
    fi

    env="${env_spec%%@*}"
    if ! pixi_env_exists "$env"; then
        echo "error: '$env' is not a pixi environment defined in pixi.toml (parsed from '$env_spec')" >&2
        status=1
        continue
    fi
    real_envs+=("$env")
done

if [ "$status" -ne 0 ]; then
    exit 1
fi

# An AWS dispatch queues a `runner-ready` gate job on the GPU box before the
# instance is started. If another AWS dispatch is already active, that gate
# job would sit behind it and burn this run's readiness window, so refuse
# early. Only AWS runs matter: an Intel benchmark uses a different runner
# and is safe to overlap.
if [ "$runner" = "aws-nvidia-l4" ]; then
    active_runs="$(gh run list --workflow run-benchmarks.yml --limit 20 \
        --json databaseId,status \
        --jq '.[] | select(.status == "queued" or .status == "in_progress" or .status == "waiting" or .status == "pending" or .status == "requested") | .databaseId')"
    while IFS= read -r run_id; do
        [ -n "$run_id" ] || continue
        has_gate_job="$(gh run view "$run_id" --json jobs \
            --jq '[.jobs[] | select(.name == "runner-ready")] | length > 0')"
        if [ "$has_gate_job" = "true" ]; then
            echo "error: AWS benchmark run $run_id is still active; wait for it to finish (or cancel it) before dispatching another one" >&2
            exit 1
        fi
    done <<< "$active_runs"
fi

# Validate the config under each real pixi environment being dispatched:
# configs read PIXI_ENVIRONMENT_NAME (via implementations_for_pixi_env) to
# pick which implementations to generate cases for, so validation has to run
# per-env rather than under a single fixed environment.
#
# The workflow re-validates on the benchmark host before running, so this is
# only a fast local feedback loop. Skip it where it cannot work: the
# skl-nvidia environment can only be installed on a machine with a CUDA GPU,
# and installing it just to check a config would pull a multi-GB CUDA stack.
# The workflow validates it on the benchmark host instead. CPU-only
# environments are still validated here when they are named explicitly;
# [all]/[builds] are resolved per runner by the workflow and validated there.
validated=()
for env in "${real_envs[@]}"; do
    already_validated=false
    for v in "${validated[@]}"; do
        if [ "$v" = "$env" ]; then
            already_validated=true
            break
        fi
    done
    if [ "$already_validated" = true ]; then
        continue
    fi
    validated+=("$env")

    if [ "$env" = "skl-nvidia" ]; then
        echo "note: skipping local validation for '$env' (validated on the benchmark host)" >&2
        continue
    fi

    echo "=== pixi run --frozen -e $env python -m sklbench --config $config --validate-only ===" >&2
    if ! pixi run --frozen -e "$env" python -m sklbench --config "$config" --validate-only; then
        echo "error: config validation failed for pixi environment '$env', aborting before dispatching the workflow" >&2
        exit 1
    fi
done

cmd=(gh workflow run run-benchmarks.yml --ref "$branch" -f env="$env_input" -f runner="$runner" -f config="$config")

echo "=== ${cmd[*]} ===" >&2
"${cmd[@]}"
