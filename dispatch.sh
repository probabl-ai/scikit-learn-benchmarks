#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/scripts/pixi_env_check.sh"

usage() {
    echo "Usage: $0 env1 [env2 ...] --config PATH [--runner RUNNER]" >&2
    echo "  Triggers the 'Run Benchmarks' workflow (.github/workflows/run-benchmarks.yml)" >&2
    echo "  via 'gh workflow run', passing env1 [env2 ...] as its 'env' input." >&2
    echo "  Environments are the leading arguments, same as run.sh: space-separated," >&2
    echo "  may include '[all]' / '[builds]' tokens, and may use the env@owner:ref" >&2
    echo "  form (see run.sh --help)." >&2
    echo "" >&2
    echo "  --config PATH    benchmark config script (required)" >&2
    echo "  --runner RUNNER  intel-laptop (default) | intel-gnr | both" >&2
    echo "" >&2
    echo "  Example: $0 sklearn-pypi intel --config configs/all_models_test.py" >&2
    echo "  Example: $0 '[all]' --config configs/all_models.py --runner both" >&2
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

envs=()
runner="intel-laptop"
config=""
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

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "main" ]; then
    echo "error: current branch is '$current_branch', not 'main'; switch to main before dispatching" >&2
    exit 1
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)" || upstream=""
if [ -z "$upstream" ]; then
    echo "error: 'main' has no upstream tracking branch; push it before dispatching" >&2
    exit 1
fi
local_head="$(git rev-parse HEAD)"
remote_head="$(git rev-parse "$upstream")"
if [ "$local_head" != "$remote_head" ]; then
    echo "error: 'main' is not pushed to '$upstream' (the workflow runs off the remote ref, not your local checkout); push before dispatching" >&2
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

# Validate the config under each real pixi environment being dispatched:
# configs read PIXI_ENVIRONMENT_NAME (via implementations_for_pixi_env) to
# pick which implementations to generate cases for, so validation has to run
# per-env rather than under a single fixed environment.
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

    echo "=== pixi run --frozen -e $env python -m sklbench --config $config --validate-only ===" >&2
    if ! pixi run --frozen -e "$env" python -m sklbench --config "$config" --validate-only; then
        echo "error: config validation failed for pixi environment '$env', aborting before dispatching the workflow" >&2
        exit 1
    fi
done

cmd=(gh workflow run run-benchmarks.yml --ref main -f env="$env_input" -f runner="$runner" -f config="$config")

echo "=== ${cmd[*]} ===" >&2
"${cmd[@]}"
