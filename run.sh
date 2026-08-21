#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/scripts/pixi_env_check.sh"

# Collects --config/-c values from a sklbench arg list into the array named
# by $1 (argparse nargs='+' semantics: consume tokens up to the next
# '-'-prefixed flag).
extract_configs() {
    local -n out_ref="$1"
    shift
    local i=0
    local rest=("$@")
    while [ "$i" -lt "${#rest[@]}" ]; do
        if [ "${rest[$i]}" = "--config" ] || [ "${rest[$i]}" = "-c" ]; then
            i=$((i + 1))
            while [ "$i" -lt "${#rest[@]}" ] && [[ "${rest[$i]}" != -* ]]; do
                out_ref+=("${rest[$i]}")
                i=$((i + 1))
            done
        else
            i=$((i + 1))
        fi
    done
}

usage() {
    echo "Usage: $0 env1 [env2 ...] [sklbench args...]" >&2
    echo "  Runs 'pixi run -e <env> python -m sklbench <args...>' for each" >&2
    echo "  environment given. Environments are the leading arguments, up to" >&2
    echo "  the first one starting with '-'; everything from there on is" >&2
    echo "  passed through to sklbench." >&2
    echo "  Example: $0 sklearn-pypi intel --config configs/all_models_test.py" >&2
    echo "" >&2
    echo "  An environment may instead be given as env@owner:ref, e.g." >&2
    echo "  sklearn-dev@cakedev0:hgb/use_threads_if. Before running sklbench" >&2
    echo "  for that environment, this checks out https://github.com/<owner>/scikit-learn.git" >&2
    echo "  at <ref> via scripts/setup_sklearn_ref.sh and installs it editable" >&2
    echo "  into <env>. Use this to compare a PR branch against a base ref," >&2
    echo "  e.g.:" >&2
    echo "  $0 sklearn-dev@cakedev0:hgb/use_threads_if sklearn-dev@scikit-learn:main \\" >&2
    echo "      --config configs/hgb_scaling.py" >&2
    echo "" >&2
    echo "  <env> works with any Pixi environment that path-depends on" >&2
    echo "  sklearn-src (currently sklearn-dev and sklearn-dev-libomp), so the" >&2
    echo "  same ref can also be compared across those environments, e.g. to" >&2
    echo "  isolate an OpenMP-runtime effect on the exact same commit:" >&2
    echo "  $0 sklearn-dev@scikit-learn:main sklearn-dev-libomp@scikit-learn:main \\" >&2
    echo "      --config configs/hgb_scaling.py" >&2
}

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

envs=()
args=()
parsing_envs=true

for arg in "$@"; do
    if [ "$parsing_envs" = true ] && [[ "$arg" != -* ]]; then
        envs+=("$arg")
    else
        parsing_envs=false
        args+=("$arg")
    fi
done

if [ "${#envs[@]}" -eq 0 ]; then
    usage
    exit 2
fi

configs=()
extract_configs configs "${args[@]}"
if [ "${#configs[@]}" -gt 0 ]; then
    echo "=== pixi run --frozen -e default python -m sklbench --config ${configs[*]} --validate-only ===" >&2
    if ! pixi run --frozen -e default python -m sklbench --config "${configs[@]}" --validate-only; then
        echo "error: config validation failed, aborting before running any environment" >&2
        exit 1
    fi
fi

status=0
for env_spec in "${envs[@]}"; do
    env="$env_spec"

    if [[ "$env_spec" == *@* ]]; then
        env="${env_spec%%@*}"
        owner_ref="${env_spec#*@}"
        owner="${owner_ref%%:*}"
        ref="${owner_ref#*:}"
        if [ -z "$owner" ] || [ -z "$ref" ] || [ "$owner" = "$owner_ref" ]; then
            echo "error: invalid env spec '$env_spec', expected env@owner:ref" >&2
            status=1
            continue
        fi
    fi

    # Pixi environment names are lowercase letters, numbers, and dashes only.
    # A bare (no "@") env spec that fails this is almost always an
    # env@owner:ref spec that's missing its "@" (e.g. a colon-containing ref
    # pasted without it) - catch that here instead of letting it reach `pixi
    # run` as a bogus environment name, which fails late/confusingly (only
    # surfaced downstream, e.g. by a CI "Classify benchmark results" step).
    if [[ ! "$env" =~ ^[a-z0-9-]+$ ]]; then
        echo "error: '$env' is not a valid pixi environment name (parsed from '$env_spec'); did you forget the '@' before the owner in an env@owner:ref spec?" >&2
        status=1
        continue
    fi

    if ! pixi_env_exists "$env"; then
        echo "error: '$env' is not a pixi environment defined in pixi.toml (parsed from '$env_spec')" >&2
        status=1
        continue
    fi

    if [[ "$env_spec" == *@* ]]; then
        remote="https://github.com/$owner/scikit-learn.git"

        echo "=== scripts/setup_sklearn_ref.sh --remote $remote --ref $ref --env $env ===" >&2
        if ! "$script_dir/scripts/setup_sklearn_ref.sh" --remote "$remote" --ref "$ref" --env "$env"; then
            status=1
            continue
        fi
    fi

    echo "=== pixi run --frozen -e $env python -m sklbench ${args[*]} ===" >&2
    if ! pixi run --frozen -e "$env" python -m sklbench "${args[@]}"; then
        status=1
    fi
done

exit "$status"
