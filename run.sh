#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/scripts/pixi_env_check.sh"

usage() {
    echo "Usage: $0 env1 [env2 ...] [sklbench args...]" >&2
    echo "  Runs 'pixi run -e <env> python -m sklbench <args...>' for each" >&2
    echo "  environment given. Environments are the leading arguments, up to" >&2
    echo "  the first one starting with '-'; everything from there on is" >&2
    echo "  passed through to sklbench." >&2
    echo "  Example: $0 sklearn-pypi intel --config configs/smoke_check_test.py" >&2
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
    echo "  If cuml is one of the environments, every environment in that" >&2
    echo "  command (including sklearn-pypi) is limited to the estimators the" >&2
    echo "  cuML wrappers cover, via SKLBENCH_LIMIT_TO_CUML_ESTIMATORS=1." >&2
    echo "  A later run without cuml keeps the full suite." >&2
    echo "" >&2
    echo "  <env> works with any Pixi environment that path-depends on" >&2
    echo "  sklearn-src (currently sklearn-dev, sklearn-dev-libomp," >&2
    echo "  sklearn-dev-mkl and sklearn-dev-freethreading), so the same ref" >&2
    echo "  can also be compared" >&2
    echo "  across those environments, e.g. to" >&2
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

# run.sh is the only place that sees every environment in this command.
# The Python filter runs in a separate process per environment, so it cannot
# tell that cuml was requested alongside sklearn-pypi. Export the flag before
# the loop, from the whole list, so the sklearn process is limited too.
limit_to_cuml_estimators=false
for env_spec in "${envs[@]}"; do
    if [ "${env_spec%%@*}" = "cuml" ]; then
        limit_to_cuml_estimators=true
        break
    fi
done
if [ "$limit_to_cuml_estimators" = true ]; then
    export SKLBENCH_LIMIT_TO_CUML_ESTIMATORS=1
else
    unset SKLBENCH_LIMIT_TO_CUML_ESTIMATORS
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
