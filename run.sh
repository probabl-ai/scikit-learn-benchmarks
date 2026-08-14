#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
