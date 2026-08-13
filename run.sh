#!/usr/bin/env bash

usage() {
    echo "Usage: $0 env1 [env2 ...] [sklbench args...]" >&2
    echo "  Runs 'pixi run -e <env> python -m sklbench <args...>' for each" >&2
    echo "  environment given. Environments are the leading arguments, up to" >&2
    echo "  the first one starting with '-'; everything from there on is" >&2
    echo "  passed through to sklbench." >&2
    echo "  Example: $0 sklearn-pypi intel --config configs/all_models_test.py" >&2
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
for env in "${envs[@]}"; do
    echo "=== pixi run --frozen -e $env python -m sklbench ${args[*]} ===" >&2
    if ! pixi run --frozen -e "$env" python -m sklbench "${args[@]}"; then
        status=1
    fi
done

exit "$status"
