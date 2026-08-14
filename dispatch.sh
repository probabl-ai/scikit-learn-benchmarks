#!/usr/bin/env bash

usage() {
    echo "Usage: $0 env1 [env2 ...] --config PATH [--runner RUNNER] [--ref REF]" >&2
    echo "  Triggers the 'Run Benchmarks' workflow (.github/workflows/run-benchmarks.yml)" >&2
    echo "  via 'gh workflow run', passing env1 [env2 ...] as its 'env' input." >&2
    echo "  Environments are the leading arguments, same as run.sh: space-separated," >&2
    echo "  may include '[all]' / '[builds]' tokens, and may use the env@owner:ref" >&2
    echo "  form (see run.sh --help)." >&2
    echo "" >&2
    echo "  --config PATH    benchmark config script (required)" >&2
    echo "  --runner RUNNER  intel-laptop (default) | intel-gnr | both" >&2
    echo "  --ref REF        git ref the workflow file is read from (default: current" >&2
    echo "                   branch), forwarded to 'gh workflow run --ref'" >&2
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
ref=""
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
        --ref)
            ref="$2"
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

env_input="${envs[*]}"

if [ -z "$ref" ]; then
    ref="$(git rev-parse --abbrev-ref HEAD)"
fi

cmd=(gh workflow run run-benchmarks.yml --ref "$ref" -f env="$env_input" -f runner="$runner" -f config="$config")

echo "=== ${cmd[*]} ===" >&2
"${cmd[@]}"
