#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [intel|nvidia]" >&2
    echo "  no argument: run CPU-only benchmarks" >&2
    echo "  intel:       also run Intel GPU benchmarks" >&2
    echo "  nvidia:      also run NVIDIA GPU benchmarks" >&2
}

if [ "$#" -gt 1 ]; then
    usage
    exit 2
fi

mode="${1:-cpu}"

run_cpu() {
    pixi run -e sklearn python -m sklbench --config configs/sklearn.json
    pixi run -e skl-cpu python -m sklbench --config configs/array-api-cpu.json
    pixi run -e intel python -m sklbench --config configs/sklearnex-cpu.json
}

case "$mode" in
    cpu|intel|nvidia)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage
        exit 2
        ;;
esac

run_cpu

case "$mode" in
    cpu)
        ;;
    intel)
        pixi run -e skl-intel python -m sklbench --config configs/array-api-intel.json
        pixi run -e intel python -m sklbench --config configs/sklearnex-gpu.json
        ;;
    nvidia)
        pixi run -e skl-nvidia python -m sklbench --config configs/array-api-nvidia.json
        ;;
esac
