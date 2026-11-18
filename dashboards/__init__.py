"""Dashboard generator entry points."""

from argparse import ArgumentParser
from pathlib import Path


# Single source of truth for hardware-hash -> display name, shared by every
# dashboard so a rename doesn't have to be repeated file by file. Names give
# the machine kind plus a recognizable product line (Xeon, Ultra, M4, L4),
# not exact model numbers or vendor codenames.
# Dashboards list machines in this order: laptops, desktops, then servers.
HARDWARE_NAMES = {
    "3b5e61": "Intel Core Ultra laptop",
    "b281b2": "Apple M4 laptop",
    "3dd14d": "Apple M2 laptop",
    "be1055": "Low-end Intel laptop",
    "332097": "Intel Core Ultra desktop",
    "5dcf30": "AMD Workstation",
    "534824": "High-end Intel Xeon server",
    "d189c4": "NVIDIA L4 VM",
    "7c42d6": "64-core AMD EPYC VM",
}

# So that a machine has the same color on every dashboard.
HARDWARE_COLORS = {
    "3b5e61": "#2294c4",
    "b281b2": "#f7931e",
    "3dd14d": "#9aa500",
    "be1055": "#15688c",
    "332097": "#8045e5",
    "5dcf30": "#b76c13",
    "534824": "#2e9e5b",
    "d189c4": "#d1495b",
    "7c42d6": "#c2379a",
}

GPU_NAMES = {
    "3b5e61": "Intel Core Ultra laptop GPU",
    "b281b2": "Apple M4 laptop GPU",
    "3dd14d": "Apple M2 laptop GPU",
    "332097": "Intel Arc B580",
    "5dcf30": "NVIDIA RTX 2060",
    "d189c4": "NVIDIA L4 g6.2xlarge",
}

# Set of Pixi platforms (as declared by `platforms = [...]` in pixi.toml's
# [feature.*] blocks) each hardware hash can run, so
# `scripts/what_to_rerun.py --hardware` can drop envs that can't even be
# installed there (see `sklbench.reporting.envs.env_platforms`). This is
# genuinely new information (which OS/arch each physical machine runs) that
# isn't derivable from anywhere else in the repo. A machine with a CUDA GPU
# can run both the plain `linux-64` envs and the `linux-64-cuda-12-9`
# variants; a CPU-only Linux machine only the former.
HARDWARE_PLATFORMS = {
    "3b5e61": {"linux-64"},
    "534824": {"linux-64"},
    "be1055": {"linux-64"},
    "b281b2": {"osx-arm64"},
    "3dd14d": {"osx-arm64"},
    "5dcf30": {"win-64"},
    "332097": {"linux-64"},
    "d189c4": {"linux-64", "linux-64-cuda-12-9"},
    "7c42d6": {"linux-64"},
}

# `SOURCE_CONFIGS`/`SOURCE_ENVS` of gen_cross_comparison, which draws on the broad
# configs/all_models.py matrix across every Pixi env it sweeps. configs/all_models_16gb.py is
# a strict subset of it (same cases), run on machines with 16GB of RAM.
# configs/all_models_cuml_only.py is the same kind of subset, limited to the
# estimators with a native cuML wrapper.
# configs/smoke_check_test.py
# is deliberately excluded - it's a CI-only sanity config, not meant to
# produce dashboard-worthy results.
GENERAL_SOURCE_CONFIGS = [
    "configs/all_models.py",
    "configs/all_models_16gb.py",
    "configs/all_models_cuml_only.py",
]

GENERAL_SOURCE_ENVS = [
    "sklearn-pypi",
    "sklearn-cf-default",
    "sklearn-cf-libgomp-openblas",
    "sklearn-cf-libomp-openblas",
    "sklearn-cf-libomp-openblas-omp",
    "sklearn-cf-mkl",
    "skl-cpu",
    "skl-intel",
    "skl-nvidia",
    "skl-mps",
    "intel",
]


def dashboard_output_dir() -> Path:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_site"),
        help="Directory where the generated dashboard HTML file is written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.output_dir
