"""Dashboard generator entry points."""

from argparse import ArgumentParser
from pathlib import Path


# Single source of truth for hardware-hash -> display name, shared by every
# dashboard so a rename doesn't have to be repeated file by file. Names are
# meant to be readable by non-hardware-specialists (relative age/power),
# not model numbers or vendor codenames.
HARDWARE_NAMES = {
    "3b5e61": "Modern Intel laptop",
    "534824": "High-end Intel server",
    "be1055": "Low-end Intel laptop",
    "5dcf30": "AMD Workstation",
    "b281b2": "Apple M4",
}

GPU_NAMES = {
    "3b5e61": "Modern Intel laptop GPU",
    "b281b2": "Apple M4 GPU",
    "5dcf30": "NVIDIA RTX 2060"
}

# Pixi platform (as declared by `platforms = [...]` in pixi.toml's
# [feature.*] blocks) each hardware hash actually runs, so
# `scripts/what_to_rerun.py --hardware` can drop envs that can't even be
# installed there (see `sklbench.reporting.envs.env_platforms`). This is
# genuinely new information (which OS/arch each physical machine runs) that
# isn't derivable from anywhere else in the repo.
HARDWARE_PLATFORMS = {
    "3b5e61": "linux-64",
    "534824": "linux-64",
    "be1055": "linux-64",
    "b281b2": "osx-arm64",
    "5dcf30": "win-64",
}

# `SOURCE_CONFIGS`/`SOURCE_ENVS` shared by the three general-comparison
# dashboards (gen_softwares_comparison, gen_builds_comparison,
# gen_hardware_comparisons), which all draw on the same broad
# configs/all_models.py matrix across every Pixi env it sweeps - kept here
# once rather than tripled across those modules. configs/smoke_check_test.py
# is deliberately excluded - it's a CI-only sanity config, not meant to
# produce dashboard-worthy results.
GENERAL_SOURCE_CONFIGS = [
    "configs/all_models.py",
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
