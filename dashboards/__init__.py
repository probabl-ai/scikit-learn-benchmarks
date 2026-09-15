"""Dashboard generator entry points."""

from argparse import ArgumentParser
from pathlib import Path


# Single source of truth for hardware-hash -> display name, shared by every
# dashboard so a rename doesn't have to be repeated file by file. Names are
# meant to be readable by non-hardware-specialists (relative age/power),
# not model numbers or vendor codenames.
HARDWARE_NAMES = {
    "3b5e61": "Modern Intel laptop with GPU",
    "534824": "High-end Intel server",
    "be1055": "Low-end Intel laptop",
}


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


def dashboard_output_path(default_filename: str) -> Path:
    return dashboard_output_dir() / default_filename
