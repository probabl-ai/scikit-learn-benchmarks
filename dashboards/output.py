from argparse import ArgumentParser
from pathlib import Path


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
