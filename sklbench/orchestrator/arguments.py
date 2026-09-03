import argparse


def add_orchestrator_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to one or more Python config scripts.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory path to store scikit-learn_bench results.",
    )
    parser.add_argument(
        "--exit-on-error",
        default=False,
        action="store_true",
        help="Interrupt orchestrator and exit if last benchmark failed with error.",
    )
    parser.add_argument(
        "--load-datasets-only",
        default=False,
        action="store_true",
        help=(
            "Load (and cache) the dataset for every case from the given "
            "config(s) without running any benchmarks. Useful for "
            "pre-filling the dataset cache and for validating configs."
        ),
    )
    parser.add_argument(
        "--no-system-telemetry",
        dest="no_system_telemetry",
        default=False,
        action="store_true",
        help=(
            "Disable the background system telemetry sampler (CPU load, "
            "per-core frequency, temperature) that otherwise runs for the "
            "whole session and writes to results/system-telemetry/."
        ),
    )
    parser.add_argument(
        "--system-telemetry-interval",
        type=float,
        default=2.0,
        help="Seconds between system telemetry samples (default: 2.0).",
    )
    parser.add_argument(
        "--validate-only",
        default=False,
        action="store_true",
        help=(
            "Load and validate every case from the given config(s), then "
            "exit without loading datasets or running benchmarks. Fast "
            "sanity check for config changes (schema errors, bad env "
            "values, generate_cases() exceptions)."
        ),
    )
    return parser


def get_orchestrator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sklbench",
        description="Scikit-learn_bench orchestrator",
    )
    add_orchestrator_arguments(parser)
    return parser
