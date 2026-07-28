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
    return parser


def get_orchestrator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sklbench",
        description="Scikit-learn_bench orchestrator",
    )
    add_orchestrator_arguments(parser)
    return parser
