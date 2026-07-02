import argparse

import pandas as pd


def get_parser_description(parser: argparse.ArgumentParser) -> pd.DataFrame:
    def get_argument_actions(parser: argparse.ArgumentParser) -> list:
        arg_actions = []

        for action in parser._actions:
            if isinstance(action, argparse._ArgumentGroup):
                for subaction in action._group_actions:
                    arg_actions.append(subaction)
            else:
                arg_actions.append(action)
        return arg_actions

    def parse_action(action: argparse.Action) -> dict:
        return {
            "Name": "</br>".join(map(lambda x: f"`{x}`", action.option_strings)),
            "Type": action.type.__name__ if action.type is not None else None,
            "Default value": (
                action.default if action.default is not argparse.SUPPRESS else None
            ),
            "Choices": action.choices,
            "Description": action.help,
        }

    return pd.DataFrame(map(parse_action, get_argument_actions(parser))).to_markdown(
        index=False
    )


def add_orchestrator_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to a Python config script.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory path to store scikit-learn_bench results.",
    )
    parser.add_argument(
        "--prefetch-datasets",
        default=False,
        action="store_true",
        help="Load all requested datasets in parallel before running benchmarks.",
    )
    parser.add_argument(
        "--exit-on-error",
        default=False,
        action="store_true",
        help="Interrupt orchestrator and exit if last benchmark failed with error.",
    )
    parser.add_argument(
        "--describe-parser",
        default=False,
        action="store_true",
        help="Print parser description in Markdown table format and exit.",
    )
    return parser


def get_orchestrator_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sklbench",
        description="Scikit-learn_bench orchestrator",
    )
    add_orchestrator_arguments(parser)
    return parser
