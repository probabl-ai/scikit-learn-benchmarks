"""
What to rerun to refresh a dashboard (or all of them), and the reverse: which
dashboards a given config feeds. Reads `dashboards/gen_*.py`'s declared
`SOURCE_CONFIGS`/`SOURCE_ENVS` (see `dashboards/index.py`'s `DASHBOARDS`
list) instead of anyone having to read `sklbench/reporting/matching.py`'s
filtering by hand.

    python scripts/what_to_rerun.py --dashboard "HistGradientBoosting thread-scalability breakdown"
    python scripts/what_to_rerun.py --dashboard all
    python scripts/what_to_rerun.py --dashboard all --hardware "Modern Intel laptop"
    python scripts/what_to_rerun.py --config configs/hgb_scalability.py

`--hardware` (a `HARDWARE_NAMES` display name or hash) narrows the printed
envs to those actually installable there, per `sklbench.reporting.envs.
env_platforms()` and `dashboards.HARDWARE_PLATFORMS` - e.g. `skl-mps`/`intel`
never get suggested for hardware that can't run them.

A module may also declare `SOURCE_CONFIG_OVERRIDES` (see
`dashboards/gen_hgb_scalability_breakdown.py` for an example) to restrict one
specific config to certain hardware and/or a narrower env list than the
dashboard's own `SOURCE_ENVS` - e.g. a config that only makes sense on
multi-socket server hardware, or only under one build family. A config
absent from `SOURCE_CONFIG_OVERRIDES` has no extra restriction.
"""

import argparse

from dashboards import HARDWARE_NAMES, HARDWARE_PLATFORMS
from dashboards.index import DASHBOARDS
from sklbench.config.registry import to_repo_relative
from sklbench.reporting.envs import env_platforms


def _resolve_hardware(hardware_arg: str) -> str:
    if hardware_arg in HARDWARE_NAMES:
        return hardware_arg
    lowered = hardware_arg.lower()
    for hardware_hash, name in HARDWARE_NAMES.items():
        if name.lower() == lowered:
            return hardware_hash
    known = ", ".join(f'"{name}"' for name in HARDWARE_NAMES.values())
    raise SystemExit(f"Unknown hardware {hardware_arg!r}. Known: {known}")


def _dashboards_by_label(label: str):
    if label.lower() == "all":
        return list(DASHBOARDS)
    matches = [entry for entry in DASHBOARDS if entry[0].lower() == label.lower()]
    if not matches:
        known = ", ".join(f'"{entry[0]}"' for entry in DASHBOARDS)
        raise SystemExit(f"Unknown dashboard {label!r}. Known: {known}")
    return matches


def _envs_for_hardware(source_envs: list[str], hardware_hash: str | None) -> list[str]:
    if hardware_hash is None:
        return list(source_envs)
    platform = HARDWARE_PLATFORMS[hardware_hash]
    platforms = env_platforms()
    return [env for env in source_envs if platform in platforms.get(env, frozenset())]


def _config_envs_and_note(
    module, config: str, hardware_hash: str | None
) -> tuple[list[str] | None, str | None]:
    """envs to run `config` under for `hardware_hash` (None = unfiltered),
    honoring the module's optional `SOURCE_CONFIG_OVERRIDES` for this config.
    Returns (None, None) if `config` doesn't apply to `hardware_hash` at all.
    The second element is a human note to print when a hardware restriction
    exists but no specific `--hardware` was given to check it against."""
    override = getattr(module, "SOURCE_CONFIG_OVERRIDES", {}).get(config, {})
    allowed_hardware = override.get("hardware")
    note = None
    if allowed_hardware is not None:
        if hardware_hash is not None:
            if hardware_hash not in allowed_hardware:
                return None, None
        else:
            names = ", ".join(
                HARDWARE_NAMES.get(h, h) for h in sorted(allowed_hardware)
            )
            note = f"only meaningful on: {names}"
    config_envs = override.get("envs", module.SOURCE_ENVS)
    return _envs_for_hardware(config_envs, hardware_hash), note


def _print_rerun_commands(dashboards, hardware_hash: str | None) -> None:
    # config -> envs to run it under, unioned across every dashboard that
    # shares it, in first-seen order (so ./run.sh gets one deduped call).
    envs_by_config: dict[str, list[str]] = {}
    notes_by_config: dict[str, str] = {}
    for _, module, _ in dashboards:
        for config in module.SOURCE_CONFIGS:
            envs, note = _config_envs_and_note(module, config, hardware_hash)
            if envs is None:
                continue
            existing = envs_by_config.setdefault(config, [])
            for env in envs:
                if env not in existing:
                    existing.append(env)
            if note:
                notes_by_config[config] = note

    for config, envs in envs_by_config.items():
        if note := notes_by_config.get(config):
            print(f"# {config}: {note}")
        if not envs:
            print(f"# no envs available for {config} on this hardware")
            continue
        print(f"./run.sh {' '.join(envs)} --config {config}")


def _print_dashboards_for_config(config_arg: str) -> None:
    config = to_repo_relative(config_arg)
    matches = [
        label for label, module, _ in DASHBOARDS if config in module.SOURCE_CONFIGS
    ]
    if not matches:
        print(f"No dashboard declares {config} in SOURCE_CONFIGS.")
        return
    for label in matches:
        print(label)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dashboard",
        help='Dashboard label (see dashboards/index.py\'s DASHBOARDS), or "all".',
    )
    target.add_argument(
        "--config",
        help="Config path (e.g. configs/hgb_scalability.py) - "
        "prints the dashboards it feeds instead.",
    )
    parser.add_argument(
        "--hardware",
        help="HARDWARE_NAMES display name or hash - only meaningful with "
        "--dashboard, narrows envs to those installable there.",
    )
    args = parser.parse_args()

    if args.config is not None:
        if args.hardware is not None:
            raise SystemExit("--hardware only applies to --dashboard")
        _print_dashboards_for_config(args.config)
        return

    hardware_hash = _resolve_hardware(args.hardware) if args.hardware else None
    _print_rerun_commands(_dashboards_by_label(args.dashboard), hardware_hash)


if __name__ == "__main__":
    main()
