"""Write a manifest of this run's `sklearn-dev@...:main` result files under
./results/ - lets a future PR-comparison run download just those files from
the previously-deployed site instead of re-benchmarking scikit-learn `main`
(the "[skip main]" commit-message flag, see
../.github/workflows/pr-comparison.yml and COMPARISONS_PR.md).

Only files belonging to a `...:main` build are listed, never the PR
branch's own (which changes every push and would otherwise get reused as a
stale second build by pr_comparison_dashboard.py, which expects exactly one
base + one variant build per env).

Must be run with `results/` (relative to cwd) as the same benchmark results
directory pr_comparison_dashboard.py reads - see that script's module
docstring and .github/workflows/pr-comparison.yml's RESULTS_DIR comment.
"""

import argparse
from pathlib import Path

from sklbench.reporting.envs import is_sklearn_dev_build, software_build_name
from sklbench.reporting.matching import read_benchmark_records

RESULTS_DIR = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = read_benchmark_records()

    main_software_hashes = set()
    for record in records:
        build = software_build_name(record.software_hash)
        if is_sklearn_dev_build(build) and build.rsplit(":", 1)[-1] == "main":
            main_software_hashes.add(record.software_hash)

    relpaths = set()
    hardware_hashes = set()
    for record in records:
        if record.software_hash not in main_software_hashes:
            continue
        hardware_hashes.add(record.hardware_hash)
        relpaths.add(str(record.record_path.relative_to(RESULTS_DIR)))
        if record.profile_path is not None:
            relpaths.add(str(record.profile_path.relative_to(RESULTS_DIR)))

    for software_hash in main_software_hashes:
        relpaths.add(f"software-envs/{software_hash}.json")
    for hardware_hash in hardware_hashes:
        relpaths.add(f"hardware-envs/{hardware_hash}.json")

    Path(args.output).write_text("".join(f"{p}\n" for p in sorted(relpaths)))


if __name__ == "__main__":
    main()
