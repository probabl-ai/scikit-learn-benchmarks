"""
Visualisation script for scikit-learn-benchmarks.

Reads JSON results produced by sklbench and generates comparison plots
for sklearn vs scikit-learn-intelex (sklearnex) across different estimators
and dataset sizes.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_files: list[str]) -> list[dict]:
    """Load and merge results from one or more JSON result files."""
    all_results = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
    return all_results


def extract_metrics(results: list[dict]) -> dict:
    """
    Extract fit/inference time per (estimator, library, n_samples) from results.

    Returns a dict of the form:
        {
            (estimator, n_samples): {library: {"fit": time, "inference": time}},
            ...
        }
    """
    metrics: dict = {}
    for entry in results:
        algo = entry.get("algorithm", {})
        data = entry.get("data", {})
        bench = entry.get("bench", {})

        estimator = algo.get("estimator", "unknown")
        library = algo.get("library", "unknown")
        n_samples = data.get("generation_kwargs", {}).get("n_samples", "?")

        key = (estimator, n_samples)
        if key not in metrics:
            metrics[key] = {}

        lib_metrics: dict = {}
        for stage in ("fit", "inference"):
            time_key = f"{stage}_time"
            if time_key in bench:
                lib_metrics[stage] = bench[time_key]

        if lib_metrics:
            metrics[key][library] = lib_metrics

    return metrics


def speedup_bar_chart(
    metrics: dict,
    output_dir: Path,
    baseline: str = "sklearn",
    target: str = "sklearnex",
) -> list[Path]:
    """
    Generate speed-up bar charts (target / baseline) per estimator.

    Returns paths to all generated PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Group keys by estimator
    estimators: dict[str, list] = {}
    for estimator, n_samples in metrics:
        estimators.setdefault(estimator, []).append(n_samples)

    for estimator, sample_sizes in sorted(estimators.items()):
        sample_sizes = sorted(sample_sizes)
        stages = ["fit", "inference"]

        # Build speedup data
        speedups: dict[str, list[float | None]] = {s: [] for s in stages}
        labels: list[str] = []

        for n_samples in sample_sizes:
            key = (estimator, n_samples)
            entry = metrics[key]
            if baseline not in entry or target not in entry:
                continue

            labels.append(f"n={n_samples:,}")
            for stage in stages:
                base_time = entry[baseline].get(stage)
                tgt_time = entry[target].get(stage)
                if base_time and tgt_time and tgt_time > 0:
                    speedups[stage].append(base_time / tgt_time)
                else:
                    speedups[stage].append(None)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, stage in enumerate(stages):
            values = [v if v is not None else 0 for v in speedups[stage]]
            bars = ax.bar(x + i * width, values, width, label=stage.capitalize())
            for bar, val in zip(bars, speedups[stage]):
                if val is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        f"{val:.2f}×",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

        ax.axhline(y=1, color="grey", linestyle="--", linewidth=0.8, label="Baseline (1×)")
        ax.set_xlabel("Dataset size")
        ax.set_ylabel(f"Speed-up ({target} / {baseline})")
        ax.set_title(f"{estimator}: {target} vs {baseline} speed-up")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        out_path = output_dir / f"{estimator.lower()}_speedup.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        generated.append(out_path)
        print(f"Saved {out_path}")

    return generated


def absolute_time_chart(metrics: dict, output_dir: Path) -> list[Path]:
    """
    Generate absolute fit/inference time charts per estimator.

    Returns paths to all generated PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    estimators: dict[str, list] = {}
    for estimator, n_samples in metrics:
        estimators.setdefault(estimator, []).append(n_samples)

    for estimator, sample_sizes in sorted(estimators.items()):
        sample_sizes = sorted(sample_sizes)
        stages = ["fit", "inference"]

        all_libraries = sorted(
            {lib for (est, _), libs in metrics.items() if est == estimator for lib in libs}
        )
        if not all_libraries:
            continue

        fig, axes = plt.subplots(1, len(stages), figsize=(6 * len(stages), 5))
        if len(stages) == 1:
            axes = [axes]

        for ax, stage in zip(axes, stages):
            x = np.arange(len(sample_sizes))
            width = 0.8 / len(all_libraries)

            for i, library in enumerate(all_libraries):
                times = []
                for n_samples in sample_sizes:
                    key = (estimator, n_samples)
                    t = metrics[key].get(library, {}).get(stage)
                    times.append(t if t is not None else 0)

                ax.bar(x + i * width, times, width, label=library)

            ax.set_xlabel("Dataset size")
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{estimator} – {stage.capitalize()} time")
            ax.set_xticks(x + width * (len(all_libraries) - 1) / 2)
            ax.set_xticklabels([f"n={n:,}" for n in sample_sizes])
            ax.legend()

        plt.suptitle(estimator, fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = output_dir / f"{estimator.lower()}_times.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        generated.append(out_path)
        print(f"Saved {out_path}")

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualisations from sklbench result files."
    )
    parser.add_argument(
        "--result-files",
        nargs="+",
        required=True,
        help="Path(s) to sklbench JSON result file(s).",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write output PNG files (default: plots/).",
    )
    parser.add_argument(
        "--baseline",
        default="sklearn",
        help="Library to use as baseline for speed-up computation (default: sklearn).",
    )
    parser.add_argument(
        "--target",
        default="sklearnex",
        help="Library to compare against the baseline (default: sklearnex).",
    )
    args = parser.parse_args()

    results = load_results(args.result_files)
    if not results:
        print("No results found in the provided files.")
        return

    metrics = extract_metrics(results)
    output_dir = Path(args.output_dir)

    speedup_bar_chart(metrics, output_dir, baseline=args.baseline, target=args.target)
    absolute_time_chart(metrics, output_dir)


if __name__ == "__main__":
    main()
