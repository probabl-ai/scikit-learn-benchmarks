"""What the speed-up dashboards plot, from small hand-written results.

Each test writes benchmark records with known times, runs a real dashboard
generator, then runs `speedup_core.js` in Node on what it generated, like the page
does in the browser. These check which results get paired, the speed-up values, and
the markers and columns. UI details are left out on purpose.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from dashboards import (
    gen_builds_comparison,
    gen_cross_comparison,
    gen_hardware_comparisons,
    gen_softwares_comparison,
)


LAPTOP, XEON, DESKTOP, M4, L4 = "3b5e61", "534824", "332097", "b281b2", "d189c4"
LAPTOP_NAME = "Intel Core Ultra laptop"
XEON_NAME = "High-end Intel Xeon server"
L4_CPU_NAME = "L4 VM (8 CPU, 32 GB RAM)"
L4_GPU_NAME = "NVIDIA L4 g6.2xlarge"

# Software hash -> pixi env. Fixed across tests, since env lookups can be cached by hash.
ENVS = {
    "pypi00": "sklearn-pypi",
    "pypi01": "sklearn-pypi",
    "mkl000": "sklearn-cf-mkl",
    "cfdef0": "sklearn-cf-default",
    "intel0": "intel",
    "torch0": "skl-cpu",
    "cuml00": "cuml",
}
SKLEARN = {"software": "pypi00", "implementation": {"library": "sklearn"}}
MKL = {"software": "mkl000", "implementation": {"library": "sklearn"}}
CF_DEFAULT = {"software": "cfdef0", "implementation": {"library": "sklearn"}}
SKLEARNEX = {"software": "intel0", "implementation": {"library": "sklearnex", "device": "cpu"}}
CUML = {"software": "cuml00", "implementation": {"library": "cuml", "device": "cuda"}}
TORCH_CPU = {
    "software": "torch0",
    "implementation": {"library": "sklearn", "device": "cpu", "data_library": "torch"},
}
TORCH_XPU = {
    "software": "torch0",
    "implementation": {"library": "sklearn", "device": "xpu", "data_library": "torch"},
}
TORCH_CUDA = {
    "software": "torch0",
    "implementation": {"library": "sklearn", "device": "cuda", "data_library": "torch"},
}

N_SAMPLES = 1000
SYNTHETIC = {
    "source": "make_classification",
    "generation_kwargs": {"n_samples": N_SAMPLES, "n_features": 20},
}
REAL = {"dataset": "covtype", "preprocessing_kind": "trees"}
# sklearnex tree cases come in two flavors, see configs/_synthetic_trees.py.
EXACT = {"max_bins": N_SAMPLES}
FOREST = "RandomForestClassifier"
BINNING_WARNING = "histogram-based"
CUML_WARNING = "quantile-based"
GREY = "#9e9e9e"


@dataclass(frozen=True)
class Point:
    name: str
    x: float
    y: float
    symbol: str
    color: str
    text: str

    @property
    def speedup(self) -> float:
        return 2**self.y


class Results:
    """Writes records and env files in `root`, laid out like the repo's `results/`."""

    def __init__(self, root: Path):
        self.root = root
        self.count = 0
        for kind in ("records", "software-envs", "hardware-envs"):
            (root / "results" / kind).mkdir(parents=True)
        for software_hash, name in ENVS.items():
            env = {"pixi_environment_name": name, "pixi_packages": {"python": {"version": "3.14"}}}
            self._write_env("software", software_hash, env)
        for hardware_hash in (LAPTOP, XEON, DESKTOP, M4, L4):
            self._write_env("hardware", hardware_hash, {"CPU": {"name": "test CPU"}})

    def _write_env(self, kind, env_hash, env):
        (self.root / "results" / f"{kind}-envs" / f"{env_hash}.json").write_text(json.dumps(env))

    def _write(self, hardware, software, case, runs, age, failed_case=None):
        self.count += 1
        # Older records get earlier timestamps. The count keeps file names unique.
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=-age, microseconds=self.count
        )
        name = f"record_{timestamp.strftime('%Y%m%dT%H%M%S%f')}Z.json"
        (self.root / "results" / "records" / name).write_text(json.dumps({
            "hardware_hash": hardware,
            "software_hash": software,
            "case": case,
            "results": runs,
            "failed_case": failed_case,
        }))

    def _case(self, variant, estimator, params, data, config="configs/all_models.py", **extra):
        return {
            "metadata": {"source_config": config},
            "algorithm": {"estimator": estimator, "estimator_params": params or {}},
            "data": data,
            "implementation": variant["implementation"],
            **extra,
        }

    def add(
        self,
        hardware,
        variant,
        fit=(100,),
        predict=(100,),
        *,
        estimator="Ridge",
        params=None,
        data=SYNTHETIC,
        preprocessing=None,
        metrics=None,
        attributes=None,
        age=0,
    ):
        """`fit`, `predict` and `preprocessing` are times in ms, one per repeat."""
        n_samples = data.get("generation_kwargs", {}).get("n_samples", 5000)
        desc = {"samples": n_samples, "features": 20}
        runs = []
        for index, fit_time in enumerate(fit):
            times = {"fit": fit_time, "predict": predict[min(index, len(predict) - 1)]}
            if preprocessing is not None:
                times["preprocessing"] = preprocessing[index]
            runs.append({
                "time_ms": times,
                "data_desc": dict.fromkeys(times, desc),
                "metrics": metrics or {},
                "attributes": attributes or {},
            })
        case = self._case(variant, estimator, params, data)
        self._write(hardware, variant["software"], case, runs, age)

    def add_failed(self, hardware, variant, *, estimator="Ridge", params=None, data=SYNTHETIC, age=0):
        case = self._case(variant, estimator, params, data)
        self._write(hardware, variant["software"], case, [], age, failed_case={"return_code": 1})

    def add_search(self, hardware, variant, n_jobs, seconds, n_fits=100):
        """A hyper-parameter search taking `seconds` overall."""
        case = self._case(
            variant,
            FOREST,
            {},
            SYNTHETIC,
            config="configs/hptuning.py",
            hptuning={"n_jobs": n_jobs, "n_splits": n_jobs, "candidates": [{"a": 1}]},
        )
        run = {"duration_s": seconds, "cv_task_times": [1] * n_fits}
        self._write(hardware, variant["software"], case, [run], 0)


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Results(tmp_path)


PAGES = {
    "software": (gen_softwares_comparison, "per_hardware.html"),
    "builds": (gen_builds_comparison, "builds_comparison.html"),
    "hardware": (gen_hardware_comparisons, "hardware_comparisons.html"),
    "cross": (gen_cross_comparison, "cross_comparison.html"),
}


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is required to test the dashboards")
        pytest.skip("node is not installed")
    return node


def render(page: str, views: list[dict]) -> tuple[list[dict], dict]:
    """Generates `page` from the results of the current directory, then computes
    `views` like the page would. Returns them with the page's config."""
    module, file_name = PAGES[page]
    node = _node()
    output_dir = Path("site")
    output_dir.mkdir(exist_ok=True)
    module.generate(output_dir.resolve())
    html = (output_dir / file_name).read_text()
    config = json.loads(
        re.search(r'id="comparison-config">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</")
    )
    runner_input = {
        "corePath": str(files("sklbench.reporting.html").joinpath("speedup_core.js")),
        "dataDir": str(output_dir / config["dataDir"]),
        "config": config,
        "views": views,
    }
    process = subprocess.run(
        [node, str(Path(__file__).with_name("speedup_core_runner.js"))],
        input=json.dumps(runner_input),
        capture_output=True,
        encoding="utf-8",
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout), config


def view(page: str, **spec) -> dict:
    spec.setdefault("kind", "cross" if page == "cross" else "tabs")
    views, _ = render(page, [spec])
    return views[0]


def cell(rendered: dict, category: str, method: str = "fit") -> dict:
    row = next(row for row in rendered["grid"]["rows"] if row["category"] == category)
    return next(cell for cell in row["cells"] if cell["method"] == method)


def table(rendered: dict, category: str) -> list[dict]:
    row = next(row for row in rendered["grid"]["rows"] if row["category"] == category)
    return row["table"]["rows"] if row["table"] else []


def plot_points(plot: dict | None) -> list[Point]:
    points = []
    if plot is None:
        return points
    for trace in plot["traces"]:
        if trace["x"][0] is None:  # legend entry
            continue
        for index, x in enumerate(trace["x"]):
            symbol, color = trace["marker"]["symbol"], trace["marker"]["color"]
            points.append(Point(
                name=trace["name"],
                x=x,
                y=trace["y"][index],
                symbol=symbol[index] if isinstance(symbol, list) else symbol,
                color=color[index] if isinstance(color, list) else color,
                text=trace["text"][index],
            ))
    return points


def points(rendered: dict, category: str = "linear", method: str = "fit") -> list[Point]:
    return plot_points(cell(rendered, category, method).get("plot"))


def only(items: list):
    assert len(items) == 1, items
    return items[0]


def by_time(points_list: list[Point], baseline_ms: float) -> dict[float, Point]:
    """Points by their candidate time, read from the speed-up against a baseline."""
    return {round(baseline_ms / point.speedup): point for point in points_list}


# --- Speed-up values ---


def test_speedup_is_baseline_time_over_candidate_time(results):
    results.add(LAPTOP, SKLEARN, fit=[100], predict=[10])
    results.add(LAPTOP, SKLEARNEX, fit=[50], predict=[40])

    rendered = view("software", tab=LAPTOP_NAME)
    assert only(points(rendered, method="fit")).speedup == pytest.approx(2)
    assert only(points(rendered, method="predict")).speedup == pytest.approx(0.25)

    flipped = view("software", tab=LAPTOP_NAME, baseline="sklearnex-cpu")
    assert only(points(flipped, method="fit")).speedup == pytest.approx(0.5)
    assert only(points(flipped, method="predict")).speedup == pytest.approx(4)


def test_speedup_uses_the_median_of_repeats(results):
    results.add(LAPTOP, SKLEARN, fit=[100, 100, 900])
    results.add(LAPTOP, SKLEARNEX, fit=[50, 50, 1])

    assert only(points(view("software", tab=LAPTOP_NAME))).speedup == pytest.approx(2)


def test_table_and_plot_agree(results):
    results.add(LAPTOP, SKLEARN, fit=[100], predict=[10])
    results.add(LAPTOP, SKLEARNEX, fit=[50], predict=[40])

    rows = {row["variant"]: row for row in table(view("software", tab=LAPTOP_NAME), "linear")}
    assert rows["sklearn-pypi"]["is_baseline"]
    assert rows["sklearn-pypi"]["fit_speedup"] == pytest.approx(1)
    assert rows["sklearnex-cpu"]["fit_time"] == pytest.approx(50)
    assert rows["sklearnex-cpu"]["fit_speedup"] == pytest.approx(2)
    assert rows["sklearnex-cpu"]["predict_speedup"] == pytest.approx(0.25)


def test_preprocessing_view_adds_preprocessing_to_fit_on_real_datasets_only(results):
    results.add(LAPTOP, SKLEARN, fit=[100], predict=[10], data=REAL, preprocessing=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50], predict=[20], data=REAL, preprocessing=[150])
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 2})
    results.add(LAPTOP, SKLEARNEX, fit=[10], params={"alpha": 2})

    default = view("software", tab=LAPTOP_NAME)
    assert sorted(point.speedup for point in points(default)) == pytest.approx([2, 10])

    rendered = view("software", tab=LAPTOP_NAME, preprocessing=True)
    assert only(points(rendered)).speedup == pytest.approx(1)
    assert only(points(rendered, method="predict")).speedup == pytest.approx(0.5)


# --- Which results get paired ---


def test_different_cases_are_not_paired(results):
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, params={"alpha": 2})

    assert points(view("software", tab=LAPTOP_NAME)) == []


def test_n_jobs_is_part_of_a_case_on_one_machine_but_not_across_machines(results):
    results.add(LAPTOP, SKLEARN, fit=[100], params={"n_jobs": 14})
    results.add(LAPTOP, SKLEARNEX, fit=[50], params={"n_jobs": 8})
    results.add(XEON, SKLEARN, fit=[50], params={"n_jobs": 155})

    assert points(view("software", tab=LAPTOP_NAME)) == []
    across_machines = view("hardware", tab="sklearn-pypi")
    assert only(points(across_machines)).speedup == pytest.approx(2)


def test_baseline_uses_its_latest_run(results):
    results.add(LAPTOP, {**SKLEARN, "software": "pypi01"}, fit=[400], age=100)
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])

    assert only(points(view("software", tab=LAPTOP_NAME))).speedup == pytest.approx(2)


def test_stale_runs_are_dropped_across_machines(results):
    results.add(LAPTOP, SKLEARN, fit=[400], params={"n_jobs": 14}, age=100)
    results.add(LAPTOP, SKLEARN, fit=[100], params={"n_jobs": -1})
    results.add(XEON, SKLEARN, fit=[900], params={"n_jobs": 86}, age=100)
    results.add(XEON, SKLEARN, fit=[50], params={"n_jobs": -1})

    assert only(points(view("hardware", tab="sklearn-pypi"))).speedup == pytest.approx(2)


# --- Across machines ---


def test_forest_times_are_rescaled_to_the_same_size(results):
    results.add(LAPTOP, SKLEARN, fit=[100], predict=[100], estimator=FOREST, params={"n_estimators": 100})
    results.add(XEON, SKLEARN, fit=[100], predict=[400], estimator=FOREST, params={"n_estimators": 200})

    rendered = view("hardware", tab="sklearn-pypi")
    assert only(points(rendered, "tree-based", "fit")).speedup == pytest.approx(2)
    assert only(points(rendered, "tree-based", "predict")).speedup == pytest.approx(0.5)


def test_preprocessing_time_is_not_rescaled_with_the_forest(results):
    forest = {"estimator": FOREST, "data": REAL}
    results.add(LAPTOP, SKLEARN, fit=[100], preprocessing=[100], params={"n_estimators": 100}, **forest)
    results.add(XEON, SKLEARN, fit=[100], preprocessing=[50], params={"n_estimators": 200}, **forest)

    rendered = view("hardware", tab="sklearn-pypi", preprocessing=True)
    assert only(points(rendered, "tree-based")).speedup == pytest.approx(2)


def test_metrics_of_different_forest_sizes_are_not_compared(results):
    metrics = lambda auc: {"predict": {"ROC AUC": auc}}
    forest = {"estimator": FOREST}
    results.add(LAPTOP, SKLEARN, params={"n_estimators": 100}, metrics=metrics(0.7), **forest)
    results.add(XEON, SKLEARN, params={"n_estimators": 200}, metrics=metrics(0.9), **forest)

    assert only(points(view("hardware", tab="sklearn-pypi"), "tree-based")).symbol == "circle"


def test_cpu_and_gpu_runs_of_a_library_are_separate_tabs(results):
    results.add(LAPTOP, TORCH_CPU, fit=[100])
    results.add(XEON, TORCH_CPU, fit=[50])
    results.add(LAPTOP, TORCH_XPU, fit=[100])
    results.add(L4, TORCH_CUDA, fit=[25])

    _, config = render("hardware", [])
    tabs = {tab["label"]: [variant["label"] for variant in tab["variants"]] for tab in config["outerTabs"]}
    assert tabs == {
        "sklearn-torch-cpu": [LAPTOP_NAME, XEON_NAME],
        "sklearn-torch-gpu": [f"{LAPTOP_NAME} GPU", L4_GPU_NAME],
    }
    gpu = only(points(view("hardware", tab="sklearn-torch-gpu")))
    assert (gpu.name, gpu.speedup) == (L4_GPU_NAME, pytest.approx(4))


def test_software_on_a_single_machine_gets_no_hardware_tab(results):
    results.add(L4, CUML)
    results.add(L4, SKLEARN)
    results.add(XEON, SKLEARN)

    _, config = render("hardware", [])
    assert [tab["label"] for tab in config["outerTabs"]] == ["sklearn-pypi"]


def test_search_compares_time_per_fit_at_each_machine_best_n_jobs(results):
    results.add(LAPTOP, SKLEARN)
    results.add(XEON, SKLEARN)
    results.add_search(LAPTOP, SKLEARN, n_jobs=2, seconds=100)
    results.add_search(LAPTOP, SKLEARN, n_jobs=4, seconds=40)
    results.add_search(XEON, SKLEARN, n_jobs=16, seconds=20)

    search = view("hardware", tab="sklearn-pypi")["search"]
    point = only(plot_points(search["plot"]))
    assert point.speedup == pytest.approx(2)  # 400 ms vs 200 ms per fit
    assert "outer n_jobs: 4 vs 16" in point.text


# --- Tree binning ---


def _binning_results(results):
    """On one machine, with times telling the cases apart."""
    forest = {"estimator": FOREST}
    results.add(L4, SKLEARN, fit=[1000], **forest)
    results.add(L4, SKLEARNEX, fit=[500], params=EXACT, **forest)
    results.add(L4, SKLEARNEX, fit=[200], **forest)
    results.add(L4, CUML, fit=[100], **forest)


def test_sklearn_vs_sklearnex_has_a_binned_column_with_squares_and_an_exact_one(results):
    _binning_results(results)

    rendered = view("software", tab=L4_CPU_NAME)
    sklearnex = [point for point in points(rendered, "tree-based") if point.name == "sklearnex-cpu"]
    by_speedup = by_time(sklearnex, 1000)
    exact, binned = by_speedup[500], by_speedup[200]
    assert exact.symbol == "circle" and BINNING_WARNING not in exact.text
    assert binned.symbol == "square" and BINNING_WARNING in binned.text
    assert exact.x < binned.x


def test_sklearnex_baseline_vs_sklearn_has_the_same_two_columns(results):
    _binning_results(results)

    rendered = view("software", tab=L4_CPU_NAME, baseline="sklearnex-cpu")
    sklearn = [point for point in points(rendered, "tree-based") if point.name == "sklearn-pypi"]
    assert len(sklearn) == 2
    exact, binned = sorted(sklearn, key=lambda point: point.x)
    assert exact.speedup == pytest.approx(0.5) and exact.symbol == "circle"
    assert binned.speedup == pytest.approx(0.2) and binned.symbol == "square"


@pytest.mark.parametrize("baseline, other", [("sklearn-pypi", "cuml-cuda"), ("cuml-cuda", "sklearn-pypi")])
def test_sklearn_vs_cuml_has_one_column_with_squares(results, baseline, other):
    _binning_results(results)

    rendered = view("software", tab=L4_CPU_NAME, baseline=baseline)
    point = only([point for point in points(rendered, "tree-based") if point.name == other])
    assert point.symbol == "square" and CUML_WARNING in point.text


def test_cuml_is_only_paired_with_binned_sklearnex_without_a_marker(results):
    _binning_results(results)

    from_cuml = view("software", tab=L4_CPU_NAME, baseline="cuml-cuda")
    point = only([point for point in points(from_cuml, "tree-based") if point.name == "sklearnex-cpu"])
    assert point.speedup == pytest.approx(0.5)  # the binned 200 ms, not the exact 500 ms
    assert point.symbol == "circle"
    assert CUML_WARNING not in point.text and BINNING_WARNING not in point.text

    from_sklearnex = view("software", tab=L4_CPU_NAME, baseline="sklearnex-cpu")
    point = only([point for point in points(from_sklearnex, "tree-based") if point.name == "cuml-cuda"])
    assert point.speedup == pytest.approx(2)
    assert point.symbol == "circle"


def test_failed_cuml_run_stays_in_cuml_column(results):
    forest = {"estimator": FOREST}
    results.add(L4, SKLEARN, fit=[1000], **forest)
    results.add(L4, CUML, fit=[100], **forest)
    results.add(L4, SKLEARN, fit=[1000], params={"max_depth": 3}, **forest)
    results.add_failed(L4, CUML, params={"max_depth": 3}, **forest)

    tree_points = [
        point for point in points(view("software", tab=L4_CPU_NAME), "tree-based")
        if point.name == "cuml-cuda"
    ]
    failed = only([point for point in tree_points if point.symbol == "x"])
    plotted = only([point for point in tree_points if point.symbol != "x"])
    assert failed.x == pytest.approx(plotted.x)


def test_exact_sklearnex_failure_is_left_out_against_cuml(results):
    forest = {"estimator": FOREST}
    results.add(L4, CUML, fit=[100], **forest)
    results.add(L4, SKLEARNEX, fit=[200], **forest)
    results.add_failed(L4, SKLEARNEX, params=EXACT, **forest)

    rendered = view("software", tab=L4_CPU_NAME, baseline="cuml-cuda")
    assert only(points(rendered, "tree-based")).symbol == "circle"


def test_sklearnex_across_machines_pairs_each_split_kind_with_itself(results):
    forest = {"estimator": FOREST}
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **forest)
    results.add(XEON, SKLEARNEX, fit=[50], params=EXACT, **forest)
    results.add(XEON, SKLEARNEX, fit=[25], **forest)

    rendered = view("hardware", tab="sklearnex-cpu")
    by_speedup = {round(point.speedup): point for point in points(rendered, "tree-based")}
    exact, binned = by_speedup[2], by_speedup[4]
    assert exact.x < binned.x
    assert exact.symbol == binned.symbol == "circle"


# --- Markers ---


def test_metric_difference_gives_a_diamond_within_tolerance_a_circle(results):
    metrics = lambda auc: {"predict": {"ROC AUC": auc}}
    results.add(LAPTOP, SKLEARN, metrics=metrics(0.8))
    results.add(LAPTOP, SKLEARNEX, metrics=metrics(0.9))
    results.add(LAPTOP, SKLEARN, params={"alpha": 2}, metrics=metrics(0.8))
    results.add(LAPTOP, SKLEARNEX, params={"alpha": 2}, metrics=metrics(0.805))

    symbols = sorted(point.symbol for point in points(view("software", tab=LAPTOP_NAME)))
    assert symbols == ["circle", "diamond-open"]


@pytest.mark.parametrize(
    "base_attributes, attributes, expected",
    [
        ({"solver": "lbfgs"}, {"solver": "newton-cg"}, "(lbfgs vs newton-cg)"),
        ({"n_iter": 10}, {"n_iter": 100}, "(10 vs 100)"),
    ],
)
def test_setup_difference_gives_a_square(results, base_attributes, attributes, expected):
    results.add(LAPTOP, SKLEARN, attributes=base_attributes)
    results.add(LAPTOP, SKLEARNEX, attributes=attributes)

    point = only(points(view("software", tab=LAPTOP_NAME)))
    assert point.symbol == "square" and expected in point.text


def test_sklearnex_fallback_is_grey_without_warnings(results):
    results.add(LAPTOP, SKLEARN, estimator=FOREST, attributes={"n_iter": 10})
    fallback = {"has_onedal_estimator": False, "n_iter": 100}
    results.add(LAPTOP, SKLEARNEX, estimator=FOREST, attributes=fallback)

    point = only(points(view("software", tab=LAPTOP_NAME), "tree-based"))
    assert point.color == GREY and point.symbol == "circle"
    assert "fell back" in point.text and BINNING_WARNING not in point.text


# --- Failed runs ---


def test_failed_candidate_is_below_its_column_and_keeps_the_baseline_row(results):
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, fit=[50], params={"alpha": 1})
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 2})
    results.add_failed(LAPTOP, SKLEARNEX, params={"alpha": 2})

    rendered = view("software", tab=LAPTOP_NAME)
    failed = only([point for point in points(rendered) if point.symbol == "x"])
    plotted = only([point for point in points(rendered) if point.symbol != "x"])
    assert failed.x == pytest.approx(plotted.x)
    assert failed.y < plotted.y

    rows = table(rendered, "linear")
    assert sorted(row["status"] for row in rows if row["variant"] == "sklearnex-cpu") == ["failed", "ok"]
    baseline_rows = [row for row in rows if row["variant"] == "sklearn-pypi"]
    assert len(baseline_rows) == 2
    assert sorted(str(row["fit_speedup"]) for row in baseline_rows) == ["1", "None"]


def test_failed_baseline_is_not_plotted_but_shows_in_the_table(results):
    results.add_failed(LAPTOP, SKLEARN, params={"alpha": 2})
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, fit=[77], params={"alpha": 2})

    rendered = view("software", tab=LAPTOP_NAME)
    assert only(points(rendered)).symbol == "circle"
    rows = table(rendered, "linear")
    assert [row["status"] for row in rows if row["status"] != "ok"] == ["failed"]
    candidate = next(row for row in rows if row.get("fit_time") == 77)
    assert candidate["fit_speedup"] is None


def test_failure_superseded_by_a_success_is_dropped_across_machines(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(XEON, SKLEARN, fit=[50])
    results.add_failed(XEON, SKLEARN, age=100)

    assert only(points(view("hardware", tab="sklearn-pypi"))).symbol == "circle"


# --- Layout ---


def test_columns_stay_on_their_estimator_with_many_columns(results):
    forest = {"estimator": FOREST}
    other_forest = {"estimator": "RandomForestRegressor"}
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **other_forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **other_forest)
    for hardware in (LAPTOP, XEON, DESKTOP, M4, L4):
        results.add(hardware, SKLEARN, fit=[25], **forest)
        results.add(hardware, SKLEARN, fit=[10], **other_forest)

    rendered = view("cross", baseline=f"sklearnex-cpu__{LAPTOP}", target="software-sklearn-pypi")
    tree_cell = cell(rendered, "tree-based")
    assert tree_cell["plot"]["layout"]["xaxis"]["ticktext"] == ["RFClf", "RFReg"]
    all_points = plot_points(tree_cell["plot"])
    assert len(all_points) == 20
    assert len({round(point.x - round(point.x), 4) for point in all_points}) == 10
    for point in all_points:
        expected_tick = 0 if point.speedup == pytest.approx(4) else 1
        assert round(point.x) == expected_tick, point


# --- Each page's variants ---


def test_cross_page_compares_one_software_on_every_machine(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(XEON, SKLEARN, fit=[50])
    results.add(L4, SKLEARN, fit=[25])
    results.add(L4, CUML, fit=[10])

    rendered = view("cross", baseline=f"sklearn-pypi__{LAPTOP}", target="software-sklearn-pypi")
    assert rendered["baselineLabel"] == f"sklearn-pypi on {LAPTOP_NAME}"
    speedups = {point.name: point.speedup for point in points(rendered)}
    assert speedups == {XEON_NAME: pytest.approx(2), L4_CPU_NAME: pytest.approx(4)}


def test_cross_page_compares_every_software_on_one_machine(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(XEON, SKLEARN, fit=[50])
    results.add(L4, SKLEARN, fit=[25])
    results.add(L4, CUML, fit=[10])

    rendered = view("cross", baseline=f"cuml-gpu__{L4}", target=f"hardware-{L4}")
    assert rendered["baselineLabel"] == f"cuml-gpu on {L4_GPU_NAME}"
    assert only(points(rendered)).name == "sklearn-pypi"
    assert only(points(rendered)).speedup == pytest.approx(0.4)


def test_cross_page_only_offers_targets_sharing_a_case(results):
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(XEON, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, MKL, params={"alpha": 2})

    _, config = render("cross", [])
    assert config["pages"][f"sklearn-pypi__{LAPTOP}"] == ["software-sklearn-pypi", f"hardware-{XEON}"]
    assert f"sklearn-cf-mkl__{LAPTOP}" not in config["pages"]


def test_software_page_offers_mkl_as_baseline_but_does_not_plot_it(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])
    results.add(LAPTOP, MKL, fit=[25])

    default = view("software", tab=LAPTOP_NAME)
    assert only(points(default)).name == "sklearnex-cpu"

    from_mkl = view("software", tab=LAPTOP_NAME, baseline="sklearn-cf-mkl")
    speedups = {point.name: point.speedup for point in points(from_mkl)}
    assert speedups == {"sklearn-pypi": pytest.approx(0.25), "sklearnex-cpu": pytest.approx(0.5)}


def test_builds_page_compares_scikit_learn_builds_with_pypi_only(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, MKL, fit=[50])
    results.add(LAPTOP, CF_DEFAULT, fit=[25])
    results.add(LAPTOP, SKLEARNEX, fit=[10])
    results.add(LAPTOP, TORCH_CPU, fit=[10])

    rendered = view("builds", tab=LAPTOP_NAME)
    speedups = {point.name: point.speedup for point in points(rendered)}
    assert speedups == {"sklearn-cf-mkl": pytest.approx(2), "sklearn-cf-default": pytest.approx(4)}
    assert rendered["baselineLabel"] == "sklearn-pypi"


def test_hardware_page_keeps_mkl_but_not_the_other_conda_forge_builds(results):
    for variant in (SKLEARN, MKL, CF_DEFAULT):
        results.add(LAPTOP, variant)
        results.add(XEON, variant)

    _, config = render("hardware", [])
    assert [tab["label"] for tab in config["outerTabs"]] == ["sklearn-pypi", "sklearn-cf-mkl"]
