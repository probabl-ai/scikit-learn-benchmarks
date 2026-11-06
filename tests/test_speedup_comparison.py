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

from dashboards import gen_cross_comparison, gen_hardware_parallel_fits


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

LOGICAL_CPUS = 32
N_SAMPLES = 1000
SYNTHETIC = {
    "source": "make_classification",
    "generation_kwargs": {"n_samples": N_SAMPLES, "n_features": 20},
}
REAL = {"dataset": "covtype", "preprocessing_kind": "trees"}
# sklearnex tree cases come in two flavors, see configs/_synthetic_trees.py.
EXACT = {"max_bins": N_SAMPLES}
FOREST = "RandomForestClassifier"
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
            self._write_env("hardware", hardware_hash, {"CPU": {"name": "test CPU", "logical_cpus": LOGICAL_CPUS}})

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

    def _case(self, variant, estimator, params, data, config="configs/all_models.py", env=None, **extra):
        if env:
            extra["bench"] = {"env": env}
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
        env=None,
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
        case = self._case(variant, estimator, params, data, env=env)
        self._write(hardware, variant["software"], case, runs, age)

    def add_failed(self, hardware, variant, *, estimator="Ridge", params=None, data=SYNTHETIC, age=0):
        case = self._case(variant, estimator, params, data)
        self._write(hardware, variant["software"], case, [], age, failed_case={"return_code": 1})

    def add_search(self, hardware, variant, seconds, n_fits, mask=LOGICAL_CPUS):
        """A hyper-parameter search of `configs/hptuning_taskset.py` taking `seconds`
        overall, restricted to `mask` logical CPUs."""
        case = self._case(
            variant,
            FOREST,
            {},
            SYNTHETIC,
            config="configs/hptuning_taskset.py",
            hptuning={"n_jobs": -1, "n_splits": n_fits, "candidates": [{"a": 1}]},
            bench={"cpu_affinity": list(range(mask))},
        )
        self._write(hardware, variant["software"], case, [{"duration_s": seconds, "cv_task_times": [1] * n_fits}], 0)


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return Results(tmp_path)


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is required to test the dashboards")
        pytest.skip("node is not installed")
    return node


def render(views: list[dict], page=gen_cross_comparison) -> tuple[list[dict], dict]:
    """Generates the page from the results of the current directory, then computes
    `views` like the page would. Returns them with the page's config."""
    node = _node()
    output_dir = Path("site")
    output_dir.mkdir(exist_ok=True)
    page.generate(output_dir.resolve())
    html = (output_dir / page.ENTRY_PAGE).read_text()
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


def view(**spec) -> dict:
    views, _ = render([spec])
    return views[0]


def on_machine(hardware: str, baseline: str = "sklearn-pypi", **spec) -> dict:
    return view(baseline=f"{baseline}__{hardware}", target=f"hardware-{hardware}", **spec)


def across_machines(software: str = "sklearn-pypi", **spec) -> dict:
    return view(baseline=f"{software}__{LAPTOP}", target=f"software-{software}", **spec)


def cell(rendered: dict, category: str) -> dict:
    row = next(row for row in rendered["grid"]["rows"] if row["category"] == category)
    return only(row["cells"])


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


def points(rendered: dict, category: str = "linear") -> list[Point]:
    return plot_points(cell(rendered, category).get("plot"))


def only(items: list):
    assert len(items) == 1, items
    return items[0]


def by_time(points_list: list[Point], baseline_ms: float) -> dict[float, Point]:
    """Points by their candidate time, read from the speed-up against a baseline."""
    return {round(baseline_ms / point.speedup): point for point in points_list}


# --- Speed-up values ---


def test_speedup_is_baseline_time_over_candidate_time(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])

    assert only(points(on_machine(LAPTOP))).speedup == pytest.approx(2)
    assert only(points(on_machine(LAPTOP, baseline="sklearnex-cpu"))).speedup == pytest.approx(0.5)


def test_predict_times_are_left_out(results):
    results.add(LAPTOP, SKLEARN, fit=[100], predict=[10])
    results.add(LAPTOP, SKLEARNEX, fit=[50], predict=[40])

    rendered, config = render([{"baseline": f"sklearn-pypi__{LAPTOP}", "target": f"hardware-{LAPTOP}"}])
    assert only(points(rendered[0])).speedup == pytest.approx(2)
    fields = [spec["column"]["field"] for spec in config["columns"] if "column" in spec]
    assert "fit_time" in fields and not any(field.startswith("predict_") for field in fields)
    assert all(row.get("predict_time") is None for row in table(rendered[0], "linear"))


def test_speedup_uses_the_median_of_repeats(results):
    results.add(LAPTOP, SKLEARN, fit=[100, 100, 900])
    results.add(LAPTOP, SKLEARNEX, fit=[50, 50, 1])

    assert only(points(on_machine(LAPTOP))).speedup == pytest.approx(2)


def test_table_and_plot_agree(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])

    rendered = on_machine(LAPTOP)
    rows = {row["variant"]: row for row in table(rendered, "linear")}
    assert rows[rendered["baselineLabel"]]["is_baseline"]
    assert rows[rendered["baselineLabel"]]["fit_speedup"] == pytest.approx(1)
    assert rows["sklearnex-cpu"]["fit_time"] == pytest.approx(50)
    assert rows["sklearnex-cpu"]["fit_speedup"] == pytest.approx(2)


def test_preprocessing_view_adds_preprocessing_to_fit_on_real_datasets_only(results):
    results.add(LAPTOP, SKLEARN, fit=[100], data=REAL, preprocessing=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50], data=REAL, preprocessing=[150])
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 2})
    results.add(LAPTOP, SKLEARNEX, fit=[10], params={"alpha": 2})

    default = on_machine(LAPTOP)
    assert sorted(point.speedup for point in points(default)) == pytest.approx([2, 10])

    rendered = on_machine(LAPTOP, preprocessing=True)
    assert only(points(rendered)).speedup == pytest.approx(1)


# --- Which results get paired ---


def test_different_cases_are_not_paired(results):
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, params={"alpha": 2})
    results.add(XEON, SKLEARN, params={"alpha": 1})

    assert points(on_machine(LAPTOP)) == []


def test_n_jobs_is_not_part_of_a_case(results):
    results.add(LAPTOP, SKLEARN, fit=[100], params={"n_jobs": 14})
    results.add(XEON, SKLEARN, fit=[50], params={"n_jobs": 155})

    assert only(points(across_machines())).speedup == pytest.approx(2)


def test_baseline_uses_its_latest_run(results):
    results.add(LAPTOP, {**SKLEARN, "software": "pypi01"}, fit=[400], age=100)
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])

    assert only(points(on_machine(LAPTOP))).speedup == pytest.approx(2)


def test_stale_runs_are_dropped_across_machines(results):
    results.add(LAPTOP, SKLEARN, fit=[400], params={"n_jobs": 14}, age=100)
    results.add(LAPTOP, SKLEARN, fit=[100], params={"n_jobs": -1})
    results.add(XEON, SKLEARN, fit=[900], params={"n_jobs": 86}, age=100)
    results.add(XEON, SKLEARN, fit=[50], params={"n_jobs": -1})

    assert only(points(across_machines())).speedup == pytest.approx(2)


# --- Across machines ---


def test_thread_caps_of_many_core_machines_dont_split_cases_across_machines(results):
    capped = {"OMP_NUM_THREADS": "64", "OPENBLAS_NUM_THREADS": "64"}
    results.add(LAPTOP, SKLEARN, fit=[100], estimator="KMeans")
    results.add(XEON, SKLEARN, fit=[900], estimator="KMeans", env={"OMP_NUM_THREADS": "128"}, age=100)
    results.add(XEON, SKLEARN, fit=[50], estimator="KMeans", env=capped)

    rendered = across_machines()
    assert only(points(rendered, "clustering")).speedup == pytest.approx(2)


def test_other_env_vars_still_split_cases_across_machines(results):
    results.add(LAPTOP, SKLEARN, env={"GOMP_SPINCOUNT": "0"})
    results.add(XEON, SKLEARN, env={"GOMP_SPINCOUNT": "300000"})
    results.add(LAPTOP, SKLEARNEX, env={"GOMP_SPINCOUNT": "0"})

    assert points(across_machines()) == []


def test_thread_caps_dont_split_cases_on_one_machine(results):
    capped = {"OMP_NUM_THREADS": "64", "OPENBLAS_NUM_THREADS": "64"}
    results.add(XEON, SKLEARN, fit=[100], estimator="KMeans", env=capped)
    results.add(XEON, SKLEARNEX, fit=[900], estimator="KMeans", env=capped, age=100)
    results.add(XEON, SKLEARNEX, fit=[50], estimator="KMeans")

    rendered = on_machine(XEON)
    assert only(points(rendered, "clustering")).speedup == pytest.approx(2)


def test_other_env_vars_still_split_cases_on_one_machine(results):
    results.add(XEON, SKLEARN, estimator="KMeans")
    results.add(XEON, SKLEARNEX, estimator="KMeans", env={"GOMP_SPINCOUNT": "0"})
    results.add(LAPTOP, SKLEARN, estimator="KMeans")

    assert points(on_machine(XEON), "clustering") == []


def test_forest_times_are_rescaled_to_the_same_size(results):
    results.add(LAPTOP, SKLEARN, fit=[100], estimator=FOREST, params={"n_estimators": 100})
    results.add(XEON, SKLEARN, fit=[100], estimator=FOREST, params={"n_estimators": 200})

    rendered = across_machines()
    assert only(points(rendered, "tree-based")).speedup == pytest.approx(2)


def test_preprocessing_time_is_not_rescaled_with_the_forest(results):
    forest = {"estimator": FOREST, "data": REAL}
    results.add(LAPTOP, SKLEARN, fit=[100], preprocessing=[100], params={"n_estimators": 100}, **forest)
    results.add(XEON, SKLEARN, fit=[100], preprocessing=[50], params={"n_estimators": 200}, **forest)

    rendered = across_machines(preprocessing=True)
    assert only(points(rendered, "tree-based")).speedup == pytest.approx(2)


def test_metrics_of_different_forest_sizes_are_not_compared(results):
    metrics = lambda auc: {"predict": {"ROC AUC": auc}}
    forest = {"estimator": FOREST}
    results.add(LAPTOP, SKLEARN, params={"n_estimators": 100}, metrics=metrics(0.7), **forest)
    results.add(XEON, SKLEARN, params={"n_estimators": 200}, metrics=metrics(0.9), **forest)

    assert only(points(across_machines(), "tree-based")).symbol == "circle"


def test_cpu_and_gpu_runs_of_a_library_are_separate_softwares(results):
    results.add(LAPTOP, TORCH_CPU, fit=[100])
    results.add(XEON, TORCH_CPU, fit=[50])
    results.add(LAPTOP, TORCH_XPU, fit=[100])
    results.add(L4, TORCH_CUDA, fit=[25])

    _, config = render([])
    machines = lambda software: {
        variant["hardwareLabel"] for variant in config["variants"].values() if variant["software"] == software
    }
    assert machines("sklearn-torch-cpu") == {LAPTOP_NAME, XEON_NAME}
    assert machines("sklearn-torch-gpu") == {f"{LAPTOP_NAME} GPU", L4_GPU_NAME}
    gpu = only(points(across_machines("sklearn-torch-gpu")))
    assert (gpu.name, gpu.speedup) == (L4_GPU_NAME, pytest.approx(4))


# --- Parallelized fits ---


def test_parallel_fits_compare_the_time_per_fit_on_all_the_cores(results):
    results.add_search(LAPTOP, SKLEARN, seconds=40, n_fits=100)
    results.add_search(LAPTOP, SKLEARN, seconds=1, n_fits=50, mask=LOGICAL_CPUS // 2)
    results.add_search(XEON, SKLEARN, seconds=80, n_fits=400)

    rendered, config = render(
        [{"tab": "sklearn-pypi"}, {"tab": "sklearn-pypi", "baseline": XEON_NAME}],
        page=gen_hardware_parallel_fits,
    )
    assert [variant["label"] for variant in only(config["outerTabs"])["variants"]] == [LAPTOP_NAME, XEON_NAME]
    point = only(points(rendered[0], "tree-based"))
    assert (point.name, point.speedup) == (XEON_NAME, pytest.approx(2))  # 400 ms vs 200 ms per fit
    assert only(points(rendered[1], "tree-based")).speedup == pytest.approx(0.5)


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

    rendered = on_machine(L4)
    sklearnex = [point for point in points(rendered, "tree-based") if point.name == "sklearnex-cpu"]
    by_speedup = by_time(sklearnex, 1000)
    exact, binned = by_speedup[500], by_speedup[200]
    assert exact.symbol == "circle"
    assert binned.symbol == "square"
    assert exact.x < binned.x


def test_sklearnex_baseline_vs_sklearn_has_the_same_two_columns(results):
    _binning_results(results)

    rendered = on_machine(L4, baseline="sklearnex-cpu")
    sklearn = [point for point in points(rendered, "tree-based") if point.name == "sklearn-pypi"]
    assert len(sklearn) == 2
    exact, binned = sorted(sklearn, key=lambda point: point.x)
    assert exact.speedup == pytest.approx(0.5) and exact.symbol == "circle"
    assert binned.speedup == pytest.approx(0.2) and binned.symbol == "square"


@pytest.mark.parametrize("baseline, other", [("sklearn-pypi", "cuml-gpu"), ("cuml-gpu", "sklearn-pypi")])
def test_sklearn_vs_cuml_has_one_column_with_squares(results, baseline, other):
    _binning_results(results)

    rendered = on_machine(L4, baseline=baseline)
    point = only([point for point in points(rendered, "tree-based") if point.name == other])
    assert point.symbol == "square"


def test_cuml_is_only_paired_with_binned_sklearnex_without_a_marker(results):
    _binning_results(results)

    from_cuml = on_machine(L4, baseline="cuml-gpu")
    point = only([point for point in points(from_cuml, "tree-based") if point.name == "sklearnex-cpu"])
    assert point.speedup == pytest.approx(0.5)  # the binned 200 ms, not the exact 500 ms
    assert point.symbol == "circle"

    from_sklearnex = on_machine(L4, baseline="sklearnex-cpu")
    point = only([point for point in points(from_sklearnex, "tree-based") if point.name == "cuml-gpu"])
    assert point.speedup == pytest.approx(2)
    assert point.symbol == "circle"


def test_failed_cuml_run_stays_in_cuml_column(results):
    forest = {"estimator": FOREST}
    results.add(L4, SKLEARN, fit=[1000], **forest)
    results.add(L4, CUML, fit=[100], **forest)
    results.add(L4, SKLEARN, fit=[1000], params={"max_depth": 3}, **forest)
    results.add_failed(L4, CUML, params={"max_depth": 3}, **forest)

    tree_points = [
        point for point in points(on_machine(L4), "tree-based")
        if point.name == "cuml-gpu"
    ]
    failed = only([point for point in tree_points if point.symbol == "x"])
    plotted = only([point for point in tree_points if point.symbol != "x"])
    assert failed.x == pytest.approx(plotted.x)


def test_exact_sklearnex_failure_is_left_out_against_cuml(results):
    forest = {"estimator": FOREST}
    results.add(L4, CUML, fit=[100], **forest)
    results.add(L4, SKLEARNEX, fit=[200], **forest)
    results.add_failed(L4, SKLEARNEX, params=EXACT, **forest)

    rendered = on_machine(L4, baseline="cuml-gpu")
    assert only(points(rendered, "tree-based")).symbol == "circle"


def test_sklearnex_across_machines_pairs_each_split_kind_with_itself(results):
    forest = {"estimator": FOREST}
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **forest)
    results.add(XEON, SKLEARNEX, fit=[50], params=EXACT, **forest)
    results.add(XEON, SKLEARNEX, fit=[25], **forest)

    rendered = across_machines("sklearnex-cpu")
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

    symbols = sorted(point.symbol for point in points(on_machine(LAPTOP)))
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

    point = only(points(on_machine(LAPTOP)))
    assert point.symbol == "square" and expected in point.text


def test_sklearnex_fallback_is_grey_without_warnings(results):
    results.add(LAPTOP, SKLEARN, estimator=FOREST, attributes={"n_iter": 10})
    fallback = {"has_onedal_estimator": False, "n_iter": 100}
    results.add(LAPTOP, SKLEARNEX, estimator=FOREST, attributes=fallback)

    point = only(points(on_machine(LAPTOP), "tree-based"))
    assert point.color == GREY and point.symbol == "circle"


# --- Failed runs ---


def test_failed_candidate_is_below_its_column_and_keeps_the_baseline_row(results):
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, fit=[50], params={"alpha": 1})
    results.add(LAPTOP, SKLEARN, fit=[100], params={"alpha": 2})
    results.add_failed(LAPTOP, SKLEARNEX, params={"alpha": 2})

    rendered = on_machine(LAPTOP)
    failed = only([point for point in points(rendered) if point.symbol == "x"])
    plotted = only([point for point in points(rendered) if point.symbol != "x"])
    assert failed.x == pytest.approx(plotted.x)
    assert failed.y < plotted.y

    rows = table(rendered, "linear")
    assert sorted(row["status"] for row in rows if row["variant"] == "sklearnex-cpu") == ["failed", "ok"]
    baseline_rows = [row for row in rows if row["variant"] == rendered["baselineLabel"]]
    assert len(baseline_rows) == 2
    assert sorted(str(row["fit_speedup"]) for row in baseline_rows) == ["1", "None"]


def test_failed_baseline_is_not_plotted_but_shows_in_the_table(results):
    results.add_failed(LAPTOP, SKLEARN, params={"alpha": 2})
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, params={"alpha": 1})
    results.add(LAPTOP, SKLEARNEX, fit=[77], params={"alpha": 2})

    rendered = on_machine(LAPTOP)
    assert only(points(rendered)).symbol == "circle"
    rows = table(rendered, "linear")
    assert [row["status"] for row in rows if row["status"] != "ok"] == ["failed"]
    candidate = next(row for row in rows if row.get("fit_time") == 77)
    assert candidate["fit_speedup"] is None


def test_failure_superseded_by_a_success_is_dropped_across_machines(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(XEON, SKLEARN, fit=[50])
    results.add_failed(XEON, SKLEARN, age=100)

    assert only(points(across_machines())).symbol == "circle"


# --- Layout ---


def test_columns_stay_on_their_estimator_with_many_columns(results, monkeypatch):
    monkeypatch.setattr(gen_cross_comparison, "HIDDEN_HARDWARE", set())
    forest = {"estimator": FOREST}
    other_forest = {"estimator": "RandomForestRegressor"}
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], params=EXACT, **other_forest)
    results.add(LAPTOP, SKLEARNEX, fit=[100], **other_forest)
    for hardware in (LAPTOP, XEON, DESKTOP, M4, L4):
        results.add(hardware, SKLEARN, fit=[25], **forest)
        results.add(hardware, SKLEARN, fit=[10], **other_forest)

    rendered = view(baseline=f"sklearnex-cpu__{LAPTOP}", target="software-sklearn-pypi")
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

    rendered = view(baseline=f"sklearn-pypi__{LAPTOP}", target="software-sklearn-pypi")
    assert rendered["baselineLabel"] == f"sklearn-pypi on {LAPTOP_NAME}"
    speedups = {point.name: point.speedup for point in points(rendered)}
    assert speedups == {XEON_NAME: pytest.approx(2), L4_CPU_NAME: pytest.approx(4)}


def test_cross_page_compares_every_software_on_one_machine(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(XEON, SKLEARN, fit=[50])
    results.add(L4, SKLEARN, fit=[25])
    results.add(L4, CUML, fit=[10])

    rendered = view(baseline=f"cuml-gpu__{L4}", target=f"hardware-{L4}")
    assert rendered["baselineLabel"] == f"cuml-gpu on {L4_GPU_NAME}"
    assert only(points(rendered)).name == "sklearn-pypi"
    assert only(points(rendered)).speedup == pytest.approx(0.4)


def test_cross_page_software_groups_filter_a_machines_softwares_only(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])
    results.add(LAPTOP, MKL, fit=[25])
    results.add(XEON, MKL, fit=[10])
    baseline = f"sklearnex-cpu__{LAPTOP}"

    def names(**spec):
        return {point.name for point in points(view(baseline=baseline, **spec))}

    machine = f"hardware-{LAPTOP}"
    assert names(target=machine) == {"sklearn-pypi"}
    assert names(target=machine, softwareGroups=["non-pypi"]) == {"sklearn-pypi", "sklearn-cf-mkl"}
    assert names(target="software-sklearn-cf-mkl", softwareGroups=[]) == {LAPTOP_NAME, XEON_NAME}


def test_cross_page_only_offers_targets_sharing_a_case(results):
    results.add(LAPTOP, SKLEARN, params={"alpha": 1})
    results.add(XEON, SKLEARN, params={"alpha": 1})
    results.add(LAPTOP, MKL, params={"alpha": 2})

    _, config = render([])
    assert config["pages"][f"sklearn-pypi__{LAPTOP}"] == ["software-sklearn-pypi", f"hardware-{XEON}"]
    assert f"sklearn-cf-mkl__{LAPTOP}" not in config["pages"]


def test_mkl_is_not_plotted_by_default_but_can_be_the_baseline(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, SKLEARNEX, fit=[50])
    results.add(LAPTOP, MKL, fit=[25])

    assert only(points(on_machine(LAPTOP))).name == "sklearnex-cpu"

    from_mkl = on_machine(LAPTOP, baseline="sklearn-cf-mkl")
    speedups = {point.name: point.speedup for point in points(from_mkl)}
    assert speedups == {"sklearn-pypi": pytest.approx(0.25), "sklearnex-cpu": pytest.approx(0.5)}


def test_builds_group_compares_scikit_learn_builds_with_pypi_only(results):
    results.add(LAPTOP, SKLEARN, fit=[100])
    results.add(LAPTOP, MKL, fit=[50])
    results.add(LAPTOP, CF_DEFAULT, fit=[25])
    results.add(LAPTOP, SKLEARNEX, fit=[10])
    results.add(LAPTOP, TORCH_CPU, fit=[10])

    rendered = on_machine(LAPTOP, softwareGroups=gen_cross_comparison.BUILDS_SOFTWARE_GROUPS)
    speedups = {point.name: point.speedup for point in points(rendered)}
    assert speedups == {"sklearn-cf-mkl": pytest.approx(2), "sklearn-cf-default": pytest.approx(4)}
