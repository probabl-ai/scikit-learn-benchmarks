"""Fit time vs test score (ROC AUC or R2) Pareto fronts of gradient boosting
libraries, from `configs/gbdt_pareto.py`, `configs/gbdt_pareto_regression.py`
and their forced active wait variants.

One tab per (machine, OpenMP wait policy), one plot per dataset, one series
per library. A HistGradientBoosting run from another env than `gbdt` (e.g. a
`sklearn-dev` branch) is its own series.
"""
from html import escape
from pathlib import Path
from statistics import mean, median

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import (
    active_wait_label_suffix, has_active_wait, read_env, software_build_name,
    summarize_software_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    SERIES_COLORS,
    SOFTWARE_TEMPLATE,
    pareto_plot_html,
    render_hardware_tabs,
    render_software_tabs,
)
from sklbench.reporting.matching import (
    BenchmarkRecord, Implementation, date_range, matches_source_configs,
    read_benchmark_records,
)


TITLE = "Gradient boosting libraries: fit time vs accuracy"
SOURCE_CONFIGS = [
    "configs/gbdt_pareto.py",
    "configs/gbdt_pareto_force_active_wait.py",
    "configs/gbdt_pareto_regression.py",
    "configs/gbdt_pareto_regression_force_active_wait.py",
]
SOURCE_ENVS = ["gbdt"]

GBDT_ENV = "gbdt"
LIBRARY_LABELS = {
    "sklearn": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
# Series labels for HistGradientBoosting builds other than the `gbdt` env's.
BUILD_LABELS = {
    # https://github.com/scikit-learn/scikit-learn/pull/34935
    "sklearn-dev@cakedev0:hgb/active_wait": "HGB (PR)",
}
DATASET_ORDER = [
    "bank_marketing",
    "amazon_employee_access",
    "kick",
    "kddcup09_churn",
    "covtype",
    "ames_housing",
    "california_housing",
    "medical_charges_nominal",
    "year_prediction_msd",
]
METRICS = {"classification": "ROC AUC", "regression": "R2"}
# Libraries whose threads don't depend on the OpenMP wait policy: their
# points are shown in every wait policy tab of their machine.
NO_OPENMP_LIBRARIES = {"catboost"}
# Lowest score shown at first, below each plot's best point.
Y_ZOOM = 0.06

ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares scikit-learn's <b>HistGradientBoosting</b> with
  <b>XGBoost</b>, <b>LightGBM</b> and <b>CatBoost</b> on real classification
  and regression datasets. A single fit time comparison depends a lot on the
  <dfn>hyperparameters</dfn>: a library can be faster with few small trees and
  slower with many wide ones. So each library fits the same ladder of 8
  settings, from 10 stumps to 500 trees of 63 leaves, and each plot shows the
  resulting trade-off between fit time and accuracy.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine and <dfn>OpenMP</dfn> wait policy, and one
    plot per dataset. The x-axis is
    the median <code><dfn>fit</dfn></code> time over the repeats (log scale),
    the y-axis is the mean test score: <dfn>ROC AUC</dfn> for classification,
    <dfn>R2</dfn> for regression. The solid line
    joins the points of each library's <dfn>Pareto front</dfn>, faded points
    are dominated. A front further up and to the left is better. Hover a
    point to see its setting.</p>
    <p>The plots start zoomed on the most accurate points: double-click a plot
    to see all of them.</p>
    <p>The settings are matched across libraries: number of trees, leaves
    per tree, learning rate, 255 bins, best-first (lossguide) growth, at
    least 20 samples per leaf, no L2 regularization and no early stopping.
    There are two exceptions: for classification, XGBoost has no minimum
    number of samples per leaf (it uses a minimum hessian sum of 0.001
    instead), and CatBoost keeps
    its default L2 regularization (<code>l2_leaf_reg=3</code>). Every
    library uses one <dfn>thread</dfn> per <dfn>physical core</dfn>.</p>
    <p>Categorical columns: <b>HistGradientBoosting</b>, <b>XGBoost</b> and
    <b>LightGBM</b> get them as native categorical columns capped at 20
    categories (rarer ones are grouped), plus a <dfn>target encoding</dfn> of
    each column on all its categories. <b>CatBoost</b> gets every category and
    encodes them itself. See "The kddcup09_churn case" for why. With this
    preprocessing, <b>XGBoost</b> is still less accurate on kddcup09_churn and
    amazon_employee_access: its native splits on the capped columns still
    overfit, and a larger minimum hessian sum per leaf doesn't fix it.</p>
    <p>All libraries come from <dfn>conda-forge</dfn>, except CatBoost (from
    <dfn>PyPI</dfn>). HistGradientBoosting, XGBoost and LightGBM share the
    same <dfn>OpenMP runtime</dfn>. Whether it uses <dfn>active wait</dfn> by
    default depends on the machine: not on the laptop (the tab marked "no
    active wait"), but yes on the Xeon server. The laptop's other tab forces
    active wait with environment variables (<code>GOMP_SPINCOUNT=300000</code>,
    <code>KMP_BLOCKTIME=200ms</code>). The OpenMP section of each tab shows the
    setting. CatBoost uses its own thread pool, so its points are the same in
    both of a machine's tabs.</p>
    <p><b>HGB (PR)</b> is HistGradientBoosting from
    <a href="https://github.com/scikit-learn/scikit-learn/pull/34935">scikit-learn#34935</a>,
    which picks the number of threads per tree and per parallel loop from
    the amount of work and from the wait policy. It is built from source in
    the <code>sklearn-dev</code> environment, with the same conda-forge
    OpenMP runtime.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases, comparing fit times at the same setting
    (geometric mean over the 8 settings of a dataset):</p>
    <ul>
      <li>The accuracy is close across libraries: the best test scores are
      within ~0.01 of each other on every dataset, except for <b>XGBoost</b> on
      amazon_employee_access and kddcup09_churn (~0.04 lower) and
      <b>CatBoost</b> on covtype (~0.02 lower).</li>
      <li>On the laptop, forcing <dfn>active wait</dfn> makes
      <b>HistGradientBoosting</b>, <b>XGBoost</b> and <b>LightGBM</b> ~2x to 9x
      faster on every dataset but the largest, year_prediction_msd (up to
      ~1.5x). The wait policy matters as much for the three libraries.</li>
      <li><b>HGB (PR)</b> gets most of that gain without active wait: on the
      laptop, it is then ~1.2x to 6x faster than the released
      <b>HistGradientBoosting</b> (~13x on california_housing), and faster than
      <b>LightGBM</b> on 7 of the 9 datasets. With active wait, it is ~1.1x to
      2.7x faster than the released version, and <b>LightGBM</b> stays ~1.3x
      to 2.2x faster, except on year_prediction_msd. On the Xeon server,
      where active wait is the default, <b>HGB (PR)</b> is ~1.3x to 6x faster
      than the released version, and faster than <b>LightGBM</b> on 8 of the 9
      datasets. The accuracy is the same as the released version.</li>
      <li><b>LightGBM</b> is ~1.4x to 2.7x faster than the released
      <b>HistGradientBoosting</b> on the laptop on most datasets, and ~1.3x to
      1.8x faster on the Xeon server.</li>
      <li><b>XGBoost</b> is ~1.1x to 2.2x faster than the released
      <b>HistGradientBoosting</b> on the laptop, except on year_prediction_msd
      with active wait (~1.4x slower), and ~1.1x to 1.3x faster on the Xeon
      server.</li>
      <li><b>CatBoost</b> doesn't use OpenMP, so its fit times don't depend on
      the wait policy. On the laptop, it is faster than the released
      <b>HistGradientBoosting</b> on 6 of the 9 datasets without active wait,
      and ~1.2x to 3.5x slower with it. It is the most accurate on
      amazon_employee_access and kick.</li>
    </ul>
  </details>
  <details class="about-section">
    <summary>The kddcup09_churn case</summary>
    <p>This case is why categorical columns are preprocessed as described in
    "How to read". With native categorical splits on columns capped at 252
    categories for every library, <b>CatBoost</b> reached a test ROC AUC of
    ~0.74 on kddcup09_churn, while the other libraries stayed between ~0.67
    and ~0.70. Refitting the same settings on the same data, changing one
    thing at a time, shows why. The numbers below are the mean test ROC AUC
    over 3 train/test splits, for 300 trees of 7 leaves:</p>
    <ul>
      <li>Native categorical splits hurt here. Dropping the 34 categorical
      columns entirely makes <b>HistGradientBoosting</b> better (0.692 to
      0.714). Treating the category codes as plain numbers is better still
      (0.735), and the same holds for <b>LightGBM</b> (0.701 to 0.726) and
      <b>XGBoost</b> (0.671 to 0.736).</li>
      <li><b>CatBoost</b>'s edge is its categorical encoding. It replaces each
      category with a smoothed <dfn>target encoding</dfn>, computed on earlier
      rows only. <b>HistGradientBoosting</b> on columns encoded by
      scikit-learn's cross-fitted <code>TargetEncoder</code> does the same kind
      of encoding and matches <b>CatBoost</b> (0.737 vs 0.738). Without its
      encoding, <b>CatBoost</b> drops to the others' level (0.725).</li>
      <li>Regularization and randomness barely matter. Without its L2
      regularization, <b>CatBoost</b> loses 0.001, and the same L2
      regularization doesn't help <b>HistGradientBoosting</b> or
      <b>LightGBM</b>. Turning off the random noise that
      <b>CatBoost</b> adds to split scores (<code>random_strength</code>) makes
      no clear difference. Nor is the
      missing minimum leaf size why <b>XGBoost</b> is last: a minimum hessian
      sum of 20 gains only 0.003.</li>
      <li>The damage comes from the high-cardinality columns. Keeping native
      splits only for the 21 columns with at most 50 categories, and plain
      numbers for the other 13, gives 0.733, nearly as good as all plain
      numbers. Allowing up to 100 categories already drops it to 0.722. 8 of
      those 13 columns are at the <dfn>preprocessing</dfn> cap of 252
      categories.</li>
      <li>Why it overfits: with only 7% positives, searching for the best way
      to split ~250 categories into two groups finds splits that fit noise.
      The minimum of 20 samples per leaf doesn't stop that, because it counts
      samples per leaf, not per category.</li>
    </ul>
    <p>Lowering the cap alone is not enough: amazon_employee_access needs the
    identity of its rare categories, and <b>HistGradientBoosting</b> drops from
    0.843 to 0.775 with 20 categories. Native columns capped at 20 categories
    plus a target encoding of each work on both datasets: 0.739 on
    kddcup09_churn, and 0.868 on amazon_employee_access, more than any
    library with the previous preprocessing.</p>
  </details>
</section>"""


def _dataset(record: BenchmarkRecord) -> str:
    return record.case["data"]["dataset"]


def _hp_setting(record: BenchmarkRecord) -> str:
    return record.case["metadata"]["hp_setting"]


def _case_env(record: BenchmarkRecord) -> dict:
    # `read_benchmark_records` moves `bench.env` to `case["env"]`.
    return record.case.get("env") or {}


def _active_wait(record: BenchmarkRecord) -> bool:
    return has_active_wait(_case_env(record), record.software_hash)


def _uses_openmp(record: BenchmarkRecord) -> bool:
    return record.implementation.library not in NO_OPENMP_LIBRARIES


def _series_label(record: BenchmarkRecord) -> str:
    library = record.implementation.library
    label = LIBRARY_LABELS.get(library, library)
    build = software_build_name(record.software_hash)
    if build in BUILD_LABELS:
        return BUILD_LABELS[build]
    if build != GBDT_ENV:
        label += f" ({build})"
    return label


def _label_sort_key(label: str) -> tuple:
    order = list(LIBRARY_LABELS.values())
    return (order.index(label) if label in order else len(order), label)


def _series_colors(labels: set[str]) -> dict[str, str]:
    """Each library keeps its color across tabs."""
    return {
        label: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, label in enumerate(sorted(set(LIBRARY_LABELS.values()) | labels, key=_label_sort_key))
    }


def _dedup_latest(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    latest: dict[tuple, BenchmarkRecord] = {}
    for record in records:
        key = (
            record.hardware_hash,
            record.software_hash,
            record.implementation.library,
            _dataset(record),
            _hp_setting(record),
            _active_wait(record),
        )
        current = latest.get(key)
        if current is None or record.timestamp_recorded > current.timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _param(params: dict, *names: str):
    return next((params[name] for name in names if name in params), "?")


def _setting_line(record: BenchmarkRecord) -> str:
    params = record.case["algorithm"]["estimator_params"]
    n_trees = _param(params, "max_iter", "n_estimators", "iterations")
    n_leaves = _param(params, "max_leaf_nodes", "max_leaves", "num_leaves")
    learning_rate = _param(params, "learning_rate")
    return (
        f"{_hp_setting(record)}: {n_trees} trees, {n_leaves} leaves, "
        f"learning rate {learning_rate}"
    )


def _metric(record: BenchmarkRecord) -> str:
    return METRICS[record.case["metadata"]["task"]]


def _point(record: BenchmarkRecord) -> tuple[float, float, str] | None:
    metric = _metric(record)
    runs = [run for run in record.runs if metric in run["metrics"].get("predict", {})]
    if not runs:
        return None
    fit_s = median(run["time_ms"]["fit"] for run in runs) / 1000
    test_score = mean(run["metrics"]["predict"][metric] for run in runs)
    train_score = mean(run["metrics"]["fit"][metric] for run in runs)
    hover = "<br>".join([
        escape(_setting_line(record)),
        f"fit time: {fit_s:.3g}s (median of {len(runs)})",
        f"test {metric}: {test_score:.4f}",
        f"train {metric}: {train_score:.4f}",
    ])
    return fit_s, test_score, hover


def _shape_line(records: list[BenchmarkRecord]) -> str:
    for record in records:
        for run in record.runs:
            fit = run.get("data_desc", {}).get("fit", {})
            if fit.get("samples") and fit.get("features"):
                line = f"Train: {fit['samples']:,} x {fit['features']}"
                if fit.get("n_categorical_features"):
                    line += f" ({fit['n_categorical_features']} categorical)"
                if fit.get("n_classes"):
                    line += f", {fit['n_classes']} classes"
                else:
                    line += ", regression"
                return line
    return ""


def _dataset_cell_html(dataset: str, records: list[BenchmarkRecord], colors: dict) -> str:
    series: dict[str, list] = {}
    for record in records:
        point = _point(record)
        if point is not None:
            series.setdefault(_series_label(record), []).append(point)
    if not series:
        return ""
    series = {label: series[label] for label in sorted(series, key=_label_sort_key)}
    plot = pareto_plot_html(
        series, colors=colors, y_title=f"test {_metric(records[0])}", y_zoom=Y_ZOOM
    )
    subtitle = f'<div class="plot-subtitle">{escape(_shape_line(records))}</div>'
    return f'<section class="plot-cell"><h3>{escape(dataset)}</h3>{subtitle}{plot}</section>'


def _software_tabs_html(tab_records: list[BenchmarkRecord]) -> str:
    cards = []
    for software_hash in sorted({record.software_hash for record in tab_records}):
        build = software_build_name(software_hash)
        # The OpenMP summary shows this tab's wait policy override, if any.
        case_env = next(
            (
                _case_env(record) for record in tab_records
                if record.software_hash == software_hash and _uses_openmp(record)
            ),
            None,
        )
        summary = summarize_software_env(
            read_env("software", software_hash),
            Implementation(library="sklearn", device=None, data_library=None),
            software_hash=software_hash,
            case_env=case_env,
            extra_packages=["xgboost", "lightgbm", "catboost"] if build == GBDT_ENV else (),
        )
        summary["name"] = build
        cards.append(SOFTWARE_TEMPLATE.render(**summary))
    return render_software_tabs(cards)


def _failure_reason(record: BenchmarkRecord) -> str:
    logs = (record.failed_case or {}).get("logs", {}) or {}
    stderr_lines = str(logs.get("stderr", "")).strip().splitlines()
    if stderr_lines:
        return stderr_lines[-1]
    return_code = (record.failed_case or {}).get("return_code")
    return f"return code {return_code}" if return_code is not None else "failed"


def _failed_cases_html(hw_records: list[BenchmarkRecord]) -> str:
    failed = [record for record in hw_records if record.failed_case is not None and not record.runs]
    if not failed:
        return ""
    items = "".join(
        f"<li>{escape(_series_label(record))} / {escape(_dataset(record))} "
        f"({escape(_hp_setting(record))}): {escape(_failure_reason(record))}</li>"
        for record in sorted(failed, key=lambda r: (_series_label(r), _dataset(r), _hp_setting(r)))
    )
    return f'<section class="panel"><h3>Failed cases</h3><ul class="compact">{items}</ul></section>'


def _dataset_sort_key(dataset: str) -> tuple:
    return (
        DATASET_ORDER.index(dataset) if dataset in DATASET_ORDER else len(DATASET_ORDER),
        dataset,
    )


def render_tab_page(tab_records: list[BenchmarkRecord], colors: dict[str, str]) -> str:
    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(tab_records))}</div>',
        f'<div class="page-row">{_software_tabs_html(tab_records)}</div>',
    ]
    failed_html = _failed_cases_html(tab_records)
    if failed_html:
        sections.append(f'<div class="page-row">{failed_html}</div>')

    by_dataset: dict[str, list[BenchmarkRecord]] = {}
    for record in tab_records:
        if record.runs:
            by_dataset.setdefault(_dataset(record), []).append(record)
    cells = [
        cell_html
        for dataset in sorted(by_dataset, key=_dataset_sort_key)
        if (cell_html := _dataset_cell_html(dataset, by_dataset[dataset], colors))
    ]
    if cells:
        grid = (
            '<section class="plot-grid" '
            'style="grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));">'
            + "".join(cells)
            + "</section>"
        )
        sections.append(f'<div class="page-row">{grid}</div>')
    return "".join(sections)


def _hardware_sort_index(hardware_hash: str) -> int:
    try:
        return list(HARDWARE_NAMES).index(hardware_hash)
    except ValueError:
        return len(HARDWARE_NAMES)


def generate(output_dir: Path) -> None:
    records = _dedup_latest(
        [
            record for record in read_benchmark_records()
            if matches_source_configs(record.case, SOURCE_CONFIGS)
        ]
    )
    tabs: dict[tuple[str, bool], list[BenchmarkRecord]] = {}
    for record in records:
        if _uses_openmp(record):
            tabs.setdefault((record.hardware_hash, _active_wait(record)), []).append(record)
    for record in records:
        if not _uses_openmp(record):
            keys = [key for key in tabs if key[0] == record.hardware_hash]
            for key in keys or [(record.hardware_hash, _active_wait(record))]:
                tabs.setdefault(key, []).append(record)

    colors = _series_colors({_series_label(record) for record in records})
    pages = [
        (
            HARDWARE_NAMES.get(hardware_hash, hardware_hash)
            + active_wait_label_suffix(active_wait),
            render_tab_page(tab_records, colors),
        )
        for (hardware_hash, active_wait), tab_records in sorted(
            tabs.items(), key=lambda item: (_hardware_sort_index(item[0][0]), not item[0][1])
        )
    ]
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, render_hardware_tabs(pages)],
    )
    output = output_dir / "gbdt_pareto.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
