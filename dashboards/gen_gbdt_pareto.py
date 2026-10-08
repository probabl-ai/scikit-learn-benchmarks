"""Fit time vs test ROC AUC Pareto fronts of gradient boosting libraries, from
`configs/gbdt_pareto.py` (and its forced active wait variant).

One tab per machine, one plot per dataset, one series per library. A
HistGradientBoosting run from another env than `gbdt` (e.g. a `sklearn-dev`
branch) is its own series, and so is a run with forced active wait (same
color as its default counterpart, dashed).
"""
from html import escape
from pathlib import Path
from statistics import mean, median

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name, summarize_software_env
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
SOURCE_CONFIGS = ["configs/gbdt_pareto.py", "configs/gbdt_pareto_force_active_wait.py"]
SOURCE_ENVS = ["gbdt"]

GBDT_ENV = "gbdt"
LIBRARY_LABELS = {
    "sklearn": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
DATASET_ORDER = [
    "bank_marketing",
    "amazon_employee_access",
    "kick",
    "kddcup09_churn",
    "covtype",
]
ACTIVE_WAIT_SUFFIX = " (active wait)"
# Lowest ROC AUC shown at first, below each plot's best point.
Y_ZOOM = 0.06

ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares scikit-learn's <b>HistGradientBoosting</b> with
  <b>XGBoost</b>, <b>LightGBM</b> and <b>CatBoost</b> on real classification
  datasets. A single fit time comparison depends a lot on the
  <dfn>hyperparameters</dfn>: a library can be faster with few small trees and
  slower with many wide ones. So each library fits the same ladder of 8
  settings, from 10 stumps to 500 trees of 63 leaves, and each plot shows the
  resulting trade-off between fit time and accuracy.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine and one plot per dataset. The x-axis is
    the median <code><dfn>fit</dfn></code> time over the repeats (log scale),
    the y-axis is the mean <dfn>ROC AUC</dfn> on the test split. The solid line
    joins the points of each library's <dfn>Pareto front</dfn>, faded points
    are dominated. A front further up and to the left is better. Hover a
    point to see its setting.</p>
    <p>The plots start zoomed on the most accurate points: double-click a plot
    to see all of them.</p>
    <p>The settings are matched across libraries: number of trees, leaves
    per tree, learning rate, 255 bins, best-first (lossguide) growth, at
    least 20 samples per leaf, no L2 regularization and no early stopping.
    There are two exceptions: XGBoost has no minimum number of samples per
    leaf (it uses a minimum hessian sum of 0.001 instead), and CatBoost keeps
    its default L2 regularization (<code>l2_leaf_reg=3</code>). Categorical
    columns are passed as pandas <code>category</code> columns and handled
    natively by every library. Every library uses one <dfn>thread</dfn> per
    <dfn>physical core</dfn>.</p>
    <p>All libraries come from <dfn>conda-forge</dfn>, except CatBoost (from
    <dfn>PyPI</dfn>). HistGradientBoosting, XGBoost and LightGBM share the
    same <dfn>OpenMP runtime</dfn>, which doesn't use
    <dfn>active wait</dfn> by default. CatBoost uses its own thread pool.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li><b>LightGBM</b>'s front is above and to the left of
      <b>HistGradientBoosting</b>'s on every dataset: for the same setting, it
      is ~1.3x to 3x faster, with a similar accuracy.</li>
      <li><b>XGBoost</b> is up to ~2.5x faster than
      <b>HistGradientBoosting</b> for the same setting, but less accurate on
      the datasets with many rare categories or many noisy features
      (amazon_employee_access, kddcup09_churn). This may come from its missing
      minimum leaf size.</li>
      <li><b>CatBoost</b> is the most accurate on the noisy datasets (kick,
      kddcup09_churn) and the least accurate on amazon_employee_access and
      covtype. Its default L2 regularization likely plays a part.</li>
    </ul>
  </details>
</section>"""


def _dataset(record: BenchmarkRecord) -> str:
    return record.case["data"]["dataset"]


def _hp_setting(record: BenchmarkRecord) -> str:
    return record.case["metadata"]["hp_setting"]


def _is_active_wait(record: BenchmarkRecord) -> bool:
    return "GOMP_SPINCOUNT" in (record.case.get("bench", {}).get("env") or {})


def _series_label(record: BenchmarkRecord) -> str:
    library = record.implementation.library
    label = LIBRARY_LABELS.get(library, library)
    build = software_build_name(record.software_hash)
    if build != GBDT_ENV:
        label += f" ({build})"
    if _is_active_wait(record):
        label += ACTIVE_WAIT_SUFFIX
    return label


def _base_label(label: str) -> str:
    return label.removesuffix(ACTIVE_WAIT_SUFFIX)


def _label_sort_key(label: str) -> tuple:
    base = _base_label(label)
    order = list(LIBRARY_LABELS.values())
    return (order.index(base) if base in order else len(order), base, label)


def _series_colors(labels: list[str]) -> dict[str, str]:
    base_labels = sorted({_base_label(label) for label in labels}, key=_label_sort_key)
    base_colors = {
        base: SERIES_COLORS[index % len(SERIES_COLORS)] for index, base in enumerate(base_labels)
    }
    return {label: base_colors[_base_label(label)] for label in labels}


def _dedup_latest(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    latest: dict[tuple, BenchmarkRecord] = {}
    for record in records:
        key = (
            record.hardware_hash,
            record.software_hash,
            record.implementation.library,
            _dataset(record),
            _hp_setting(record),
            _is_active_wait(record),
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


def _point(record: BenchmarkRecord) -> tuple[float, float, str] | None:
    runs = [run for run in record.runs if "ROC AUC" in run["metrics"].get("predict", {})]
    if not runs:
        return None
    fit_s = median(run["time_ms"]["fit"] for run in runs) / 1000
    test_auc = mean(run["metrics"]["predict"]["ROC AUC"] for run in runs)
    train_auc = mean(run["metrics"]["fit"]["ROC AUC"] for run in runs)
    hover = "<br>".join([
        escape(_setting_line(record)),
        f"fit time: {fit_s:.3g}s (median of {len(runs)})",
        f"test ROC AUC: {test_auc:.4f}",
        f"train ROC AUC: {train_auc:.4f}",
    ])
    return fit_s, test_auc, hover


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
        series,
        colors=colors,
        line_dashes={label: "dash" for label in series if label.endswith(ACTIVE_WAIT_SUFFIX)},
        y_zoom=Y_ZOOM,
    )
    subtitle = f'<div class="plot-subtitle">{escape(_shape_line(records))}</div>'
    return f'<section class="plot-cell"><h3>{escape(dataset)}</h3>{subtitle}{plot}</section>'


def _software_tabs_html(hw_records: list[BenchmarkRecord]) -> str:
    cards = []
    for software_hash in sorted({record.software_hash for record in hw_records}):
        build = software_build_name(software_hash)
        summary = summarize_software_env(
            read_env("software", software_hash),
            Implementation(library="sklearn", device=None, data_library=None),
            software_hash=software_hash,
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


def render_hardware_page(records: list[BenchmarkRecord], hardware_hash: str) -> str | None:
    hw_records = [record for record in records if record.hardware_hash == hardware_hash]
    if not hw_records:
        return None

    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(hw_records))}</div>',
        f'<div class="page-row">{_software_tabs_html(hw_records)}</div>',
    ]
    failed_html = _failed_cases_html(hw_records)
    if failed_html:
        sections.append(f'<div class="page-row">{failed_html}</div>')

    colors = _series_colors([_series_label(record) for record in hw_records])
    by_dataset: dict[str, list[BenchmarkRecord]] = {}
    for record in hw_records:
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


def generate(output_dir: Path) -> None:
    records = _dedup_latest(
        [
            record for record in read_benchmark_records()
            if matches_source_configs(record.case, SOURCE_CONFIGS)
        ]
    )
    hardware_hashes = sorted(
        {record.hardware_hash for record in records},
        key=lambda hardware_hash: (
            list(HARDWARE_NAMES).index(hardware_hash)
            if hardware_hash in HARDWARE_NAMES
            else len(HARDWARE_NAMES),
            HARDWARE_NAMES.get(hardware_hash, hardware_hash),
        ),
    )
    hardware_pages = [
        (
            HARDWARE_NAMES.get(hardware_hash, hardware_hash),
            render_hardware_page(records, hardware_hash),
        )
        for hardware_hash in hardware_hashes
    ]
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, render_hardware_tabs(hardware_pages)],
    )
    output = output_dir / "gbdt_pareto.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
