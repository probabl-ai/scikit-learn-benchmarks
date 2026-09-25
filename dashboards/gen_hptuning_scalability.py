"""Outer-parallelism scalability for `configs/hptuning.py`'s
`RandomizedSearchCV` sweep.

That config leaves both the outer (`RandomizedSearchCV(n_jobs=...)`) and
inner (each candidate's own parallelism - RF/ET's explicit `n_jobs=-1`,
HGB's OpenMP thread pool, BLAS threads under Ridge/LogisticRegression)
parallelism to do whatever they do on their own, and sweeps outer `n_jobs`
log-spaced from 1 up to half the machine's physical cores (see
`configs/hptuning.py`'s `_case`/`get_n_cores_list` usage) specifically to see
whether that composition scales or falls over. One small multiple per
(estimator, dataset): seconds per fit vs. outer `n_jobs`, with each swept
environment (`sklearn-pypi`/`sklearn-cf-mkl`/`intel`) as its own colored line
on the same plot (see `ENV_COLORS`) rather than a separate plot per
environment, so a build's effect on scaling reads directly off one cell. Mean
CPU utilization (from the record's `system_telemetry` - meant to tell "busy
but inefficient", e.g. HGB's own thread pool colliding with concurrent outer
candidates, apart from "idle", e.g. dispatch/IPC overhead dominating over
too-cheap candidates) sizes each point's marker rather than getting its own
panel - see `_env_series`.

Seconds *per fit* (`duration_s / (n_iter * cv_n_splits)` - see
`_seconds_per_fit`), i.e. mean wall-clock time per `.fit()` call across the
whole search, not raw wall time: `configs/hptuning.py`'s `n_iter` is fixed
per machine (same value for every point in a row's `n_jobs` sweep - see its
`_case`), so this is just `duration_s` rescaled by a row-constant factor and
answers the actual question directly - does raising outer `n_jobs` let the
machine push through more fits per second? A line dropping roughly ∝
`1 / n_jobs` is ideal (linear) scaling; flat means more outer workers buy
nothing; rising means outer parallelism is actively hurting (typically
oversubscription against an inner parallelism the candidate already uses -
see RF/ET's explicit `n_jobs=-1`).

Records are identified by `metadata.source_config` (see `SOURCE_CONFIGS` and
`sklbench.reporting.matching.matches_source_configs`), stamped at load time
by `sklbench.config.loader.load_cases_from_script`. `duration_s`/
`cpu_percent` aren't columns `MethodResult`/
`read_all_results()` understands (that machinery expects a `time_ms` dict
keyed by run method, e.g. "fit"/"predict" - `sklbench/runners/hptuning.py`
writes flat `duration_s`/`best_score` rows instead), so this reads raw
`BenchmarkRecord`s directly, same as `gen_hgb_scalability_breakdown.py` does
for its own custom instrumented fields. `system_telemetry` isn't parsed into
`BenchmarkRecord` at all, so it's read straight from `record.record_path`.

Two datasets sharing a `data.source` name (e.g. `make_classification`) can
still be two different shapes - `SYNTHETIC_SPECS` in `configs/hptuning.py`
pairs `LogisticRegression` with a different `make_classification` shape in
two different specs, one per tree-estimator counterpart (RandomForest vs
ExtraTrees) - so the small-multiple identity below is (estimator, dataset
name, shape), not just (estimator, dataset name).
"""
from html import escape
import json
from pathlib import Path
from statistics import mean, median

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name, summarize_software_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    format_duration_ms,
    render_hardware_tabs,
    render_software_tabs,
    scaling_line_plot_html,
    variant_color_map,
)
from sklbench.reporting.matching import (
    BenchmarkRecord, Implementation, date_range, matches_source_configs,
    read_benchmark_records,
)


ENV_ORDER = ["sklearn-pypi", "sklearn-cf-mkl", "intel"]
ENV_COLORS = variant_color_map(ENV_ORDER)

ESTIMATOR_ORDER = [
    "Ridge",
    "LogisticRegression",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "ExtraTreesRegressor",
    "ExtraTreesClassifier",
    "HistGradientBoostingRegressor",
    "HistGradientBoostingClassifier",
]

DURATION_METRIC = "s / fit"
CPU_METRIC = "CPU utilization (%)"


SOURCE_CONFIGS = ["configs/hptuning.py"]
SOURCE_ENVS = ENV_ORDER

ABOUT_HTML = """<section class="panel">
  <p>RandomizedSearchCV has two levels of parallelism: its outer
  <code>n_jobs</code> runs candidates in parallel, and each candidate can use
  threads itself (<code>n_jobs=-1</code> for RandomForest and ExtraTrees,
  OpenMP for HistGradientBoosting, BLAS for Ridge and LogisticRegression).
  This dashboard sweeps the outer <code>n_jobs</code> to check how the two
  levels interact. There is one tab per machine and one plot per (estimator,
  dataset) pair. The x-axis is the outer <code>n_jobs</code> (log scale), the
  y-axis is the mean time per <code>.fit()</code> call over the search, with
  one line per environment and marker size for mean CPU usage. A line going
  down as 1/n_jobs is ideal scaling. A flat line means more outer workers
  bring nothing, and a rising one usually means oversubscription.</p>
  <p>On the benchmarked cases, outer parallelism helps in every case. Ridge,
  LogisticRegression and HistGradientBoosting scale close to linearly.
  RandomForest and ExtraTrees only reach ~5-6x at 86 outer workers, because
  their inner <code>n_jobs=-1</code> competes for the same cores.</p>
</section>"""


def _estimator(record: BenchmarkRecord) -> str:
    return record.case["algorithm"]["estimator"]


def _dataset_name(record: BenchmarkRecord) -> str:
    data = record.case.get("data", {})
    return data.get("dataset") or data.get("source") or "unknown"


def _data_desc_raw(run: dict) -> dict:
    """`data_desc`'s pre-preprocessing shape - subsampled (per
    `hptuning.max_samples`) but not yet through whatever `preprocessing_kind`
    transform the pipeline applies. Older records (from before the runner
    split `data_desc` into `raw`/`fit` - see `sklbench/runners/hptuning.py`)
    stored this same shape flat instead of nested under `"raw"`."""
    desc = run.get("data_desc") or {}
    return desc.get("raw", desc)


def _dataset_shape(record: BenchmarkRecord) -> tuple[int | None, int | None]:
    generation_kwargs = record.case.get("data", {}).get("generation_kwargs")
    if generation_kwargs:
        return generation_kwargs.get("n_samples"), generation_kwargs.get("n_features")
    # Real datasets don't carry shape in the case - `hptuning.max_samples`
    # can subsample them, so read the actual post-subsampling shape off a
    # run's own `data_desc` instead (same for every n_jobs point of a given
    # spec, since `max_samples` doesn't vary with n_jobs).
    for run in record.runs:
        desc = _data_desc_raw(run)
        if desc.get("n_samples") and desc.get("n_features"):
            return desc["n_samples"], desc["n_features"]
    return None, None


def _fit_shape(row_records: list[BenchmarkRecord]) -> tuple[int | None, int | None]:
    """Shape actually seen by `.fit()`, after whatever `preprocessing_kind`
    transform the pipeline applies (e.g. one-hot encoding categorical
    columns expanding `n_features`) - `(None, None)` for records from
    before the runner captured this (see `_data_desc_raw`), or when no run
    in this row has it."""
    for record in row_records:
        for run in record.runs:
            fit = (run.get("data_desc") or {}).get("fit") or {}
            if fit.get("n_samples") and fit.get("n_features"):
                return fit["n_samples"], fit["n_features"]
    return None, None


def _row_key(record: BenchmarkRecord) -> tuple[str, str, int | None, int | None]:
    return (_estimator(record), _dataset_name(record), *_dataset_shape(record))


def _n_jobs(record: BenchmarkRecord) -> int:
    # HPTuning.n_jobs defaults to 1, so a case swept at n_jobs=1 (the
    # baseline point of every sweep) is serialized without this key at all
    # (`BaseCase.json_dict`'s `exclude_defaults=True`) - same reason
    # `_implementation` below falls back for a missing "implementation" key.
    return record.case["hptuning"].get("n_jobs", 1)


def _n_iter(record: BenchmarkRecord) -> int:
    return record.case["hptuning"].get("n_iter", 14)


def _n_fits(record: BenchmarkRecord) -> int:
    """Total `.fit()` calls `RandomizedSearchCV` makes for this case:
    `n_iter` candidates x `cv_n_splits` folds each - `cv_n_splits` defaults
    to 3 and `configs/hptuning.py` never overrides it, so it's omitted from
    the serialized case the same way `n_jobs=1`/plain-sklearn
    `implementation` are (see `_n_jobs`/`_implementation`)."""
    return _n_iter(record) * record.case["hptuning"].get("cv_n_splits", 3)


def _env(record: BenchmarkRecord) -> str:
    return software_build_name(record.software_hash)


def _env_sort_key(env: str) -> tuple[int, str]:
    return (ENV_ORDER.index(env) if env in ENV_ORDER else len(ENV_ORDER), env)


def _implementation(record: BenchmarkRecord) -> Implementation:
    # `HPTuningCase.implementation` defaults to plain sklearn - a case whose
    # implementation matches that default is serialized without an
    # "implementation" key at all (`BaseCase.json_dict`'s
    # `exclude_defaults=True`), so `record.case["implementation"]` would
    # KeyError for those - same reason `gen_models_scalability.py` keys its
    # env tabs off `software_build_name` rather than this.
    implementation = record.case.get("implementation") or {}
    return Implementation(
        library=implementation.get("library", "sklearn"),
        device=implementation.get("device"),
        data_library=implementation.get("data_library"),
    )


def _duration_s(record: BenchmarkRecord) -> float | None:
    # Median (not mean) across the n_runs repeats, so one stalled repeat
    # (scheduler/thermal noise, not the parallelism behavior under test)
    # doesn't skew the point - same convention as
    # `gen_hgb_scalability_breakdown.py`'s `_phase_breakdown_ms`.
    durations = [run["duration_s"] for run in record.runs if "duration_s" in run]
    return median(durations) if durations else None


def _seconds_per_fit(record: BenchmarkRecord) -> float | None:
    """Mean wall-clock time per `.fit()` call across the whole search:
    `duration_s / n_fits` - i.e. the reciprocal of fits-per-second
    throughput. Decreasing with `n_jobs` means more outer workers are
    letting the machine get through fits faster; flat/rising means they
    aren't (or are actively hurting, e.g. oversubscription against a
    candidate's own inner parallelism)."""
    duration = _duration_s(record)
    if duration is None:
        return None
    return duration / _n_fits(record)


def _cpu_percent_mean(record: BenchmarkRecord) -> float | None:
    if record.record_path is None:
        return None
    raw = json.loads(record.record_path.read_text())
    samples = [
        sample["cpu_percent"]
        for sample in raw.get("system_telemetry", [])
        if sample.get("cpu_percent") is not None
    ]
    return mean(samples) if samples else None


def _memory_percent_max(record: BenchmarkRecord) -> float | None:
    if record.record_path is None:
        return None
    raw = json.loads(record.record_path.read_text())
    samples = [
        sample["memory"]["used_percent"]
        for sample in raw.get("system_telemetry", [])
        if sample.get("memory", {}).get("used_percent") is not None
    ]
    return max(samples) if samples else None


def _hover_extra(record: BenchmarkRecord) -> str:
    parts = [f"n_iter: {_n_iter(record)}"]
    duration = _duration_s(record)
    if duration is not None:
        parts.append(f"wall time: {format_duration_ms(duration * 1000)}")
    cpu_percent = _cpu_percent_mean(record)
    if cpu_percent is not None:
        parts.append(f"CPU load: {cpu_percent:.0f}%")
    memory_percent = _memory_percent_max(record)
    if memory_percent is not None:
        parts.append(f"max memory: {memory_percent:.0f}%")
    if record.failed_case is not None:
        # A record can carry both partial `runs` and a `failed_case` (e.g.
        # 1 of 3 repeats completed before the orchestrator's time limit hit)
        # - `_failed_cases_html` only surfaces records with *no* usable
        # data, so a partial one like this still gets plotted normally and
        # needs its own flag here instead, since its point is a noisier,
        # single-repeat (or otherwise incomplete) median rather than the
        # usual n_runs one.
        parts.append("timed out (incomplete run)")
    return "<br>".join(parts)


def _dedup_latest(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    """Keep only the latest-timestamp record per (hardware, software, row,
    n_jobs) - `read_benchmark_records()` returns every historical run
    (including ones from a config shape that's since changed, e.g. this
    config's outer-`n_jobs` sweep or its RF/ET `n_jobs` used to be
    different), so without this a rerun would leave two points at the same
    x position for the same small multiple."""
    latest: dict[tuple, BenchmarkRecord] = {}
    for record in records:
        key = (record.hardware_hash, record.software_hash, _row_key(record), _n_jobs(record))
        current = latest.get(key)
        if current is None or record.timestamp_recorded > current.timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _params_line(record: BenchmarkRecord) -> str:
    params = record.case.get("algorithm", {}).get("estimator_params", {})
    if not params:
        return "params: (defaults)"
    formatted = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"params: {formatted}"


def _row_title(row_key: tuple, fit_shape: tuple[int | None, int | None]) -> str:
    estimator, dataset, n_samples, n_features = row_key
    if not n_samples or not n_features:
        return f"{estimator} / {dataset} (shape unknown)"
    dims = f"{n_samples:,} x {n_features}"
    fit_n_samples, fit_n_features = fit_shape
    if fit_n_features and (fit_n_samples, fit_n_features) != (n_samples, n_features):
        dims += f" → {fit_n_samples:,} x {fit_n_features}"
    return f"{estimator} / {dataset} ({dims})"


def _row_subtitle_html(record: BenchmarkRecord) -> str:
    return f'<div class="plot-subtitle">{escape(_params_line(record))}</div>'


def _env_series(
    row_records: list[BenchmarkRecord], point_fn
) -> dict[str, list[tuple[float, float, str, float | None]]]:
    """`point_fn(record) -> float | None` per env, keyed by env label - one
    line per environment on the same plot (`ENV_COLORS` keeps a build's color
    consistent across cells) rather than a separate plot per environment.
    Each point's marker is additionally sized by mean CPU utilization (see
    `scaling_line_plot_html`'s 4th point element) - a visual signal for
    "busy but inefficient" (e.g. HGB's own thread pool colliding with
    concurrent outer candidates) alongside the exact number already in
    `_hover_extra`'s hover text."""
    series = {}
    for env in sorted({_env(record) for record in row_records}, key=_env_sort_key):
        env_records = sorted(
            (record for record in row_records if _env(record) == env), key=_n_jobs
        )
        points = [
            (_n_jobs(record), value, _hover_extra(record), _cpu_percent_mean(record))
            for record in env_records
            if (value := point_fn(record)) is not None
        ]
        if points:
            series[env] = points
    return series


def _row_cell_html(row_key: tuple, row_records: list[BenchmarkRecord]) -> str:
    duration_series = _env_series(row_records, _seconds_per_fit)
    if not duration_series:
        return ""

    plot = scaling_line_plot_html(
        duration_series,
        colors=ENV_COLORS,
        x_title="outer n_jobs",
        y_title=DURATION_METRIC,
        y_unit="s",
        x_log=True,
        # Marker size ~ mean CPU utilization (%) - see `_env_series`.
        size_domain=(0, 100),
    )
    title = escape(_row_title(row_key, _fit_shape(row_records)))
    subtitle = _row_subtitle_html(row_records[0])
    return f'<section class="plot-cell"><h3>{title}</h3>{subtitle}{plot}</section>'


def _software_tabs_html(hw_records: list[BenchmarkRecord]) -> str:
    cards = []
    for env in sorted({_env(record) for record in hw_records}, key=_env_sort_key):
        record = next(r for r in hw_records if _env(r) == env)
        summary = summarize_software_env(
            read_env("software", record.software_hash),
            _implementation(record),
            software_hash=record.software_hash,
        )
        summary["name"] = env
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
    # `_dedup_latest` already keeps only each (row, n_jobs)'s latest attempt,
    # so a still-listed failure here means the most recent run of that case
    # never produced timing data - not just some earlier retry that a later
    # success superseded.
    failed = [
        record for record in hw_records if record.failed_case is not None and not record.runs
    ]
    if not failed:
        return ""
    items = "".join(
        f"<li>{escape(_estimator(record))} / {escape(_dataset_name(record))} "
        f"(n_jobs={_n_jobs(record)}, {escape(_env(record))}): "
        f"{escape(_failure_reason(record))}</li>"
        for record in sorted(
            failed, key=lambda record: (_estimator(record), _dataset_name(record), _n_jobs(record))
        )
    )
    return f'<section class="panel"><h3>Failed cases</h3><ul class="compact">{items}</ul></section>'


def _row_sort_key(row_key: tuple) -> tuple:
    estimator, dataset, n_samples, n_features = row_key
    order = ESTIMATOR_ORDER.index(estimator) if estimator in ESTIMATOR_ORDER else len(ESTIMATOR_ORDER)
    return (order, dataset, n_samples or 0, n_features or 0)


def render_hardware_page(records: list[BenchmarkRecord], hardware_hash: str) -> str:
    hw_records = [record for record in records if record.hardware_hash == hardware_hash]
    if not hw_records:
        return '<section class="empty">No benchmark results for this hardware.</section>'

    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(hw_records))}</div>',
        f'<div class="page-row">{_software_tabs_html(hw_records)}</div>',
    ]
    failed_html = _failed_cases_html(hw_records)
    if failed_html:
        sections.append(f'<div class="page-row">{failed_html}</div>')

    by_row: dict[tuple, list[BenchmarkRecord]] = {}
    for record in hw_records:
        if record.runs:
            by_row.setdefault(_row_key(record), []).append(record)

    cells = [
        cell_html
        for row_key in sorted(by_row, key=_row_sort_key)
        if (cell_html := _row_cell_html(row_key, by_row[row_key]))
    ]
    if cells:
        grid = (
            '<section class="plot-grid" '
            'style="grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));">'
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
        title="hptuning outer-parallelism scalability",
        rows=[ABOUT_HTML, render_hardware_tabs(hardware_pages)],
    )
    output = output_dir / "hptuning_scalability.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
