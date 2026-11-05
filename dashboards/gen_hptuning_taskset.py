"""`configs/hptuning_taskset.py`: the searches of `gen_hptuning_scalability.py`
under a CPU affinity sweep, with `n_jobs=-1`.

Same page as `gen_hptuning_scalability.py`, except for the x-axis (physical
cores of the mask, since `n_jobs` is -1), the log y-axis and the CPU load (share
of the mask, not of the machine).
"""
from html import escape
import json
from pathlib import Path

from dashboards import gen_hptuning_scalability as hptuning
from sklbench.reporting.envs import read_env
from sklbench.reporting.html import scaling_line_plot_html
from sklbench.reporting.matching import (
    BenchmarkRecord, matches_source_configs, read_benchmark_records,
)


TITLE = "Hyper-parameter search with n_jobs=-1"
SOURCE_CONFIGS = ["configs/hptuning_taskset.py"]
SOURCE_ENVS = hptuning.SOURCE_ENVS

X_TITLE = "physical cores"

ABOUT_HTML = """<section class="panel">
  <p>With <code>n_jobs=-1</code>, a hyper-parameter search starts one worker per
  <dfn>logical CPU</dfn> it is allowed to run on. This dashboard checks how such a
  search scales with the CPUs it may use.</p>
  <p>Each point restricts the process to some <dfn>physical cores</dfn>, with their
  <dfn term="smt sibling">SMT siblings</dfn>, through its <dfn>CPU affinity</dfn> (like
  <code>taskset</code>). The searches are those of the
  <a href="hptuning_scalability.html">hyper-parameter search dashboard</a>: 8 candidates
  on as many <dfn>CV splits</dfn> as physical cores.</p>
  <p>On the benchmarked cases, compared with one physical core:</p>
  <ul>
    <li><b>RandomForest</b> and <b>ExtraTrees</b> scale the furthest: up to ~16x on the
    24 cores of the desktop and ~55x to ~60x on the 172 cores of the Xeon, or ~70x to
    ~110x with <b>scikit-learn-intelex</b>.</li>
    <li>The exception is <b>RandomForest</b> on susy on the Xeon. It goes from ~54x at
    86 cores to ~30x at 172 cores, except with <b>scikit-learn-intelex</b>.</li>
    <li><b>HistGradientBoosting</b> reaches ~70x to ~75x on the Xeon, but peaks at
    ~39x at 86 cores on the small ames_housing dataset.</li>
    <li>The <b>linear models</b> depend more on the dataset on the Xeon:
    <b>LogisticRegression</b> reaches ~25x to ~35x, and <b>Ridge</b> goes from ~8x on
    susy to ~55x on year_prediction_msd. <b>Ridge</b> on ames_housing peaks at ~21x at
    43 cores.</li>
    <li>On the laptop, the 16 cores give ~2x to ~7x.</li>
  </ul>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine and one plot per (estimator, dataset) pair.
    The x-axis is the number of physical cores allowed (log scale), the y-axis is
    the wall time of the search divided by its number of
    <code><dfn>.fit()</dfn></code> calls (log scale), with one color per <dfn>environment</dfn>.
    The marker size is the mean load of the allowed logical CPUs (100% means all of
    them busy).</p>
    <p>Inner parallelism:</p>
    <ul>
      <li>joblib limits <dfn>OpenMP</dfn> and <dfn>BLAS</dfn> threads to 1 in each
      worker. scikit-learn-intelex applies the same limit to its <dfn>oneTBB</dfn>
      threads.</li>
      <li>joblib does not limit the nested <code>n_jobs=-1</code> of RandomForest and
      ExtraTrees (see the
      <a href="hptuning_scalability.html">hyper-parameter search dashboard</a>).</li>
    </ul>
    <p>On hybrid machines, the affinity takes the lowest CPU ids first, whatever the
    core type.</p>
  </details>
</section>"""


def _n_physical_cores(record: BenchmarkRecord) -> int:
    # The config sets `n_splits` to the physical core count of the mask.
    return hptuning._n_splits(record)


def _mask_size(record: BenchmarkRecord) -> int:
    # `record.case` doesn't keep the "bench" section.
    raw = json.loads(record.record_path.read_text())
    return len(raw["case"]["bench"]["cpu_affinity"])


def _mask_load_percent(record: BenchmarkRecord) -> float | None:
    """`cpu_percent` covers the whole machine, which would make every small mask
    look idle."""
    cpu_percent = hptuning._cpu_percent_mean(record)
    if cpu_percent is None:
        return None
    n_logical_cpus = read_env("hardware", record.hardware_hash)["CPU"]["logical_cpus"]
    return min(cpu_percent * n_logical_cpus / _mask_size(record), 100)


def _row_cell_html(row_key: tuple, row_records: list[BenchmarkRecord]) -> str:
    series: dict[str, list[tuple]] = {}
    for record in row_records:
        seconds_per_fit = hptuning._seconds_per_fit(record)
        if seconds_per_fit is not None:
            series.setdefault(hptuning._env(record), []).append((
                _n_physical_cores(record),
                seconds_per_fit,
                hptuning._hover_extra(record, cpu_percent_fn=_mask_load_percent),
                _mask_load_percent(record),
            ))
    if not series:
        return ""

    plot = scaling_line_plot_html(
        series,
        colors=hptuning.ENV_COLORS,
        x_title=X_TITLE,
        y_title=hptuning.DURATION_METRIC,
        y_unit="s",
        x_log=True,
        y_log=True,
        size_domain=(0, 100),
    )
    title = escape(hptuning._row_title(row_key))
    subtitle = hptuning._row_subtitle_html(row_key, row_records)
    return f'<section class="plot-cell"><h3>{title}</h3>{subtitle}{plot}</section>'


def read_records() -> list[BenchmarkRecord]:
    return hptuning._dedup_latest(
        [
            record
            for record in read_benchmark_records()
            if matches_source_configs(record.case, SOURCE_CONFIGS)
        ],
        sweep_point=_n_physical_cores,
    )


def generate(output_dir: Path) -> None:
    html = hptuning.render_dashboard(TITLE, ABOUT_HTML, read_records(), _row_cell_html)
    output = output_dir / "hptuning_taskset.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
