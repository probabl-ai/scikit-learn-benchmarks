"""`configs/hptuning_taskset.py`: the searches of `gen_hptuning_scalability.py`
under a CPU affinity sweep, with `n_jobs=-1` and, on SMT cores, with one worker
per physical core.

Same page as `gen_hptuning_scalability.py`, except for the x-axis (physical
cores of the mask, since `n_jobs` is -1), the `n_jobs` variants and the CPU load
(share of the mask, not of the machine).
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
LOGICAL_LABEL = "n_jobs=-1"
PHYSICAL_LABEL = "n_jobs=physical cores"
COLORS = {
    f"{env}, {n_jobs_label}": color
    for env, color in hptuning.ENV_COLORS.items()
    for n_jobs_label in (LOGICAL_LABEL, PHYSICAL_LABEL)
}
LINE_DASHES = {f"{env}, {PHYSICAL_LABEL}": "dash" for env in hptuning.ENV_COLORS}

ABOUT_HTML = """<section class="panel">
  <p>With <code>n_jobs=-1</code>, a hyper-parameter search starts one worker per
  <dfn>logical CPU</dfn> it is allowed to run on, not one per <dfn>physical core</dfn>.
  With <dfn>SMT</dfn>, that is twice as many workers. This dashboard checks how
  <code>n_jobs=-1</code> scales with the CPUs a search may use, and if one worker per
  physical core is faster.</p>
  <p>Each point restricts the process to some physical cores, with their
  <dfn term="smt sibling">SMT siblings</dfn>, through its <dfn>CPU affinity</dfn> (like
  <code>taskset</code>). The searches are those of the
  <a href="hptuning_scalability.html">hyper-parameter search dashboard</a>: 8 candidates
  on as many <dfn>CV splits</dfn> as physical cores. Both variants run the same
  fits, so their times compare directly. Machines without SMT only run
  <code>n_jobs=-1</code>, since the other variant has the same number of workers.</p>
  <p>On the benchmarked cases, all on the Xeon server since it is the only machine
  with SMT:</p>
  <ul>
    <li><code>n_jobs=-1</code> is at most ~1.2x slower than one worker per physical
    core and up to ~2.4x faster, so there is little reason to replace it.</li>
    <li>The <b>linear models</b> gain the most up to 22 physical cores:
    <code>n_jobs=-1</code> is typically ~1.6x faster with the <b><dfn>PyPI</dfn></b>
    build, ~1.3x faster with the <b><dfn>MKL conda-forge build</dfn></b> and about as
    fast with <b>scikit-learn-intelex</b>. From 43 cores, the gap depends on the case,
    from ~1.4x faster to ~1.2x slower.</li>
    <li><b>HistGradientBoosting</b> with <code>n_jobs=-1</code> is ~1.1x faster up to
    22 cores and ~1.1x to ~1.2x slower from 43 cores. On the small ames_housing
    dataset, it is up to ~2x faster up to 43 cores.</li>
    <li><b>RandomForest</b> and <b>ExtraTrees</b> barely change: the typical gap stays
    within ~3%.</li>
    <li>With <code>n_jobs=-1</code>, more cores don't always help. <b>RandomForest</b>
    on susy goes from ~54x faster than one core at 86 cores to ~30x at 172 cores,
    while <b>scikit-learn-intelex</b> reaches ~110x. <b>Ridge</b> on ames_housing peaks
    at ~21x at 43 cores.</li>
  </ul>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine and one plot per (estimator, dataset) pair.
    The x-axis is the number of physical cores allowed (log scale), the y-axis is
    the wall time of the search divided by its number of
    <code><dfn>.fit()</dfn></code> calls (log scale), with one color per <dfn>environment</dfn>. The
    solid line is <code>n_jobs=-1</code>, the dashed line is one worker per physical
    core: a dashed line below the solid one means that workers on SMT siblings
    slow the search down. The marker size is the mean load of the allowed logical
    CPUs (100% means all of them busy).</p>
    <p>The variants also differ in inner parallelism:</p>
    <ul>
      <li>joblib limits <dfn>OpenMP</dfn> and <dfn>BLAS</dfn> threads to
      <code>n_logical_cpus // n_jobs</code> in each worker: 1 thread with
      <code>n_jobs=-1</code>, 2 threads with one worker per physical core.
      scikit-learn-intelex applies the same limit to its <dfn>oneTBB</dfn> threads.</li>
      <li>joblib does not limit the nested <code>n_jobs=-1</code> of RandomForest and
      ExtraTrees (see the
      <a href="hptuning_scalability.html">hyper-parameter search dashboard</a>).</li>
    </ul>
    <p>The worker count per physical core is computed in the benchmark config:
    under a CPU affinity, <code>joblib.cpu_count(only_physical_cores=True)</code>
    returns the logical CPU count
    (<a href="https://github.com/joblib/loky/pull/651">loky#651</a>). On hybrid
    machines, the affinity takes the lowest CPU ids first, whatever the core
    type.</p>
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


def _series_label(record: BenchmarkRecord) -> str:
    n_jobs_label = LOGICAL_LABEL if hptuning._n_jobs(record) == -1 else PHYSICAL_LABEL
    return f"{hptuning._env(record)}, {n_jobs_label}"


def _row_cell_html(row_key: tuple, row_records: list[BenchmarkRecord]) -> str:
    series: dict[str, list[tuple]] = {}
    for record in row_records:
        seconds_per_fit = hptuning._seconds_per_fit(record)
        if seconds_per_fit is not None:
            series.setdefault(_series_label(record), []).append((
                _n_physical_cores(record),
                seconds_per_fit,
                hptuning._hover_extra(record, cpu_percent_fn=_mask_load_percent),
                _mask_load_percent(record),
            ))
    if not series:
        return ""

    plot = scaling_line_plot_html(
        series,
        colors=COLORS,
        line_dashes=LINE_DASHES,
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
        sweep_point=lambda record: (_n_physical_cores(record), hptuning._n_jobs(record)),
    )


def generate(output_dir: Path) -> None:
    html = hptuning.render_dashboard(TITLE, ABOUT_HTML, read_records(), _row_cell_html)
    output = output_dir / "hptuning_taskset.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
