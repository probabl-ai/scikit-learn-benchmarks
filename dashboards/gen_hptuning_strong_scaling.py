"""`configs/hptuning_strong_scaling.py`: a fixed amount of search work under a CPU
affinity sweep, with `n_jobs=-1`.

Same page as `gen_hptuning_taskset.py`, except for the y-axis: speed-up over the
smallest mask, against ideal scaling.
"""
from html import escape
from pathlib import Path

from dashboards import gen_hptuning_scalability as hptuning
from dashboards import gen_hptuning_taskset as taskset
from sklbench.reporting.envs import read_env
from sklbench.reporting.html import scaling_line_plot_html
from sklbench.reporting.matching import (
    BenchmarkRecord, matches_source_configs, read_benchmark_records,
)


TITLE = "Hyper-parameter search strong scaling"
SOURCE_CONFIGS = ["configs/hptuning_strong_scaling.py"]
SOURCE_ENVS = hptuning.SOURCE_ENVS

ABOUT_HTML = """<section class="panel">
  <p>This dashboard checks how much extra CPUs shorten a hyper-parameter search with
  <code>n_jobs=-1</code>. The search does the same fits at every point, so the
  <dfn>speed-up</dfn> compares the same job. In the
  <a href="hptuning_taskset.html">CPU affinity dashboard</a>, the fits grow with the cores
  instead.</p>
  <p>Each point restricts the process to some <dfn>physical cores</dfn>, with their
  <dfn term="smt sibling">SMT siblings</dfn>, through its <dfn>CPU affinity</dfn>. The
  searches are those of the <a href="hptuning_scalability.html">hyper-parameter search
  dashboard</a>: 8 candidates on as many <dfn>CV splits</dfn> as the machine has physical
  cores.</p>
  <p>On the benchmarked cases, comparing the whole machine with the smallest mask:</p>
  <ul>
    <li>On the Xeon, from 22 to 172 cores (ideal: ~7.8x), <b>HistGradientBoosting</b> on
    kddcup09_churn and year_prediction_msd and <b>Ridge</b> on year_prediction_msd reach
    ~4x to ~4.4x.</li>
    <li><b>RandomForest</b> and <b>ExtraTrees</b> reach ~2.6x to ~3.2x on the Xeon, and ~4x
    to ~5x with <b>scikit-learn-intelex</b>. The exception is <b>RandomForest</b> on susy
    without <b>scikit-learn-intelex</b>, at ~1.7x to ~1.9x. It is unstable at 172 cores: a
    search takes 28 s to 68 s depending on the repeat, against a steady ~34 s at 86
    cores.</li>
    <li>On the Xeon, <b>LogisticRegression</b> reaches ~1.9x to ~3.2x, and <b>Ridge</b> on
    susy only ~1.3x to ~1.5x.</li>
    <li>Extra cores hurt the shortest fits on the Xeon, where the load falls to ~10% to
    ~20% at 172 cores. On ames_housing, <b>Ridge</b> goes from ~1.2x at 43 cores to ~0.7x
    to ~0.8x at 172 cores, and <b>HistGradientBoosting</b> from ~1.6x at 86 cores to ~1.3x
    to ~1.4x at 172 cores.</li>
    <li>On the Xeon, the CPU affinity fills the <dfn>NUMA nodes</dfn> in order. From 22 to
    43 cores, the <b>linear models</b> and <b>HistGradientBoosting</b> only gain ~1x to
    ~1.3x, against ~1.6x for most <b>RandomForest</b> and <b>ExtraTrees</b> rows. The
    <a href="hptuning_taskset.html">CPU affinity dashboard</a> has the placement test.</li>
    <li>On the desktop, from 6 to 24 cores (ideal: 4x), <b>RandomForest</b> and
    <b>ExtraTrees</b> mostly reach ~2.3x to ~2.8x and <b>HistGradientBoosting</b> ~1.7x to
    ~2.4x. <b>LogisticRegression</b> and <b>Ridge</b> on susy stay at ~1x with a ~75% to
    ~90% load, probably because memory bandwidth limits them (not measured).</li>
    <li>On the laptop, from 4 to 16 cores (ideal: 4x), speed-ups are ~1.5x to ~2.2x, except
    <b>Ridge</b> on susy at ~1x to ~1.2x.</li>
    <li>The ideal assumes equal cores. On the desktop and the laptop, the smallest mask only
    has performance cores, and most of the cores added after it are efficiency cores.</li>
  </ul>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine and one plot per (estimator, dataset) pair.
    The x-axis is the number of physical cores allowed (log scale). The y-axis is the wall
    time of the search on the smallest mask divided by its wall time at this point (log
    scale), with one color per <dfn>environment</dfn> and a dashed line for ideal scaling.</p>
    <p>Each environment has its own baseline, so a higher line means better scaling, not a
    faster search. The hover gives the wall time. The marker size is the mean load of the
    allowed logical CPUs (100% means all of them busy).</p>
    <p>On hybrid machines, the affinity takes the lowest CPU ids first, whatever the core
    type. The <a href="hptuning_taskset.html">CPU affinity dashboard</a> describes the
    inner parallelism, which is the same here.</p>
  </details>
</section>"""


def _n_physical_cores(record: BenchmarkRecord) -> int:
    # `n_splits` is the physical core count of the machine, whatever the mask.
    n_logical_cpus = read_env("hardware", record.hardware_hash)["CPU"]["logical_cpus"]
    return round(taskset._mask_size(record) * hptuning._n_splits(record) / n_logical_cpus)


def _row_cell_html(row_key: tuple, row_records: list[BenchmarkRecord]) -> str:
    cores = [_n_physical_cores(record) for record in row_records]
    min_cores, max_cores = min(cores), max(cores)
    baselines = {
        hptuning._env(record): hptuning._seconds_per_fit(record)
        for record, n_cores in zip(row_records, cores)
        if n_cores == min_cores
    }

    series: dict[str, list[tuple]] = {}
    for record, n_cores in zip(row_records, cores):
        env = hptuning._env(record)
        baseline = baselines.get(env)
        seconds_per_fit = hptuning._seconds_per_fit(record)
        if baseline is None or seconds_per_fit is None:
            continue
        series.setdefault(env, []).append((
            n_cores,
            baseline / seconds_per_fit,
            hptuning._hover_extra(record, cpu_percent_fn=taskset._mask_load_percent),
            taskset._mask_load_percent(record),
        ))
    if not series:
        return ""

    plot = scaling_line_plot_html(
        series,
        colors=hptuning.ENV_COLORS,
        x_title=taskset.X_TITLE,
        y_title=f"speed-up over {min_cores} cores",
        y_unit="x",
        x_log=True,
        y_log=True,
        reference_lines={"ideal": [[(min_cores, 1), (max_cores, max_cores / min_cores)]]},
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
    output = output_dir / "hptuning_strong_scaling.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
