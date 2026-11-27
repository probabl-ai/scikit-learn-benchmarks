"""`configs/hptuning_strong_scaling.py`: a fixed amount of search work under a CPU
affinity sweep, with `n_jobs=-1`.

Same plots as `gen_hptuning_taskset.py`, with the taskset points overlaid as
dotted lines to compare the two methodologies.
"""
from pathlib import Path

from dashboards import gen_hptuning_scalability as hptuning
from dashboards import gen_hptuning_taskset as taskset
from sklbench.reporting.envs import read_env
from sklbench.reporting.matching import (
    BenchmarkRecord, matches_source_configs, read_benchmark_records,
)


TITLE = "Hyper-parameter search strong scaling"
SOURCE_CONFIGS = ["configs/hptuning_strong_scaling.py", *taskset.SOURCE_CONFIGS]
SOURCE_ENVS = hptuning.SOURCE_ENVS

OVERLAY_SUFFIX = "fits grow with cores"

ABOUT_HTML = """<section class="panel">
  <p>This dashboard checks how much extra CPUs shorten a hyper-parameter search with
  <code>n_jobs=-1</code>. The search does the same fits at every point, so the
  <dfn>speed-up</dfn> compares the same job. In the
  <a href="hptuning_taskset.html">CPU affinity dashboard</a>, the fits grow with the cores
  instead. The plots overlay its points as dotted lines to compare the two methods.</p>
  <p>Each point restricts the process to some <dfn>physical cores</dfn>, with their
  <dfn term="smt sibling">SMT siblings</dfn>, through its <dfn>CPU affinity</dfn>. The
  searches are those of the <a href="hptuning_scalability.html">hyper-parameter search
  dashboard</a>: 8 candidates on as many <dfn>CV splits</dfn> as the machine has physical
  cores.</p>
  <p>The two methods give about the same time per fit at the same core count: ~90% of the
  points are within 10% of each other. On the laptop at 4 cores, most tree and
  <b>Ridge</b> rows fit ~15% to ~20% faster in the CPU affinity dashboard, probably because
  its 4 CV splits train on 3/4 of the data instead of 15/16. On the whole machine, where
  both run the same search, ~90% of the pairs of runs agree within 5%. The largest gaps
  are ~1.3x for the unstable <b>RandomForest</b> on susy on the Xeon (see below) and ~1.2x
  for <b>Ridge</b> on susy on the laptop.</p>
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
    The x-axis is the number of physical cores allowed (log scale), the y-axis is the wall
    time of the search divided by its number of <code><dfn>.fit()</dfn></code> calls (log
    scale), with one color per <dfn>environment</dfn>. Solid lines are this dashboard's
    searches, dotted lines the <a href="hptuning_taskset.html">CPU affinity dashboard</a>'s
    from the smallest mask of this one.</p>
    <p>The dashed line has the slope of ideal scaling: a line parallel to it means the time
    per fit falls as fast as the cores grow. The hover gives the speed-up over the
    environment's first point and the wall time. The marker size is the mean load of the
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


def _row_cell_html_fn(taskset_records: list[BenchmarkRecord]):
    taskset_by_row: dict[tuple, list[BenchmarkRecord]] = {}
    for record in taskset_records:
        if record.runs:
            key = (record.hardware_hash, hptuning._row_key(record))
            taskset_by_row.setdefault(key, []).append(record)

    def row_cell_html(row_key: tuple, row_records: list[BenchmarkRecord]) -> str:
        min_cores = min(_n_physical_cores(record) for record in row_records)
        # Below the smallest mask, the taskset points would stretch the y-axis
        # with nothing to compare them to.
        overlay_records = [
            record
            for record in taskset_by_row.get((row_records[0].hardware_hash, row_key), [])
            if taskset._n_physical_cores(record) >= min_cores
        ]
        overlay_series = taskset._env_series(overlay_records, taskset._n_physical_cores)
        return taskset.seconds_per_fit_cell_html(
            row_key, row_records, _n_physical_cores, overlay=(OVERLAY_SUFFIX, overlay_series)
        )

    return row_cell_html


def read_records() -> list[BenchmarkRecord]:
    return hptuning._dedup_latest(
        [
            record
            for record in read_benchmark_records()
            if matches_source_configs(record.case, ["configs/hptuning_strong_scaling.py"])
        ],
        sweep_point=_n_physical_cores,
    )


def generate(output_dir: Path) -> None:
    html = hptuning.render_dashboard(
        TITLE, ABOUT_HTML, read_records(), _row_cell_html_fn(taskset.read_records())
    )
    output = output_dir / "hptuning_strong_scaling.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
