"""`configs/hptuning_taskset.py` with each search on all the cores of its machine,
to compare machines getting through many fits in parallel: one outer tab per software,
one inner tab per machine, any of which can be picked as the baseline.
"""
from pathlib import Path

from dashboards import HARDWARE_NAMES
from dashboards import gen_cross_comparison as fits
from dashboards import gen_hptuning_scalability as hptuning
from dashboards import gen_hptuning_taskset as taskset
from sklbench.reporting.comparison import CONTENT_HTML, baseline_pick_html, page_rows, write_data_files
from sklbench.reporting.envs import read_env, software_build_name, summarize_software_env
from sklbench.reporting.html import BASE_TEMPLATE, SOFTWARE_TEMPLATE
from sklbench.reporting.html.blocks import _slug_id
from sklbench.reporting.matching import BenchmarkRecord, MethodResult
from sklbench.reporting.utils import stable_json, without_keys


ABOUT_HTML = """<section class="panel">
  <p>Work in progress: this dashboard doesn't have many data points yet. We'll add more
  later.</p>
  <p>This dashboard compares machines getting through many fits in parallel, like a
  hyper-parameter search with <code>n_jobs=-1</code> on all the cores of the machine. A
  bigger machine often brings little to a <a href="cross_comparison.html">single fit</a>,
  but it can run more of them at once.</p>
  <p>Each search fits 8 <dfn>candidates</dfn> on as many <dfn>CV splits</dfn> as the machine
  has <dfn>physical cores</dfn>. Bigger machines have more fits to do, but each fit is the
  same on every machine (same data size and number of trees). So the time per fit compares
  machines: a 2x <dfn>speed-up</dfn> means twice as many fits per second.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a software tab. Below it, each machine has a tab describing the machine and its
    <dfn>software environment</dfn>, since versions can differ between machines. To change
    the <dfn>baseline</dfn>, open another machine tab and click "Pick as baseline".</p>
    <p>There's one color per machine. Each point is one search, and points above the dashed
    1x line get through fits faster than the baseline. Hover a point to see its dataset.
    "fit time" in "Detailed results" is the wall time of the search divided by its number
    of <code><dfn>.fit()</dfn></code> calls. Marker shapes and "Detailed results" work like in
    <a href="cross_comparison.html">the single fit dashboard</a>.</p>
    <p>Each machine scores its candidates on its own splits, so <dfn>metrics</dfn> aren't
    compared. scikit-learn-intelex trees use <dfn>exact splits</dfn> here, like scikit-learn.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li>The Xeon server has ~11x the cores of the Intel Core Ultra laptop, and most
      <b>random forests</b>, <b>extra trees</b> and <b>HistGradientBoosting</b> searches get
      ~9x to ~13x faster on it. <b>Ridge</b> on year_prediction_msd too, at ~12x to ~13x. With
      <b>scikit-learn-intelex</b>, <b>random forests</b> and <b>extra trees</b> reach ~13x to
      ~18x. On that server, its first repeat is up to ~3x slower than the next two, which
      the median leaves out.</li>
      <li>Other searches gain less on the Xeon server: ~5x to ~7x for
      <b>LogisticRegression</b>, ~4x to ~5x for <b>HistGradientBoosting</b> on the small
      ames_housing dataset, and ~2x to ~3x for <b>Ridge</b> on susy and ames_housing. Even so,
      these <b>linear models</b> are faster on the server than on the laptop, while a
      <a href="cross_comparison.html">single fit</a> is often slower on the server.</li>
      <li><b>RandomForestClassifier</b> on susy stays at ~6x on the Xeon server without
      <b>scikit-learn-intelex</b>, and it's unstable: its repeats go from ~20 ms to ~50 ms
      per fit.</li>
      <li>The Intel Core Ultra desktop has 1.5x the cores of the laptop. <b>Random forests</b>
      and <b>extra trees</b> are mostly ~2.5x to ~3x faster on it, and <b>HistGradientBoosting</b>
      ~2x to ~2.5x. <b>LogisticRegression</b> and <b>Ridge</b> on susy stay at ~1x, probably
      because memory bandwidth limits them (not measured).</li>
      <li>On the shortest fits, a very big machine doesn't help: <b>Ridge</b> on ames_housing
      takes ~6 ms per fit on both the desktop and the Xeon server.</li>
    </ul>
  </details>
</section>"""

TITLE = "Hardware comparison of parallelized fits"
SOURCE_CONFIGS = taskset.SOURCE_CONFIGS
SOURCE_ENVS = taskset.SOURCE_ENVS
ENTRY_PAGE = "hardware_parallel_fits.html"
DATA_DIR = "parallel_fits_data"
BASE_IMPLEMENTATION = "sklearn"

# The config sets the outer `n_jobs` to -1 and one CV split per physical core, so
# neither is part of a search's identity: each fit is the same on every machine.
_MATCH_EXCLUDED_NAMES = {"implementation", "max_bins", "n_jobs", "n_splits", "source_config"}


def search_match_key(result: MethodResult | BenchmarkRecord) -> str:
    return stable_json(_round_floats(without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES)))


def _round_floats(value):
    """`np.logspace` grids can differ in the last digit between numpy builds."""
    if isinstance(value, float):
        return float(f"{value:.12g}")
    if isinstance(value, dict):
        return {key: _round_floats(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _on_all_cores(record: BenchmarkRecord) -> bool:
    logical_cpus = read_env("hardware", record.hardware_hash)["CPU"]["logical_cpus"]
    return taskset._mask_size(record) == logical_cpus


def _search_result(record: BenchmarkRecord) -> MethodResult | None:
    """A search as a "fit" result whose times are the mean time per `.fit()` call
    (see `gen_hptuning_scalability._seconds_per_fit`), which divides out the
    machine-dependent `n_splits`."""
    n_fits = hptuning._n_fits(record)
    times = [run["duration_s"] * 1000 / n_fits for run in record.runs if "duration_s" in run]
    if not times:
        return None
    n_samples, n_features = hptuning._fit_shape([record])
    if n_samples is None:
        n_samples, n_features = hptuning._dataset_shape(record)
    case = without_keys(record.case, excluded_names={"bench"})
    case["implementation"] = record.case.get("implementation") or {"library": BASE_IMPLEMENTATION}
    return MethodResult(
        hardware_hash=record.hardware_hash,
        software=software_build_name(record.software_hash),
        software_hash=record.software_hash,
        method="fit",
        timestamp_recorded=record.timestamp_recorded,
        case=case,
        times=times,
        data_desc={"samples": n_samples, "features": n_features},
        metrics={},
        record=record,
    )


def _variant_of(result: MethodResult | BenchmarkRecord) -> fits.Variant:
    return fits.Variant(hptuning._env(result.record if isinstance(result, MethodResult) else result), result.hardware_hash)


def _env_html(variant_key: str, result: MethodResult) -> str:
    summary = summarize_software_env(
        read_env("software", result.software_hash),
        result.implementation,
        software_hash=result.software_hash,
    )
    summary["name"] = variant_key.split("__", 1)[0]
    return SOFTWARE_TEMPLATE.render(**summary)


def read_results() -> tuple[list[MethodResult], list[BenchmarkRecord]]:
    records = [
        record for record in taskset.read_records()
        if record.hardware_hash in HARDWARE_NAMES and _on_all_cores(record)
    ]
    results = [result for record in records if (result := _search_result(record))]
    return results, [record for record in records if record.failed_case is not None and not record.runs]


def generate(output_dir: Path) -> None:
    results, failed_records = read_results()
    results_by_variant = fits.group_by_variant(results, _variant_of)
    write_data_files(
        output_dir,
        DATA_DIR,
        {variant.key: items for variant, items in results_by_variant.items()},
        failed_by_variant={
            variant.key: items
            for variant, items in fits.group_by_variant(failed_records, _variant_of).items()
        },
        case_key=search_match_key,
        env_html=_env_html,
        hardware_html=fits.hardware_html,
    )
    colors = fits.hardware_colors()
    machine_order = list(HARDWARE_NAMES)
    outer_tabs = []
    for software in sorted({variant.software for variant in results_by_variant}, key=hptuning._env_sort_key):
        variants = sorted(
            (variant for variant in results_by_variant if variant.software == software),
            key=lambda variant: machine_order.index(variant.hardware_hash),
        )
        if len(variants) < 2:
            continue
        outer_tabs.append({
            "label": software,
            "slug": _slug_id(software),
            "emptyScope": software,
            "variants": [
                {
                    "key": variant.key,
                    "label": HARDWARE_NAMES[variant.hardware_hash],
                    "tabColor": colors[HARDWARE_NAMES[variant.hardware_hash]],
                    "pickHtml": baseline_pick_html(
                        HARDWARE_NAMES[variant.hardware_hash],
                        colors[HARDWARE_NAMES[variant.hardware_hash]],
                    ),
                }
                for variant in variants
            ],
        })
    config = {
        "dataDir": DATA_DIR,
        "outerTabs": outer_tabs,
        "colors": colors,
        "columnTitle": "Hardware",
        "labelOrder": list(HARDWARE_NAMES.values()),
        "matchNote": "Same setup as the baseline. Metrics aren't compared on this page",
    }
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        CONTENT_HTML,
        *page_rows(config),
    ])
    output = output_dir / ENTRY_PAGE
    output.write_text(html)
    print(f"Dashboard written to {output}")
