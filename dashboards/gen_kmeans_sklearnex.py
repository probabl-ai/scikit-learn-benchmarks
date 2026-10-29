"""The KMeans cases of `configs/all_models.py` (`configs/all_models_kmeans.py`):
scikit-learn `main` vs a scikit-learn branch vs scikit-learn-intelex.

One table per hardware and scikit-learn build: `sklearn-dev` (OpenBLAS +
libgomp) with the conda-forge OpenBLAS and with OpenBLAS `develop` swapped into
the env, and `sklearn-dev-mkl` (MKL + libomp). Each table compares the branch
build and sklearnex (`intel` env) with the `main` build of that table.
"""
from pathlib import Path

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name, summarize_hardware_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    detailed_results_table_html,
    render_hardware_tabs,
)
from sklbench.reporting.matching import (
    MethodResult,
    append_iterations_warning,
    date_range,
    find_matches,
    matches_source_configs,
    read_all_results,
    read_failed_records,
)
from sklbench.reporting.utils import groupby


TITLE = "KMeans: scikit-learn main vs branch vs sklearnex"
SOURCE_CONFIGS = ["configs/all_models_kmeans.py"]
SOURCE_ENVS = ["sklearn-dev", "sklearn-dev-mkl", "intel"]

# Keyed by (pixi env, OpenBLAS version): both OpenBLAS builds ran in the
# `sklearn-dev` env, so only the version that threadpoolctl reports tells them
# apart.
SKLEARN_TABLES = {
    ("sklearn-dev", "0.3.34"): "OpenBLAS 0.3.34 + libgomp (sklearn-dev)",
    ("sklearn-dev", "0.3.34.dev"): "OpenBLAS develop + libgomp (sklearn-dev)",
    ("sklearn-dev-mkl", None): "MKL + libomp (sklearn-dev-mkl)",
}
SKLEARNEX_ENV = "intel"

ABOUT_HTML = """<section class="panel">
  <p>This page compares <b>KMeans</b> in scikit-learn <code>main</code>, in the
  <a href="https://github.com/cakedev0/scikit-learn/tree/kmeans/lloyd_no_blocking_sync">kmeans/lloyd_no_blocking_sync</a>
  branch (<a href="https://github.com/scikit-learn/scikit-learn/pull/35193">scikit-learn#35193</a>)
  and in <b>scikit-learn-intelex</b>, on the KMeans cases of
  <code>configs/all_models.py</code>: real datasets, default
  <dfn>hyperparameters</dfn> (k-means++ init, default <code>tol</code>).</p>
  <p>scikit-learn is built from source with three <dfn term="blas/openmp runtime">BLAS/OpenMP runtimes</dfn>, one
  table each: <dfn>OpenBLAS</dfn> 0.3.34 from conda-forge, OpenBLAS
  <code>develop</code> and MKL. The fit <dfn>speed-up</dfn> is relative to <code>main</code> built with
  the same runtime, and sklearnex is the same run in all tables. The init and
  Lloyd speed-ups only exist for scikit-learn, whose instrumented KMeans times
  each phase. The number of iterations varies between implementations and
  between repeats (no <code>random_state</code> is set), so check the
  <code>n_iter</code> column before reading a fit speed-up.</p>
  <p>Both scikit-learn builds use threadpoolctl <code>master</code>
  (<a href="https://github.com/joblib/threadpoolctl/commit/0e9d19d7cd529ba947dd2f82777a6b62ec56e2e5">0e9d19d</a>),
  as if <a href="https://github.com/joblib/threadpoolctl/pull/283">threadpoolctl#283</a>
  was released. With threadpoolctl 3.7.0, the
  <code>"sequential_blas_under_openmp"</code> limit that scikit-learn
  <code>main</code> uses since
  <a href="https://github.com/scikit-learn/scikit-learn/pull/35180">scikit-learn#35180</a>
  doesn't limit pthreads <dfn>OpenBLAS</dfn>, so the BLAS calls inside the
  OpenMP loops of KMeans run multi-threaded.</p>
  <p>OpenBLAS <code>develop</code> is
  <a href="https://github.com/OpenMathLib/OpenBLAS/commit/06608a6c83eb24ab837f7a8081c5f17e474fc15e">06608a6</a>,
  built with the conda-forge flags and swapped into the <code>sklearn-dev</code>
  env. It has <a href="https://github.com/OpenMathLib/OpenBLAS/pull/6105">OpenBLAS#6105</a>,
  which removes the contention between threads calling BLAS at the same time,
  as the OpenMP threads of KMeans do. No OpenBLAS release has it yet.</p>
  <p>On the Xeon server, every case runs on socket 0 (86 cores, 172 logical
  CPUs). scikit-learn runs 64 OpenMP threads, the cap of
  <code>configs/_real_datasets.py</code>
  (<a href="https://github.com/OpenMathLib/OpenBLAS/issues/5958">OpenBLAS#5958</a>),
  and sklearnex uses every logical CPU of the socket.</p>
  <p>On the benchmarked cases:</p>
  <ul>
    <li>On the Xeon socket, a Lloyd iteration of <b>scikit-learn</b>
    <code>main</code> takes ~20 ms whatever the data, even with 3 features and
    10 clusters. The branch makes each iteration 1.2x to 8.5x faster (52x on
    that smallest case with <b>MKL</b>), and the fits 1.2x to 4.3x faster.</li>
    <li>On the Xeon socket, <b>OpenBLAS</b> <code>develop</code> doesn't
    change <code>main</code>, but with the branch it makes the Lloyd
    iterations up to 4.3x faster than OpenBLAS 0.3.34 on the cases with few
    features or clusters (road_network and sift with 10 clusters, sift with 100
    clusters). With both, the fits are 1.5x to 6.8x faster than
    <code>main</code> with OpenBLAS 0.3.34.</li>
    <li>On the laptop, the branch doesn't change the time per Lloyd
    iteration (0.96x to 1.07x), except on road_network with 10 clusters and
    <b>MKL</b> (0.8x, on 0.6 ms iterations). <b>OpenBLAS</b>
    <code>develop</code> only helps on road_network with 10 clusters (~2x per
    iteration).</li>
    <li>On the laptop, <b>sklearnex</b> and <b>scikit-learn</b> are within
    ~1.4x of each other, except on the smallest workloads (sift and
    road_network with 10 clusters), where sklearnex is 1.6x to 2.9x faster.</li>
    <li>On the Xeon socket, with the branch, <b>scikit-learn</b> is faster than
    <b>sklearnex</b> on 6 of the 8 cases, up to ~4x on sift with 100 clusters
    (~7x with OpenBLAS <code>develop</code>). sklearnex stays faster on
    road_network (3 features): 1.9x to 3.9x with OpenBLAS 0.3.34, ~1.8x with
    OpenBLAS <code>develop</code>.</li>
  </ul>
</section>"""


def _result_matches(base: MethodResult, candidate: MethodResult):
    warnings = []
    append_iterations_warning(base, candidate, warnings)
    return True, warnings


def _env_of(build_name: str) -> str:
    return build_name.split("@", 1)[0]


def _variant_label(result: MethodResult) -> str:
    build_name = software_build_name(result.software_hash)
    if _env_of(build_name) == SKLEARNEX_ENV:
        return "sklearnex"
    return build_name.rsplit(":", 1)[-1]


def _is_main(build_name: str) -> bool:
    return build_name.rsplit(":", 1)[-1] == "main"


def _table_key(software_hash: str) -> tuple[str, str | None]:
    env = _env_of(software_build_name(software_hash))
    if env != "sklearn-dev":
        return env, None
    threadpools = read_env("software", software_hash)["threadpool_info"]
    return env, next(t["version"] for t in threadpools if t["internal_api"] == "openblas")


def _table_html(table_key, results: list[MethodResult], failed_records: list) -> str | None:
    def in_table(software_hash: str) -> bool:
        return (
            _env_of(software_build_name(software_hash)) == SKLEARNEX_ENV
            or _table_key(software_hash) == table_key
        )

    base_results = [
        r for r in results
        if _table_key(r.software_hash) == table_key
        and _is_main(software_build_name(r.software_hash))
    ]
    if not base_results:
        return None
    base_ids = {id(r) for r in base_results}
    candidates = [
        r for r in results if in_table(r.software_hash) and id(r) not in base_ids
    ]
    matches = find_matches(base_results, candidates, _result_matches)
    matched_bases = {id(m.base_result) for m in matches}
    base_keys = {r.minimal_match_key for r in base_results}
    failed = [
        (record, _variant_label(record))
        for record in failed_records
        if in_table(record.software_hash)
    ]
    table_html = detailed_results_table_html(
        SKLEARN_TABLES[table_key],
        groupby(matches, lambda m: m.matched_result.method),
        baseline_label="main",
        variant_label=_variant_label,
        failed_records=failed,
        unmatched_base_results=[r for r in base_results if id(r) not in matched_bases],
        unmatched_candidate_results=[
            r for r in candidates if r.minimal_match_key not in base_keys
        ],
        collapsible=False,
        variant_column_title="Build",
        fit_phases=["init", "lloyd"],
    )
    return f'<section class="panel"><h2>{SKLEARN_TABLES[table_key]}</h2>{table_html}</section>'


def render_hardware_page(results, failed_records, hardware_hash: str) -> str | None:
    results = [r for r in results if r.hardware_hash == hardware_hash]
    failed_records = [r for r in failed_records if r.hardware_hash == hardware_hash]
    tables = [
        html for table_key in SKLEARN_TABLES
        if (html := _table_html(table_key, results, failed_records)) is not None
    ]
    if not tables:
        return None
    return "".join([
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash))),
        *tables,
    ])


def generate(output_dir: Path) -> None:
    results = [
        r for r in read_all_results()
        if r.method == "fit" and matches_source_configs(r.case, SOURCE_CONFIGS)
    ]
    failed_records = [
        r for r in read_failed_records() if matches_source_configs(r.case, SOURCE_CONFIGS)
    ]
    hardware_hashes = sorted({r.hardware_hash for r in results})
    hardware_pages = [
        (HARDWARE_NAMES.get(hardware_hash, hardware_hash),
         render_hardware_page(results, failed_records, hardware_hash))
        for hardware_hash in hardware_hashes
    ]
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, render_hardware_tabs(hardware_pages)],
    )
    output = output_dir / "kmeans_sklearnex.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
