"""The KMeans cases of `configs/kmeans_init_lloyd.py`: scikit-learn `main` vs a
scikit-learn branch vs scikit-learn-intelex.

Five builds per hardware: `main` and the branch, each with OpenBLAS `develop`
swapped into `sklearn-dev` and with MKL (`sklearn-dev-mkl`), plus sklearnex
(`intel` env). Any of them can be picked as the baseline of the speed-up plots
and table, as on the software comparison page.
"""
from html import escape
from pathlib import Path

from dashboards import HARDWARE_NAMES
from dashboards.gen_softwares_comparison import BASELINE_SWITCH_SCRIPT
from sklbench.reporting.envs import (
    read_env, software_build_name, summarize_hardware_env, summarize_software_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    assemble_plots_in_grid,
    detailed_results_table_html,
    render_hardware_tabs,
    render_software_tabs,
    speedup_plot_html,
    variant_color_map,
)
from sklbench.reporting.matching import (
    BenchmarkRecord,
    MethodResult,
    append_iterations_warning,
    date_range,
    find_matches,
    matches_source_configs,
    read_all_results,
    read_failed_records,
)


TITLE = "KMeans: scikit-learn main vs branch vs sklearnex"
SOURCE_CONFIGS = ["configs/kmeans_init_lloyd.py"]
SOURCE_ENVS = ["sklearn-dev", "sklearn-dev-mkl", "intel"]

SKLEARNEX_ENV = "intel"
REF_LABELS = {"main": "main", "kmeans/lloyd_no_blocking_sync": "branch"}
# Keyed by (pixi env, OpenBLAS version): OpenBLAS `develop` is swapped into the
# `sklearn-dev` env, whose own OpenBLAS is 0.3.34, so only the version that
# threadpoolctl reports tells them apart.
RUNTIME_LABELS = {
    ("sklearn-dev", "0.3.34.dev"): "OpenBLAS develop",
    ("sklearn-dev-mkl", None): "MKL",
}
VARIANTS = [
    f"{ref}, {runtime}"
    for runtime in RUNTIME_LABELS.values()
    for ref in REF_LABELS.values()
] + ["sklearnex"]
DEFAULT_BASELINE = VARIANTS[0]
METHODS = ["fit", "predict"]
ROW = "KMeans"

ABOUT_HTML = """<section class="panel">
  <p>This page compares <b>KMeans</b> in scikit-learn <code>main</code>, in the
  <a href="https://github.com/cakedev0/scikit-learn/tree/kmeans/lloyd_no_blocking_sync">kmeans/lloyd_no_blocking_sync</a>
  branch (<a href="https://github.com/scikit-learn/scikit-learn/pull/35193">scikit-learn#35193</a>)
  and in <b>scikit-learn-intelex</b>, on the cases of
  <code>configs/kmeans_init_lloyd.py</code>: 17 workloads on real datasets
  (16 to 3072 features, 10 to 3000 clusters, two of them sparse), in float32
  and float64. Each fit runs the k-means++ init, then up to 100 Lloyd
  iterations (<code>tol=0</code>, <code>max_iter=100</code>).</p>
  <p>scikit-learn is built from source with two <dfn term="blas/openmp runtime">BLAS/OpenMP runtimes</dfn>:
  <dfn>OpenBLAS</dfn> <code>develop</code> and MKL. Each build has a tab in
  each machine tab. Click "Pick as baseline" in a build's tab to compare every
  other build with it. The default baseline is <code>main</code> with OpenBLAS
  <code>develop</code>.</p>
  <p>The plots show the fit and predict <dfn>speed-up</dfn> over the baseline,
  in log scale, one point per case and build: hover a point to see the case.
  The table under the plots also has the init and Lloyd speed-ups, for
  scikit-learn builds only, since only the instrumented scikit-learn KMeans times
  each phase. Repeat <i>i</i> of a case passes <code>random_state=i</code>, so
  the scikit-learn builds start from the same centers and run the same number
  of iterations. sklearnex draws its k-means++ centers with its own random
  generator, and strict convergence (labels that stop changing) can end its
  fits at a different iteration, so check the <code>n_iter</code> column
  before reading its fit speed-up.</p>
  <p>All the scikit-learn builds use threadpoolctl <code>master</code>
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
  CPUs) with the libraries' default thread counts: 172 OpenMP threads for
  scikit-learn and every logical CPU of the socket for sklearnex.</p>
  <p>On the benchmarked cases:</p>
  <ul>
    <li>On the Xeon socket, a Lloyd iteration of <b>scikit-learn</b>
    <code>main</code> takes at least ~35 ms on every dense case with 50,000
    samples or more, whatever the number of features or clusters. The branch
    makes the Lloyd iterations 1.6x to 23x faster on the dense cases, and the
    fits 1.5x to 13x faster.</li>
    <li>On the Xeon socket, <code>main</code> is much slower on the sparse
    cases: 0.2 to 2.9 s per Lloyd iteration, 9x to 38x slower than the branch.
    One of the five repeats of rcv1 in float64 didn't finish within the time
    limit, so that case has four.</li>
    <li>On the laptop, the branch doesn't change the dense cases much (0.83x to
    1.03x per Lloyd iteration, the lowest on iterations under 16 ms), but it
    makes the sparse ones faster: 1.5x to 3.9x on 20newsgroups, 1.7x on rcv1 in
    float64.</li>
    <li>On the laptop, <b>sklearnex</b> is faster than <b>scikit-learn</b> with
    the branch on the sparse cases (~1.7x per iteration on rcv1, ~1.35x on
    20newsgroups in float32), but not on the dense ones: it takes 0.83x to 1.8x
    the fit time per iteration of the branch with <b>OpenBLAS</b>
    <code>develop</code> (up to 2x that of <b>MKL</b>), the most on
    road_network and on sift with 16 features.</li>
    <li>The different numbers of iterations of <b>sklearnex</b> come from its
    k-means++ random draws, not from its Lloyd iterations: starting from the
    same centers, it runs the same number of iterations as <b>scikit-learn</b>
    on sift and road_network, and within the spread scikit-learn has between
    64 and 8 threads on nytimes and fashion_mnist, with the same inertia.</li>
  </ul>
</section>"""


def _result_matches(base: MethodResult, candidate: MethodResult):
    warnings = []
    append_iterations_warning(base, candidate, warnings)
    return True, warnings


def _env_of(build_name: str) -> str:
    return build_name.split("@", 1)[0]


def _runtime_key(software_hash: str) -> tuple[str, str | None]:
    env = _env_of(software_build_name(software_hash))
    if env != "sklearn-dev":
        return env, None
    threadpools = read_env("software", software_hash)["threadpool_info"]
    return env, next(t["version"] for t in threadpools if t["internal_api"] == "openblas")


def variant_label(result: MethodResult | BenchmarkRecord) -> str:
    build_name = software_build_name(result.software_hash)
    if _env_of(build_name) == SKLEARNEX_ENV:
        return "sklearnex"
    ref = REF_LABELS[build_name.rsplit(":", 1)[-1]]
    return f"{ref}, {RUNTIME_LABELS[_runtime_key(result.software_hash)]}"


def _speedup_view_html(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    *,
    baseline_label: str,
    variant_colors: dict[str, str],
) -> str:
    def is_baseline(result: MethodResult | BenchmarkRecord) -> bool:
        return variant_label(result) == baseline_label

    candidate_failed = [r for r in failed_records if not is_baseline(r)]
    plots = []
    matches_by_method = {}
    unmatched_base, unmatched_candidates = [], []
    for method in METHODS:
        base_results = [r for r in results if r.method == method and is_baseline(r)]
        candidates = [r for r in results if r.method == method and not is_baseline(r)]
        matches = find_matches(base_results, candidates, _result_matches)
        matches_by_method[method] = matches
        matched_bases = {id(m.base_result) for m in matches}
        base_keys = {r.minimal_match_key for r in base_results}
        unmatched_base += [r for r in base_results if id(r) not in matched_bases]
        unmatched_candidates += [r for r in candidates if r.minimal_match_key not in base_keys]
        plots.append({
            "category": ROW,
            "method": method,
            "case_count": len({m.base_result.minimal_match_key for m in matches}),
            "plot": speedup_plot_html(
                matches,
                baseline_label=baseline_label,
                variant_colors=variant_colors,
                trace_variant=lambda match: variant_label(match.matched_result),
                variant_sort_key=VARIANTS.index,
                failed_records=candidate_failed,
            ),
        })
    table_html = detailed_results_table_html(
        ROW,
        matches_by_method,
        baseline_label=baseline_label,
        variant_label=variant_label,
        failed_records=[(record, variant_label(record)) for record in failed_records],
        unmatched_base_results=unmatched_base,
        unmatched_candidate_results=unmatched_candidates,
        open=True,
        variant_column_title="Build",
        fit_phases=["init", "lloyd"],
    )
    return assemble_plots_in_grid(
        plots,
        rows={"category": [ROW]},
        columns={"method": METHODS},
        details_by_row={ROW: table_html},
    )


def render_hardware_page(
    results: list[MethodResult],
    failed_records: list[BenchmarkRecord],
    hardware_hash: str,
    *,
    variant_colors: dict[str, str],
) -> str | None:
    results = [r for r in results if r.hardware_hash == hardware_hash]
    failed_records = [r for r in failed_records if r.hardware_hash == hardware_hash]
    present = {variant_label(r) for r in results}
    labels = [label for label in VARIANTS if label in present]
    if len(labels) < 2:
        return None
    default_baseline = DEFAULT_BASELINE if DEFAULT_BASELINE in labels else labels[0]

    baseline_views = []
    for baseline_label in labels:
        view_html = _speedup_view_html(
            results, failed_records,
            baseline_label=baseline_label, variant_colors=variant_colors,
        )
        active = " active" if baseline_label == default_baseline else ""
        baseline_views.append(
            f'<div class="baseline-view{active}" data-baseline="{escape(baseline_label)}">'
            f"{view_html}</div>"
        )

    first_result_by_label = {}
    for r in results:
        first_result_by_label.setdefault(variant_label(r), r)
    softwares = []
    for label in labels:
        r = first_result_by_label[label]
        summary = summarize_software_env(
            read_env("software", r.software_hash),
            r.implementation,
            software_hash=r.software_hash,
        )
        summary["name"] = label
        summary["actions_html"] = (
            '<button type="button" class="baseline-pick" '
            f'data-baseline="{escape(label)}" style="--variant-color: {variant_colors[label]}">'
            "Pick as baseline</button>"
        )
        softwares.append(summary)

    rows = [
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash))),
        render_software_tabs(
            [SOFTWARE_TEMPLATE.render(**summary) for summary in softwares],
            variant_colors=variant_colors,
            labels=labels,
        ),
        "".join(baseline_views),
    ]
    return (
        '<div class="baseline-switch">'
        + "".join(f'<div class="page-row">{row}</div>' for row in rows)
        + "</div>"
    )


def generate(output_dir: Path) -> None:
    results = [
        r for r in read_all_results()
        if r.method in METHODS and matches_source_configs(r.case, SOURCE_CONFIGS)
    ]
    failed_records = [
        r for r in read_failed_records() if matches_source_configs(r.case, SOURCE_CONFIGS)
    ]
    variant_colors = variant_color_map(VARIANTS)
    hardware_hashes = sorted({r.hardware_hash for r in results})
    hardware_pages = [
        (HARDWARE_NAMES.get(hardware_hash, hardware_hash),
         render_hardware_page(
             results, failed_records, hardware_hash, variant_colors=variant_colors
         ))
        for hardware_hash in hardware_hashes
    ]
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, render_hardware_tabs(hardware_pages), BASELINE_SWITCH_SCRIPT],
    )
    output = output_dir / "kmeans_sklearnex.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
