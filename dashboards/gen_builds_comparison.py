from pathlib import Path

from dashboards import HARDWARE_NAMES, GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS
from sklbench.reporting.comparison import (
    CONTENT_HTML,
    page_rows,
    same_machine_case_key,
    write_data_files,
)
from sklbench.reporting.envs import (
    is_sklearn_dev_build, is_vanilla_sklearn, read_env, software_build_name,
    summarize_software_env, summarize_hardware_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    variant_color_map,
)
from sklbench.reporting.html.blocks import _slug_id
from sklbench.reporting.matching import (
    read_all_results, read_failed_records, BenchmarkRecord, MethodResult,
    matches_source_configs,
)


BASE_IMPLEMENTATION = "sklearn"
DATA_DIR = "builds_data"
ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares <dfn>builds</dfn> of plain scikit-learn. The code is the
  same, only the <dfn>BLAS/OpenMP runtime</dfn> changes. The <dfn>PyPI wheel</dfn> is the <dfn>baseline</dfn>.
  The <dfn>conda-forge</dfn> build names encode two choices:
  </p>
  <ul>
    <li><code>libgomp</code> or <code>libomp</code>: the <dfn>OpenMP</dfn> runtime of scikit-learn's own
    loops (<dfn>libgomp</dfn> or <dfn>libomp</dfn>).</li>
    <li><code>openblas</code> or <code>openblas-omp</code>: how <dfn>OpenBLAS</dfn> runs its threads,
    with pthreads (the conda-forge default) or with OpenMP. In <code>libomp-openblas-omp</code>,
    OpenBLAS and scikit-learn share the same libomp threads. <code>mkl</code> is <dfn>MKL</dfn>,
    on libomp too.</li>
  </ul>
  <p><code>sklearn-cf-default</code> installs conda-forge's scikit-learn without pinning
  either, which gives the same libraries as <code>sklearn-cf-libgomp-openblas</code>.</p>
  <p>The "Threadpools" and "OpenMP" lines of each environment tab show what
  was loaded at run time. <code>sklearn-pypi</code> lists two <code>libscipy_openblas</code>:
  the NumPy and SciPy wheels each ship their own copy of OpenBLAS, and both get
  loaded, often in different versions. NumPy's matrix products run on NumPy's
  copy, SciPy's linear algebra (used by some solvers) on SciPy's. On
  conda-forge, NumPy and SciPy share a single OpenBLAS.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Read each cell like in
    <a href="per_hardware.html">the software/implementations dashboard</a>:
    <dfn>fit</dfn> or <dfn>predict</dfn> <dfn>speed-up</dfn> (log scale) per estimator category, one line per
    build. Marker shapes mean the same: a square (■) flags a setup
    difference, an open diamond (◇) different <dfn>metrics</dfn>.
    </p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <ul>
      <li><b><dfn>MKL</dfn></b> is a good pick for <b>linear models</b>: it speeds up <dfn term="blas">BLAS</dfn>-bound
      linear model fits, commonly by 1.3x to 1.5x.</li>
      <li>On the laptop, every <b><dfn>conda-forge</dfn></b> build fits
      <b>LogisticRegression</b> much faster than the <b><dfn>PyPI</dfn></b> one, up to
      ~10x on some datasets. This comes from the <b><dfn>OpenBLAS</dfn></b> shipped in scipy
      and is going to be fixed in scipy soon, see
      <a href="https://github.com/scipy/scipy/pull/26193#issuecomment-5886021070">scipy#26193</a>.</li>
      <li><b>HistGradientBoosting</b> varies more between builds. On laptops, this
      comes from differences in <dfn>active wait</dfn> (how long idle <dfn>OpenMP threads</dfn> spin
      before sleeping), which <b>conda-forge</b> disables by default. Without active
      wait, HGB fits on small and medium datasets can be much slower. On the
      high-end server, the <b><dfn>LLVM/Intel OpenMP</dfn></b> runtimes seem faster than
      <b><dfn>libgomp</dfn></b>. See
      <a href="https://github.com/scikit-learn/scikit-learn/issues/34764">scikit-learn#34764</a>
      for the analysis, and
      <a href="https://github.com/scikit-learn/scikit-learn/pull/34935">scikit-learn#34935</a>
      for a fix in progress, based on the insights from the
      <a href="hgb_scaling.html">HistGradientBoosting thread-scalability breakdown</a> plots.</li>
      <li><b>ExtraTrees</b> fits are ~25% slower on <b>conda-forge</b> builds.
      <a href="https://github.com/scikit-learn/scikit-learn/pull/34876">scikit-learn#34876</a>
      fixes it and will land in the next release.</li>
    </ul>
  </details>
</section>"""


def is_other_library_build(result: MethodResult | BenchmarkRecord) -> bool:
    return result.implementation.library != BASE_IMPLEMENTATION


def is_array_api_variant(result: MethodResult | BenchmarkRecord) -> bool:
    return result.implementation.data_library is not None


def is_sklearn_dev_variant(result: MethodResult | BenchmarkRecord) -> bool:
    """`sklearn-dev`/`sklearn-dev-libomp`/... are one-off git-checkout builds
    (a specific commit/PR branch, not a stable environment build), so they're
    excluded here rather than folded in as more build variants."""
    return is_sklearn_dev_build(software_build_name(result.software_hash))


def build_variant(result: MethodResult | BenchmarkRecord) -> str:
    """Label results/records by their sklearn build (pixi env), since every
    result compared on this dashboard has implementation.short_name ==
    'sklearn' regardless of build."""
    return software_build_name(result.software_hash)


def _is_compared(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.hardware_hash in HARDWARE_NAMES
        and matches_source_configs(result.case, SOURCE_CONFIGS)
        and not is_other_library_build(result)
        and not is_array_api_variant(result)
        and not is_sklearn_dev_variant(result)
    )


# Page header and index page label.
TITLE = "Builds comparison"
SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
SOURCE_ENVS = GENERAL_SOURCE_ENVS


def generate(output_dir: Path) -> None:
    results = [res for res in read_all_results() if _is_compared(res)]
    failed_records = [record for record in read_failed_records() if _is_compared(record)]

    outer_tabs = []
    results_by_variant: dict[str, list[MethodResult]] = {}
    baseline_labels: dict[str, str] = {}
    for hardware_hash, hardware_name in HARDWARE_NAMES.items():
        machine_results = [res for res in results if res.hardware_hash == hardware_hash]
        base_results = [res for res in machine_results if is_vanilla_sklearn(res.software_hash)]
        other_results = [res for res in machine_results if not is_vanilla_sklearn(res.software_hash)]
        machine_failed = [
            record for record in failed_records
            if record.hardware_hash == hardware_hash and not is_vanilla_sklearn(record.software_hash)
        ]
        methods = ("fit", "predict", "preprocessing")
        base_keys = {res.minimal_match_key for res in base_results if res.method in methods}
        other_keys = {res.minimal_match_key for res in other_results}
        if not base_results or not (machine_failed or base_keys & other_keys):
            continue

        baseline_label = build_variant(base_results[0])
        baseline_labels[hardware_hash] = baseline_label
        labels = [baseline_label, *sorted({build_variant(res) for res in other_results})]
        for res in machine_results:
            label = baseline_label if is_vanilla_sklearn(res.software_hash) else build_variant(res)
            results_by_variant.setdefault(f"{hardware_hash}__{label}", []).append(res)
        colors = variant_color_map(sorted(labels[1:]))
        outer_tabs.append({
            "label": hardware_name,
            "slug": _slug_id(hardware_name),
            "emptyScope": "this hardware",
            "topHtml": HARDWARE_TEMPLATE.render(
                summarize_hardware_env(read_env("hardware", hardware_hash))
            ),
            "colors": colors,
            "variants": [
                {
                    "key": f"{hardware_hash}__{label}",
                    "label": label,
                    "plotted": True,
                    "tabColor": colors.get(label),
                }
                for label in labels
            ],
        })

    failed_by_variant: dict[str, list[BenchmarkRecord]] = {}
    for record in failed_records:
        if record.hardware_hash in baseline_labels:
            label = (
                baseline_labels[record.hardware_hash]
                if is_vanilla_sklearn(record.software_hash)
                else build_variant(record)
            )
            failed_by_variant.setdefault(f"{record.hardware_hash}__{label}", []).append(record)

    def env_html(variant_key: str, result: MethodResult) -> str:
        summary = summarize_software_env(
            read_env("software", result.software_hash),
            result.implementation,
            software_hash=result.software_hash,
        )
        summary["name"] = variant_key.split("__", 1)[1]
        return SOFTWARE_TEMPLATE.render(**summary)

    write_data_files(
        output_dir,
        DATA_DIR,
        results_by_variant,
        failed_by_variant=failed_by_variant,
        case_key=same_machine_case_key,
        env_html=env_html,
    )
    config = {
        "page": "baseline-tabs",
        "dataDir": DATA_DIR,
        "outerTabs": outer_tabs,
        "columnTitle": "Variant name",
        "splitsFromParams": False,
        "searchNote": "",
    }
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, CONTENT_HTML, *page_rows(config)],
    )

    output = output_dir / "builds_comparison.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
