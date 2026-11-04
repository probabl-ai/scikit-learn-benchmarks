from pathlib import Path

from dashboards import HARDWARE_NAMES, GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS
from sklbench.reporting.comparison import (
    CONTENT_HTML,
    CUML_TREE_NOTE,
    PREPROCESSING_TOGGLE_HTML,
    baseline_pick_html,
    expanded_note,
    page_rows,
    same_machine_case_key,
    tree_binning_note,
    write_data_files,
)
from sklbench.reporting.envs import (
    is_vanilla_sklearn, read_env, software_build_name, summarize_software_env,
    summarize_hardware_env,
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
# Alternative scikit-learn builds offered as a baseline but never plotted as
# a candidate: next to every other variant they'd make the plots unreadable.
BASELINE_ONLY_BUILDS = {"sklearn-cf-mkl"}
# base.css's --muted, so these tabs don't look like a plotted variant.
BASELINE_ONLY_TAB_COLOR = "#48566b"
ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares <dfn>implementations</dfn> on the same machine: stock
  scikit-learn, scikit-learn-intelex and <dfn>Array API</dfn> backends (<dfn>PyTorch</dfn>, <dfn>dpnp</dfn>,
  <dfn>CuPy</dfn>). Since the hardware is fixed, a <dfn>speed-up</dfn> comes from the software:
  faster algorithms, better use of the CPU (<dfn>vectorization</dfn>, <dfn>threading</dfn>), or
  offloading to the GPU.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a machine tab. The top of the tab describes the machine and each
    <dfn>software environment</dfn> (versions, <dfn>BLAS</dfn> and <dfn>OpenMP</dfn> libraries). The <dfn>baseline</dfn>
    defaults to stock scikit-learn from <dfn>PyPI</dfn>. To change it, open another
    software tab and click "Pick as baseline". The <dfn>conda-forge</dfn> <dfn>MKL</dfn> build
    (<code>sklearn-cf-mkl</code>) can be picked as a baseline too, but isn't
    plotted otherwise.</p>
    <p>The plots are arranged in a grid: one row per estimator category
    (linear, tree-based, clustering) and one column for <code><dfn>fit</dfn></code> and
    one for <code><dfn>predict</dfn></code>. In each plot:</p>
    <ul>
      <li>the x-axis lists the estimators, with one color per implementation.
      An implementation keeps its color on every machine and for every
      baseline;</li>
      <li>each point is one benchmark case (dataset and <dfn>hyperparameters</dfn>);</li>
      <li>the y-axis is the speed-up over the baseline, in log scale. Points
      above the dashed 1x line are faster than the baseline.</li>
    </ul>
    <p>The marker shape flags cases to double check:</p>
    <ul>
      <li>circle (●): <dfn>metrics</dfn> and setup match the baseline;</li>
      <li>square (■): metrics match but the setup differs, for instance a
      different <dfn>solver</dfn> or number of iterations, scikit-learn-intelex using
      <dfn>histogram-based splits</dfn>, or cuML Random Forest using
      <dfn>quantile-based splits</dfn>, where scikit-learn uses
      <dfn term="exact splits">exact ones</dfn>;</li>
      <li>open diamond (◇): the metrics differ;</li>
      <li>grey point (<span style="color: grey">●</span>):
      scikit-learn-intelex <dfn>fell back</dfn> to scikit-learn;</li>
      <li>cross (✕): failed run.</li>
    </ul>
    <p>The notes under each plot count these cases.</p>
    <p>Hover a point to see its case. Click it to find its row in "Detailed
    results", the table below each row, which has the exact timings and
    metrics and can be filtered by column.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li><b><code>sklearnex-cpu</code></b> is the most consistently fast option.
      <b>Linear models</b> gain less than <b>tree-based models</b> and <b>KMeans</b>.</li>
      <li><b>sklearnex tree-based models</b> are fast, but don't behave exactly
      like scikit-learn:
        <ul>
          <li>By default, sklearnex random forests and extra trees <dfn>bin
          features</dfn> (<code><dfn term="max_bins">max_bins=256</dfn></code>), which scikit-learn doesn't
          implement yet. Binning roughly doubles or triples the fit
          speed-up. Cases run with <dfn>exact splits</dfn>
          (<code>max_bins=n_samples</code>) are on the left of each
          implementation in the tree-based plots.</li>
          <li>Two <b><dfn>oneDAL</dfn></b> bugs make some fits worse:
          <a href="https://github.com/uxlfoundation/oneDAL/issues/3648">oneDAL#3648</a>
          (ExtraTreesRegressor) and
          <a href="https://github.com/uxlfoundation/oneDAL/issues/3771">oneDAL#3771</a>
          (<code><dfn>max_leaf_nodes</dfn></code>).</li>
        </ul>
      </li>
      <li><b>sklearnex KMeans</b> fits ~3x faster and is never slower than
      scikit-learn.</li>
      <li><b><dfn>Array API</dfn></b> backends are mixed. Only <b>LogisticRegression</b> and
      <b>Ridge</b> are benchmarked for now:
        <ul>
          <li>LogisticRegression on GPU is fairly fast.</li>
          <li>Ridge uses the <dfn>SVD</dfn> solver instead of <dfn>Cholesky</dfn> under Array API,
          which makes it slower
          (<a href="https://github.com/scikit-learn/scikit-learn/issues/35060">scikit-learn#35060</a>).</li>
          <li><b><dfn>PyTorch</dfn></b> on CPU parallelizes every operation, including small
          vector operations. That's why LogisticRegression is slower than
          scikit-learn with NumPy.</li>
        </ul>
      </li>
      <li>cuML <b>LogisticRegression</b> always runs the quasi-Newton <dfn>solver</dfn>
      <code>qn</code> (L-BFGS or OWL-QN). The point label can still say
      <code>lbfgs</code>, <code>newton-cg</code>, or <code>newton-cholesky</code>:
      the wrapper maps every one of those names onto <code>qn</code>. On the
      cuML side, the three synthetic solver cases time the same algorithm.
      scikit-learn runs the named solver.</li>
    </ul>
  </details>
</section>"""
TREE_BINNING_NOTE = tree_binning_note("implementation") + CUML_TREE_NOTE
DATA_DIR = "per_hardware_data"


def is_alt_sklearn_build(result: MethodResult | BenchmarkRecord) -> bool:
    return (
        result.implementation.short_name == BASE_IMPLEMENTATION
        and not is_vanilla_sklearn(result.software_hash)
    )


def is_shown(result: MethodResult | BenchmarkRecord) -> bool:
    return not is_alt_sklearn_build(result) or variant_label(result) in BASELINE_ONLY_BUILDS


def is_plotted(result: MethodResult | BenchmarkRecord) -> bool:
    return not is_alt_sklearn_build(result)


def variant_label(result: MethodResult | BenchmarkRecord) -> str:
    if result.implementation.short_name == BASE_IMPLEMENTATION:
        return software_build_name(result.software_hash)
    return result.implementation.short_name


def _variant_sort_key(result: MethodResult | BenchmarkRecord) -> tuple[int, str]:
    if result.implementation.short_name == BASE_IMPLEMENTATION:
        rank = 1 if is_alt_sklearn_build(result) else 0
    elif result.implementation.library == "sklearnex":
        rank = 2
    else:  # Array API backends
        rank = 3
    return rank, variant_label(result)


def sorted_variant_labels(results: list[MethodResult]) -> list[str]:
    """Stock scikit-learn first, then its alternative builds, sklearnex and
    Array API backends."""
    return [
        variant_label(res)
        for res in sorted(
            {variant_label(res): res for res in results}.values(),
            key=_variant_sort_key,
        )
    ]


def _variant_key(result: MethodResult | BenchmarkRecord) -> str:
    return f"{result.hardware_hash}__{variant_label(result)}"


def _group_by_variant(results) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(_variant_key(result), []).append(result)
    return grouped


# Page header and index page label.
TITLE = "Software/implementations comparison"
SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
SOURCE_ENVS = GENERAL_SOURCE_ENVS


def generate(output_dir: Path) -> None:
    results = [
        res for res in read_all_results()
        if matches_source_configs(res.case, SOURCE_CONFIGS)
        and res.hardware_hash in HARDWARE_NAMES
    ]
    failed_records = [
        record for record in read_failed_records()
        if matches_source_configs(record.case, SOURCE_CONFIGS)
        and record.hardware_hash in HARDWARE_NAMES
        and is_shown(record)
    ]
    # Computed over every machine so a variant keeps its color across tabs.
    variant_colors = variant_color_map(
        sorted_variant_labels([res for res in results if is_plotted(res)])
    )
    tab_colors = {
        **variant_colors,
        **{label: BASELINE_ONLY_TAB_COLOR for label in BASELINE_ONLY_BUILDS},
    }
    results = [res for res in results if is_shown(res)]

    def env_html(_variant_key: str, result: MethodResult) -> str:
        label = variant_label(result)
        summary = summarize_software_env(
            read_env("software", result.software_hash),
            result.implementation,
            software_hash=result.software_hash,
        )
        summary["name"] = label
        summary["actions_html"] = baseline_pick_html(label, tab_colors[label])
        return SOFTWARE_TEMPLATE.render(**summary)

    outer_tabs = []
    shown_variants: dict[str, list[MethodResult]] = {}
    grouped = _group_by_variant(results)
    for hardware_hash, hardware_name in HARDWARE_NAMES.items():
        machine_results = [res for res in results if res.hardware_hash == hardware_hash]
        labels = sorted_variant_labels(machine_results)
        if len(sorted_variant_labels([res for res in machine_results if is_plotted(res)])) < 2:
            continue
        plotted = {variant_label(res) for res in machine_results if is_plotted(res)}
        variants = [
            {
                "key": f"{hardware_hash}__{label}",
                "label": label,
                "plotted": label in plotted,
                "tabColor": tab_colors[label],
            }
            for label in labels
        ]
        shown_variants |= {variant["key"]: grouped[variant["key"]] for variant in variants}
        outer_tabs.append({
            "label": hardware_name,
            "slug": _slug_id(hardware_name),
            "emptyScope": "this hardware",
            "topHtml": HARDWARE_TEMPLATE.render(
                summarize_hardware_env(read_env("hardware", hardware_hash))
            ),
            "variants": variants,
        })

    write_data_files(
        output_dir,
        DATA_DIR,
        shown_variants,
        failed_by_variant=_group_by_variant(failed_records),
        case_key=same_machine_case_key,
        env_html=env_html,
    )
    config = {
        "page": "baseline-tabs",
        "dataDir": DATA_DIR,
        "outerTabs": outer_tabs,
        "colors": variant_colors,
        "columnTitle": "Variant name",
        "treeNote": expanded_note(TREE_BINNING_NOTE),
        "splitsFromParams": False,
        "searchNote": "",
    }
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        PREPROCESSING_TOGGLE_HTML,
        CONTENT_HTML,
        *page_rows(config),
    ])

    output = output_dir / "per_hardware.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
