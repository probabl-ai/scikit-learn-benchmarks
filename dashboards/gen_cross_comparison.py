"""The time of a single fit of any software on any machine, against either one software
on every machine, or every software on one machine, which also covers the software,
builds and hardware comparisons. The number of combinations grows much faster than
the results, so the page builds the one on screen from one data file per variant.
"""
from dataclasses import dataclass, replace
from pathlib import Path

from dashboards import GENERAL_SOURCE_CONFIGS, GENERAL_SOURCE_ENVS, GPU_NAMES, HARDWARE_COLORS, HARDWARE_NAMES
from sklbench.reporting.comparison import (
    CONTENT_HTML,
    CUML_TREE_NOTE,
    PREPROCESSING_TOGGLE_HTML,
    THREAD_CAPS,
    expanded_note,
    page_rows,
    tree_binning_note,
    write_data_files,
)
from sklbench.reporting.envs import (
    read_env,
    software_build_name,
    summarize_hardware_env,
    summarize_software_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    variant_color_map,
)
from sklbench.reporting.matching import (
    BenchmarkRecord,
    MethodResult,
    matches_source_configs,
    read_all_results,
    read_failed_records,
)
from sklbench.reporting.utils import stable_json, without_keys


DEFAULT_BASELINE_SOFTWARE = "sklearn-pypi"
# In the order of the checkboxes in `COMPARED_TO_HTML`.
DEFAULT_SOFTWARE_GROUPS = ["array-api", "other-packages"]
BUILDS_SOFTWARE_GROUPS = ["non-pypi"]


def _use_case_button(software_groups: list[str], across_machines: bool = False) -> str:
    across = " data-across-machines" if across_machines else ""
    return (
        f'<button type="button" class="use-case-pick"{across} '
        f'data-software-groups="{",".join(software_groups)}">Show this comparison</button>'
    )


ABOUT_HTML = f"""<section class="panel">
  <p>This dashboard compares the time of a single <dfn>fit</dfn> of any software on any
  machine with other softwares, on the same machine or another one. There are four main
  ways to use it, each explained below with what came out of it.</p>
  <details class="about-section">
    <summary>Software comparison</summary>
    <p>Compares <dfn>implementations</dfn> on the same machine: stock scikit-learn,
    scikit-learn-intelex, cuML and <dfn>Array API</dfn> backends (<dfn>PyTorch</dfn>,
    <dfn>dpnp</dfn>, <dfn>CuPy</dfn>). Since the hardware is fixed, a <dfn>speed-up</dfn>
    comes from the software: faster algorithms, better use of the CPU
    (<dfn>vectorization</dfn>, <dfn>threading</dfn>), or offloading to the GPU. This is
    what the page shows by default.</p>
    {_use_case_button(DEFAULT_SOFTWARE_GROUPS)}
    <details class="about-section">
      <summary>How to read</summary>
      <p>Pick a machine tab. The top of the tab describes the machine and the
      <dfn>software environment</dfn> of the <dfn>baseline</dfn> (versions, <dfn>BLAS</dfn>
      and <dfn>OpenMP</dfn> libraries), and the tabs under "Compared to" describe the other
      softwares. The baseline defaults to stock scikit-learn from <dfn>PyPI</dfn>, and its
      selector changes it.</p>
      <p>There's one plot per estimator category (linear, tree-based, clustering),
      each comparing <code><dfn>fit</dfn></code> times. In each plot:</p>
      <ul>
        <li>the x-axis lists the estimators, with one color per software. A software
        keeps its color on every machine and for every baseline;</li>
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
            software in the tree-based plots.</li>
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
  </details>
  <details class="about-section">
    <summary>Builds comparison</summary>
    <p>Compares <dfn>builds</dfn> of plain scikit-learn with the <dfn>PyPI wheel</dfn>. The
    code is the same, only the <dfn>BLAS/OpenMP runtime</dfn> changes. The
    <dfn>conda-forge</dfn> build names encode two choices:</p>
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
    {_use_case_button(BUILDS_SOFTWARE_GROUPS)}
    <details class="about-section">
      <summary>How to read</summary>
      <p>The plots read like in the software comparison above, with one color per build.</p>
      <p>The "Threadpools" and "OpenMP" lines of each environment tab show what
      was loaded at run time. <code>sklearn-pypi</code> lists two <code>libscipy_openblas</code>:
      the NumPy and SciPy wheels each ship their own copy of OpenBLAS, and both get
      loaded, often in different versions. NumPy's matrix products run on NumPy's
      copy, SciPy's linear algebra (used by some solvers) on SciPy's. On
      conda-forge, NumPy and SciPy share a single OpenBLAS.</p>
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
  </details>
  <details class="about-section">
    <summary>Hardware comparison</summary>
    <p>Compares machines running the same software. Since the software is fixed, a
    <dfn>speed-up</dfn> comes from the hardware. For many fits at once, see the
    <a href="hardware_parallel_fits.html">parallelized fits dashboard</a>.</p>
    {_use_case_button(DEFAULT_SOFTWARE_GROUPS, across_machines=True)}
    <details class="about-section">
      <summary>How to read</summary>
      <p>The plots read like in the software comparison above, with one color per
      machine. The CPU and GPU runs of a library are separate softwares, like
      <code>sklearn-torch-cpu</code> and <code>sklearn-torch-gpu</code>.</p>
      <p><b>Random forests</b> and <b>extra trees</b> often have a different forest
      size on each machine: their times are rescaled to the same size, and their
      metrics aren't compared when the sizes differ. <b>KMeans</b> is capped at 64
      threads on machines with more than 64 CPUs, to avoid an
      <a href="https://github.com/OpenMathLib/OpenBLAS/issues/5958">OpenBLAS crash</a>.</p>
    </details>
    <details class="about-section">
      <summary>Findings</summary>
      <p>On the benchmarked cases:</p>
      <ul>
        <li>The high-end Intel Xeon server mostly helps <b>random forests</b> and <b>extra
        trees</b>. With <b>scikit-learn</b>, their fit is ~2x to 4x faster than on the
        Intel Core Ultra laptop.</li>
        <li><b>scikit-learn-intelex</b> makes much better use of the server's 172
        cores. With it, <b>random forests</b> and <b>extra trees</b> fit ~10x to 12x
        faster on the server than on the laptop.</li>
        <li>For <b>linear models</b>, the server is often slower than the
        laptop. A single fit doesn't use many cores well, so the laptop's
        faster cores probably win.</li>
        <li><b>HistGradientBoosting</b> results are hard to interpret, because
        of known scalability issues on laptops with <b><dfn>conda-forge</dfn></b> builds and on
        machines with many cores. See
        <a href="https://github.com/scikit-learn/scikit-learn/issues/34764">scikit-learn#34764</a>
        and the <a href="hgb_scaling.html">HistGradientBoosting
        thread-scalability breakdown</a>.</li>
        <li>On the <b><dfn>PyPI</dfn></b> build, the Apple M4 CPU is ~1.5x to 2x slower
        than the Intel Core Ultra laptop.</li>
      </ul>
    </details>
  </details>
  <details class="about-section">
    <summary>Cross comparison</summary>
    <p>Changes software and hardware at once, to answer questions like "is cuML on this
    GPU faster than scikit-learn-intelex on my server?". A <dfn>speed-up</dfn> here mixes
    both effects. To separate them, see the software and hardware comparisons above. Pick
    what to compare in "Advanced comparison settings", above the plots.</p>
    <details class="about-section">
      <summary>How to read</summary>
      <p>The plots read like in the hardware comparison above, with the same notes on
      forest sizes and <b>KMeans</b>.</p>
    </details>
    <details class="about-section">
      <summary>Findings</summary>
      <p>On the benchmarked cases:</p>
      <ul>
        <li>Software and hardware gains can compound. <b>Random forests</b> fit ~30x
        faster with <b>scikit-learn-intelex</b> on the Intel Xeon server than with
        <b>scikit-learn</b> on the Intel Core Ultra laptop, while the software alone
        gives ~2x to 3x and the server alone ~3x.</li>
        <li><b>cuML</b> on the NVIDIA L4 fits <b>LogisticRegression</b> ~3.5x and
        <b>RandomForestClassifier</b> ~2.5x faster than <b>scikit-learn</b> on the laptop.
        <b>RandomForestRegressor</b> and <b>Ridge</b> are often slower.</li>
        <li>A big CPU server can beat that GPU: <b>scikit-learn-intelex</b> on the
        Intel Xeon server fits <b>random forests</b> ~20x to 35x faster than <b>cuML</b>
        on the L4, and ~1.7x to 2.5x faster on the Intel Core Ultra laptop, with 256
        bins on both sides.</li>
        <li><b><dfn>Array API</dfn></b> on a GPU is mostly a <b>LogisticRegression</b> story:
        <b><dfn>PyTorch</dfn></b> on the L4 or the Arc B580 fits it ~10x to 20x faster than
        <b>scikit-learn</b> on the laptop, but <b>Ridge</b> is slower.</li>
      </ul>
    </details>
  </details>
</section>"""

TITLE = "Software and hardware comparison of a single fit"
SOURCE_CONFIGS = GENERAL_SOURCE_CONFIGS
SOURCE_ENVS = GENERAL_SOURCE_ENVS
ENTRY_PAGE = "cross_comparison.html"
# Merged into this page, but issues and other sites can link to them.
REDIRECTS = {
    "per_hardware.html": ENTRY_PAGE,
    "builds_comparison.html": f"{ENTRY_PAGE}#variants={','.join(BUILDS_SOFTWARE_GROUPS)}",
    "hardware_comparisons.html": f"{ENTRY_PAGE}#target=software-{DEFAULT_BASELINE_SOFTWARE}",
}


BASE_IMPLEMENTATION = "sklearn"
# "None"/"default" and the plain "cpu" tag all mean "ran on the CPU".
CPU_DEVICES = {None, "default", "cpu"}
DATA_DIR = "across_machines_data"


def is_gpu_result(result: MethodResult | BenchmarkRecord) -> bool:
    return result.implementation.device not in CPU_DEVICES


def software_label(result: MethodResult | BenchmarkRecord) -> str:
    """Hardware-agnostic, so that e.g. PyTorch on Intel's "xpu" and on Apple's "mps"
    are the same software, but keeps CPU and GPU apart."""
    implementation = result.implementation
    if implementation.library == BASE_IMPLEMENTATION and implementation.data_library is None:
        return software_build_name(result.software_hash)
    device = "gpu" if is_gpu_result(result) else "cpu"
    if implementation.library == BASE_IMPLEMENTATION:
        return f"{implementation.library}-{implementation.data_library}-{device}"
    return f"{implementation.library}-{device}"


def software_sort_key(label: str) -> tuple:
    """CPU softwares first: stock scikit-learn, sklearnex, Array API backends."""
    if label.startswith("sklearn-pypi"):
        rank = 0
    elif label.startswith("sklearn-cf-"):
        rank = 1
    elif label.startswith("sklearnex-"):
        rank = 2
    elif label.startswith("sklearn-"):
        rank = 3
    else:
        rank = 4
    return label.endswith("-gpu"), rank, label


def hardware_label(result: MethodResult | BenchmarkRecord) -> str:
    """A machine goes by its GPU's name for what ran on that GPU."""
    if is_gpu_result(result) and result.hardware_hash in GPU_NAMES:
        return GPU_NAMES[result.hardware_hash]
    return HARDWARE_NAMES[result.hardware_hash]


HARDWARE_ORDER = [*HARDWARE_NAMES.values(), *GPU_NAMES.values()]


def hardware_colors() -> dict[str, str]:
    """A machine has one color, under its CPU and GPU names."""
    colors = {HARDWARE_NAMES[hardware_hash]: color for hardware_hash, color in HARDWARE_COLORS.items()}
    colors |= {GPU_NAMES[hardware_hash]: HARDWARE_COLORS[hardware_hash] for hardware_hash in GPU_NAMES}
    return colors


# `n_jobs` and RF/ET's `n_estimators` are both derived in `_real_datasets.py`
# from `N_JOBS = floor(0.9 * cpu_count(...))` - the *local* machine's core
# count at config-generation time - so they legitimately differ between two
# machines' runs of what's otherwise the identical case.
# `source_config` is excluded for the same reason as in
# `MethodResult.minimal_match_key`.
_MATCH_EXCLUDED_NAMES = {
    "implementation", "max_bins", "n_jobs", "n_estimators", "source_config",
}

# Same normalization/target as gen_models_scalability.py's
# `NORMALIZED_N_ESTIMATORS`: RF/ET fit and predict time both scale ~linearly
# with forest size, which differs between machines (see above).
NORMALIZED_N_ESTIMATORS = 100


def _n_estimators(result: MethodResult) -> int | None:
    return result.case.get("algorithm", {}).get("estimator_params", {}).get(
        "n_estimators"
    )


def _normalize_tree_result(result: MethodResult) -> MethodResult:
    n_estimators = _n_estimators(result)
    # Preprocessing doesn't depend on the forest size.
    if not n_estimators or result.method == "preprocessing":
        return result
    scale = NORMALIZED_N_ESTIMATORS / n_estimators
    return replace(result, times=[t * scale for t in result.times])


def _cross_machine_case(result: MethodResult | BenchmarkRecord) -> dict:
    case = without_keys(result.case, excluded_names=_MATCH_EXCLUDED_NAMES | THREAD_CAPS)
    if not case.get("env"):
        case.pop("env", None)
    return case


def match_key(result: MethodResult) -> str:
    return stable_json({**_cross_machine_case(result), "method": result.method})


def comparison_key(result: MethodResult | BenchmarkRecord) -> str:
    return stable_json(_cross_machine_case(result))


def _latest_per_case(results: list[MethodResult]) -> list[MethodResult]:
    """Old runs under a previous env can differ from the new ones by the
    `n_jobs`/`n_estimators` that `match_key` ignores, which would plot a stale
    point next to the fresh one."""
    latest: dict[tuple, MethodResult] = {}
    for result in results:
        key = (
            result.hardware_hash,
            software_label(result),
            match_key(result),
            result.case.get("algorithm", {}).get("estimator_params", {}).get("max_bins"),
        )
        if key not in latest or result.timestamp_recorded > latest[key].timestamp_recorded:
            latest[key] = result
    return list(latest.values())


def read_results() -> tuple[list[MethodResult], list[BenchmarkRecord]]:
    """Results and failed records of the named machines, comparable across
    machines. A failure already superseded by a successful run is dropped."""
    def is_kept(result: MethodResult | BenchmarkRecord) -> bool:
        return (
            result.hardware_hash in HARDWARE_NAMES
            and matches_source_configs(result.case, SOURCE_CONFIGS)
        )

    results = [
        _normalize_tree_result(result)
        for result in _latest_per_case(
            [result for result in read_all_results() if is_kept(result)]
        )
    ]
    latest_success = {}
    for result in results:
        key = (result.hardware_hash, software_label(result), comparison_key(result))
        if key not in latest_success or result.timestamp_recorded > latest_success[key]:
            latest_success[key] = result.timestamp_recorded
    failed_records = []
    for record in read_failed_records():
        if not is_kept(record):
            continue
        key = (record.hardware_hash, software_label(record), comparison_key(record))
        if key not in latest_success or record.timestamp_recorded > latest_success[key]:
            failed_records.append(record)
    return results, failed_records


@dataclass(frozen=True)
class Variant:
    software: str
    hardware_hash: str

    @property
    def key(self) -> str:
        return f"{self.software}__{self.hardware_hash}"


def variant_of(result: MethodResult | BenchmarkRecord) -> Variant:
    return Variant(software_label(result), result.hardware_hash)


def group_by_variant(results, variant=variant_of) -> dict[Variant, list]:
    grouped: dict[Variant, list] = {}
    for result in results:
        grouped.setdefault(variant(result), []).append(result)
    return grouped


def _env_html(_variant_key: str, result: MethodResult) -> str:
    summary = summarize_software_env(
        read_env("software", result.software_hash),
        result.implementation,
        software_hash=result.software_hash,
    )
    summary["name"] = software_label(result)
    return SOFTWARE_TEMPLATE.render(**summary)


def hardware_html(result: MethodResult) -> str:
    return HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", result.hardware_hash)))


def write_variant_data(output_dir: Path) -> dict[Variant, list[MethodResult]]:
    """Returns the results per variant."""
    results, failed_records = read_results()
    results_by_variant = group_by_variant(results)
    with_keys = lambda grouped: {variant.key: items for variant, items in grouped.items()}
    write_data_files(
        output_dir,
        DATA_DIR,
        with_keys(results_by_variant),
        failed_by_variant=with_keys(group_by_variant(failed_records)),
        case_key=comparison_key,
        env_html=_env_html,
        hardware_html=hardware_html,
    )
    return results_by_variant


def _software_group(software: str) -> str | None:
    """What "every software on one machine" can leave out. Stock scikit-learn has no
    group: it's always compared."""
    if software == DEFAULT_BASELINE_SOFTWARE:
        return None
    if software.startswith("sklearn-cf-"):
        return "non-pypi"
    return "array-api" if software.startswith("sklearn-") else "other-packages"


def _comparable_targets(results_by_variant, softwares, hardware_hashes) -> dict[str, list[str]]:
    """Baseline key -> keys of the targets it shares at least one case with."""
    match_keys = {
        variant: {
            match_key(result)
            for result in results
            if result.method == "fit"
        }
        for variant, results in results_by_variant.items()
    }
    targets = [
        *((f"software-{software}", lambda v, s=software: v.software == s) for software in softwares),
        *((f"hardware-{h}", lambda v, h=h: v.hardware_hash == h) for h in hardware_hashes),
    ]
    pages = {}
    for software in softwares:
        for hardware_hash in hardware_hashes:
            baseline = Variant(software, hardware_hash)
            if baseline not in match_keys:
                continue
            comparable = [
                key
                for key, includes in targets
                if any(
                    includes(variant) and variant != baseline and match_keys[baseline] & keys
                    for variant, keys in match_keys.items()
                )
            ]
            if comparable:
                pages[baseline.key] = comparable
    return pages


BASELINE_HTML = """<h2 class="cross-heading">Baseline</h2>
<section class="panel">
  <div class="cross-picker">
    <select id="cross-software" aria-label="Baseline software"></select>
  </div>
  <div id="cross-baseline-env" class="cross-env"></div>
</section>"""

COMPARED_TO_HTML = """<section class="cross-compared-to">
  <h2 class="cross-heading">Compared to:</h2>
  <details class="cross-advanced" id="cross-advanced">
    <summary>Advanced comparison settings</summary>
    <div class="panel cross-targets">
      <details class="about-section cross-how-to">
        <summary>How to use</summary>
        <p>Compare the baseline with either one software on every machine (one color
        per machine), or every software on one machine (one color per software). In
        the second case, the checkboxes pick which softwares to include. Stock
        scikit-learn is always included.</p>
      </details>
      <label class="cross-picker">
        <input type="radio" name="cross-target-kind" id="cross-target-software-radio">
        <select id="cross-target-software" aria-label="Compared software"></select>
        <span id="cross-target-software-suffix"></span>
      </label>
      <label class="cross-picker">
        <input type="radio" name="cross-target-kind" id="cross-target-hardware-radio">
        <span>All softwares on</span>
        <select id="cross-target-hardware" aria-label="Compared machine"></select>
      </label>
      <fieldset class="cross-variants">
        <legend>Software variants to include</legend>
        <div class="cross-variants-boxes">
          <label><input type="checkbox" data-group="array-api"> <dfn>Array API</dfn> scikit-learn</label>
          <label><input type="checkbox" data-group="other-packages"> Other packages</label>
          <label><input type="checkbox" data-group="non-pypi"> non-<dfn>PyPI</dfn> builds</label>
        </div>
      </fieldset>
    </div>
  </details>
</section>"""

GRIDS_HTML = '<div id="comparison-grids"></div>'

# The panel holds one page: changing machine only changes what the scripts fill in.
MACHINE_TABS_HTML = f"""<section class="tabs">
  <div class="tab-buttons" id="cross-machine-tabs"></div>
  <div class="tab-panel active">
    <div class="page-row" id="cross-machine-hardware"></div>
    <div class="page-row">{BASELINE_HTML}</div>
    <div class="page-row">{COMPARED_TO_HTML}</div>
    <div class="page-row">{CONTENT_HTML}</div>
    <div class="page-row">{PREPROCESSING_TOGGLE_HTML}</div>
    {GRIDS_HTML}
  </div>
</section>"""


def generate(output_dir: Path) -> None:
    results_by_variant = write_variant_data(output_dir)
    softwares = sorted({variant.software for variant in results_by_variant}, key=software_sort_key)
    hashes = {variant.hardware_hash for variant in results_by_variant}
    hardware_hashes = [hardware_hash for hardware_hash in HARDWARE_NAMES if hardware_hash in hashes]
    pages = _comparable_targets(results_by_variant, softwares, hardware_hashes)

    config = {
        "dataDir": DATA_DIR,
        "pages": pages,
        "defaultBaseline": next(
            (key for key in pages if key.startswith(f"{DEFAULT_BASELINE_SOFTWARE}__")),
            next(iter(pages)),
        ),
        "baselineMachines": [
            hardware_hash
            for hardware_hash in hardware_hashes
            if any(Variant(software, hardware_hash).key in pages for software in softwares)
        ],
        "softwareGroups": {software: _software_group(software) for software in softwares},
        "defaultSoftwareGroups": DEFAULT_SOFTWARE_GROUPS,
        "variants": {
            variant.key: {
                "software": variant.software,
                "hardwareHash": variant.hardware_hash,
                "hardwareLabel": hardware_label(variant_results[0]),
            }
            for variant, variant_results in results_by_variant.items()
        },
        "machineNames": {h: HARDWARE_NAMES[h] for h in hardware_hashes},
        "labelOrder": [*softwares, *HARDWARE_ORDER],
        "hardwareColors": hardware_colors(),
        # The palette is shorter than the softwares: the ones shown together by default
        # come first, so they get different colors.
        "softwareColors": variant_color_map(
            sorted(softwares, key=lambda software: _software_group(software) == "non-pypi")
        ),
        "treeNotes": {
            noun: expanded_note(tree_binning_note(noun) + CUML_TREE_NOTE)
            for noun in ("machine", "software")
        },
    }
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        MACHINE_TABS_HTML,
        *page_rows(config),
    ])
    output = output_dir / ENTRY_PAGE
    output.write_text(html)
    print(f"Dashboard written to {output}")
    for page, target in REDIRECTS.items():
        (output_dir / page).write_text(
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={target}">'
            f'<a href="{target}">This page has moved.</a>\n'
        )
