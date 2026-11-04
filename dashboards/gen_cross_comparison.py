"""Any software on any machine against either one software on every machine, or
every software on one machine. The number of combinations grows much faster than
the results, so the page builds the one on screen from the data files of the
hardware dashboard.
"""
from pathlib import Path

from dashboards import HARDWARE_NAMES
from dashboards import gen_hardware_comparisons as hardware
from sklbench.reporting.comparison import (
    CONTENT_HTML,
    PREPROCESSING_TOGGLE_HTML,
    expanded_note,
    page_rows,
    tree_binning_note,
)
from sklbench.reporting.html import BASE_TEMPLATE, variant_color_map


ABOUT_HTML = """<section class="panel">
  <p>This dashboard changes software and hardware at once, to answer questions
  like "is cuML on this GPU faster than scikit-learn-intelex on my server?". A
  <dfn>speed-up</dfn> here mixes both effects. To separate them, see the
  <a href="per_hardware.html">software</a> and
  <a href="hardware_comparisons.html">hardware</a> dashboards.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Pick a <dfn>baseline</dfn>: one software on one machine. Then compare it with
    either one software on every machine (one color per machine), or one machine
    with every software (one color per software). The tabs below describe the
    baseline and each compared machine and <dfn>software environment</dfn>.</p>
    <p>The plots read like in
    <a href="per_hardware.html">the software/implementations dashboard</a>: each
    point is one benchmark case, and points above the dashed 1x line are faster
    than the baseline. Marker shapes mean the same.</p>
    <p><b>Random forests</b> and <b>extra trees</b> often have a different forest
    size on each machine: their times are rescaled to the same size, and their
    metrics aren't compared when the sizes differ. The last row compares the time
    per fit in hyper-parameter searches, where many fits run in parallel.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases:</p>
    <ul>
      <li>Software and hardware gains can compound. <b>Random forests</b> fit ~30x
      faster with <b>scikit-learn-intelex</b> on the Intel Xeon server than with
      <b>scikit-learn</b> on the Intel Core Ultra laptop, while the software alone
      gives ~2x to 3x and the server alone ~3x.</li>
      <li><b>cuML</b> on the NVIDIA L4 fits <b>LogisticRegression</b> and
      <b>RandomForestClassifier</b> ~3x faster than <b>scikit-learn</b> on the laptop.
      <b>RandomForestRegressor</b> is close to parity and <b>Ridge</b> is often slower.</li>
      <li>A big CPU server can beat that GPU: <b>scikit-learn-intelex</b> on the
      Intel Xeon server fits <b>random forests</b> over 10x faster than <b>cuML</b>
      on the L4. Against the laptop, <b>scikit-learn-intelex</b> and <b>cuML</b> are
      close on most <b>random forests</b>.</li>
      <li><b><dfn>Array API</dfn></b> on a GPU is mostly a <b>LogisticRegression</b> story:
      <b><dfn>PyTorch</dfn></b> on the L4 or the Arc B580 fits it ~10x to 20x faster than
      <b>scikit-learn</b> on the laptop, but <b>Ridge</b> is slower.</li>
    </ul>
  </details>
</section>"""

TITLE = "Cross software/hardware comparison"
SOURCE_CONFIGS = hardware.SOURCE_CONFIGS
SOURCE_ENVS = hardware.SOURCE_ENVS
ENTRY_PAGE = "cross_comparison.html"
# Stock scikit-learn on the first machine against everything the GPU VM runs,
# cuML included.
DEFAULT_BASELINE_SOFTWARE = "sklearn-pypi"
DEFAULT_TARGET = "hardware-d189c4"


def _comparable_targets(results_by_variant, softwares, hardware_hashes) -> dict[str, list[str]]:
    """Baseline key -> keys of the targets it shares at least one case with."""
    match_keys = {
        variant: {
            hardware.match_key(result)
            for result in results
            if result.method in ("fit", "predict")
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
            baseline = hardware.Variant(software, hardware_hash)
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


SELECTOR_HTML = """<section class="panel">
  <div class="comparison-selector">
    <label>Baseline software<select id="cross-software"></select></label>
    <span class="comparison-selector-sep">on</span>
    <label>Baseline machine<select id="cross-machine"></select></label>
    <span class="comparison-selector-sep">vs</span>
    <label>Compare with<select id="cross-target"></select></label>
  </div>
</section>"""


def generate(output_dir: Path) -> None:
    results_by_variant = hardware.write_variant_data(output_dir)
    results = [res for variant_results in results_by_variant.values() for res in variant_results]
    softwares = sorted({variant.software for variant in results_by_variant}, key=hardware.software_sort_key)
    hashes = {variant.hardware_hash for variant in results_by_variant}
    hardware_hashes = [hardware_hash for hardware_hash in HARDWARE_NAMES if hardware_hash in hashes]
    pages = _comparable_targets(results_by_variant, softwares, hardware_hashes)

    config = {
        "page": "cross",
        "dataDir": hardware.DATA_DIR,
        "pages": pages,
        "defaultBaseline": next(
            (key for key in pages if key.startswith(f"{DEFAULT_BASELINE_SOFTWARE}__")),
            next(iter(pages)),
        ),
        "defaultTarget": DEFAULT_TARGET,
        "variants": {
            variant.key: {
                "software": variant.software,
                "hardwareHash": variant.hardware_hash,
                "hardwareLabel": hardware.hardware_label(variant_results[0]),
            }
            for variant, variant_results in results_by_variant.items()
        },
        "machineNames": {h: HARDWARE_NAMES[h] for h in hardware_hashes},
        "labelOrder": [*softwares, *hardware.HARDWARE_ORDER],
        "hardwareColors": hardware.hardware_colors(results),
        "softwareColors": variant_color_map(softwares),
        "treeNotes": {
            noun: expanded_note(tree_binning_note(noun)) for noun in ("machine", "software")
        },
        "splitsFromParams": True,
        "searchNote": expanded_note(hardware.SEARCH_NOTE),
    }
    html = BASE_TEMPLATE.render(title=TITLE, rows=[
        ABOUT_HTML,
        SELECTOR_HTML,
        PREPROCESSING_TOGGLE_HTML,
        CONTENT_HTML,
        *page_rows(config),
    ])
    output = output_dir / ENTRY_PAGE
    output.write_text(html)
    print(f"Dashboard written to {output}")
