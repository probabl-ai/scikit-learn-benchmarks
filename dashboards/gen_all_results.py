"""One table with every benchmark result, from every config, machine and
build, for looking up a raw number or checking what a recent run produced.
"""
from pathlib import Path

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import software_build_name
from sklbench.reporting.html import BASE_TEMPLATE, DATE_RANGE_TEMPLATE, detailed_results_table_html
from sklbench.reporting.matching import date_range, read_all_results, read_failed_records


TITLE = "All results"
# Reads every result, whatever config or env produced it.
SOURCE_CONFIGS = []
SOURCE_ENVS = []

ABOUT_HTML = """<section class="panel">
  <p>This table lists every benchmark result, one row per benchmarked case,
  machine, <dfn>build</dfn> and <dfn>implementation</dfn>. It is meant for
  looking up a raw number or checking what a recent run produced. The other
  dashboards plot the same results.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>Times are medians over repeats. When a case was run several times with
    the same build on the same machine, only the latest run is kept. The
    <i>Date</i> column is when the result was recorded: pick a day in its
    filter to keep only results recorded on or after it. The <i>Config</i>
    column is the benchmark config that produced the result. Click a row to
    bring the same case on other machines and builds next to it.</p>
  </details>
</section>"""


def _hardware_label(result) -> str:
    return HARDWARE_NAMES.get(result.hardware_hash, result.hardware_hash)


def _build_label(result) -> str:
    return software_build_name(result.software_hash)


def generate(output_dir: Path) -> None:
    results = read_all_results()
    failed_records = read_failed_records()
    # `results` has one entry per method (fit, predict...) of each record.
    records = list({id(result.record): result.record for result in results}.values())
    records += failed_records
    table = detailed_results_table_html(
        "all",
        {},
        baseline_label="",
        variant_label=lambda result: result.implementation.short_name,
        unmatched_candidate_results=results,
        failed_records=[
            (record, record.implementation.short_name) for record in failed_records
        ],
        collapsible=False,
        variant_column_title="Implementation",
        hardware_label=_hardware_label,
        env_label=_build_label,
        extra_columns=("date", "config"),
    )
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[
            ABOUT_HTML,
            DATE_RANGE_TEMPLATE.render(**date_range(records)),
            f'<section class="panel">{table}</section>',
        ],
    )
    output = output_dir / "all_results.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
