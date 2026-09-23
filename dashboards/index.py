"""Sole entry point for the full dashboard site: generates every dashboard
listed in DASHBOARDS (in-process, by calling each module's `generate()`)
plus the index page linking to them. CI (dashboard-pages.yml,
dashboard-preview-build.yml) and watch_dashboards.py just run this one
script - see index_comparison.py for the separate PR-comparison entry
point, which reads an unrelated ephemeral results/ dir and is never part
of this loop.
"""

from html import escape

from dashboards import (
    HARDWARE_NAMES,
    dashboard_output_dir,
    gen_builds_comparison,
    gen_hardware_comparisons,
    gen_hgb_dev_scalability_breakdown,
    gen_hgb_dev_speedup_breakdown,
    gen_hgb_scalability_breakdown,
    gen_hptuning_scalability,
    gen_models_scalability,
    gen_per_hardware,
)
from sklbench.reporting.envs import read_env, summarize_hardware_env
from sklbench.reporting.html import BASE_TEMPLATE, HARDWARE_TEMPLATE, render_software_tabs


ABOUT_HTML = """<section class="panel">
  <p>This site tracks scikit-learn's performance across hardware, software
  builds, and accelerated backends, to make trade-offs visible: which
  workloads benefit from scikit-learn-intelex or an Array API backend, which
  results are directly comparable to the plain scikit-learn baseline, and
  where a different BLAS/OpenMP build or a different machine actually
  changes anything. Each dashboard below isolates one of those variables
  (implementation, build, or hardware) while holding the others fixed, or
  looks at how a single fit scales with more CPU cores &mdash; see each
  dashboard's own intro for specifics on how to read it. In short, as of the
  latest full run: <code>scikit-learn-intelex</code> on Intel CPUs is the
  most consistently fast option today, especially for tree-based model
  fitting; other backends and builds help in narrower, more workload-specific
  cases.</p>
</section>"""


DASHBOARDS = [
    ("Software/implementations comparison", gen_per_hardware, "per_hardware.html"),
    ("Builds comparison", gen_builds_comparison, "builds_comparison.html"),
    ("Hardware comparison", gen_hardware_comparisons, "hardware_comparisons.html"),
    ("Model thread-scalability", gen_models_scalability, "models_scalability.html"),
    ("HGB thread-scalability breakdown", gen_hgb_scalability_breakdown, "hgb_scaling.html"),
    ("[dev] HGB thread-scalability breakdown", gen_hgb_dev_scalability_breakdown, "hgb_dev_scaling.html"),
    ("[dev] HGB speed-up breakdown", gen_hgb_dev_speedup_breakdown, "hgb_speedup_breakdown.html"),
    ("hptuning outer-parallelism scalability", gen_hptuning_scalability, "hptuning_scalability.html"),
]


def _hardware_overview_html() -> str:
    badges = [
        f"<h3>{escape(name)}</h3>"
        + HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash)))
        for hardware_hash, name in HARDWARE_NAMES.items()
    ]
    return f"""
    <section class="panel">
      <h2>Benchmark hardware</h2>
      {render_software_tabs(badges)}
    </section>
    """


if __name__ == "__main__":
    output_dir = dashboard_output_dir()

    for _, module, _ in DASHBOARDS:
        module.generate(output_dir)

    links = "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for label, _, href in DASHBOARDS
    )
    html = BASE_TEMPLATE.render(
        title="sklbench dashboards",
        rows=[
            ABOUT_HTML,
            f"""
            <section class="panel">
              <h2>Dashboards</h2>
              <ul class="compact">
                {links}
              </ul>
            </section>
            """,
            _hardware_overview_html(),
        ],
    )

    output = output_dir / "index.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
