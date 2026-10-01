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
    gen_hgb_scalability_breakdown,
    gen_hptuning_scalability,
    gen_models_scalability,
    gen_softwares_comparison,
)
from sklbench.reporting.envs import read_env, summarize_hardware_env
from sklbench.reporting.html import BASE_TEMPLATE, HARDWARE_TEMPLATE, render_software_tabs


ABOUT_HTML = """<section class="panel">
  <p>
  This site tracks scikit-learn's performance across machines, builds and
  accelerated backends. It shows which workloads benefit from
  <a href="https://uxlfoundation.github.io/scikit-learn-intelex/latest/">scikit-learn-intelex</a>
  or an <a href="https://scikit-learn.org/stable/modules/array_api.html">Array API</a> backend,
  and when a different <dfn>BLAS</dfn>/<dfn>OpenMP</dfn> <dfn>build</dfn> or machine changes anything.
  Most dashboards below vary one of these (<dfn>implementation</dfn>, build or hardware)
  and keep the others fixed. The scalability ones look at how a fit speeds up
  with more <dfn>CPU cores</dfn>. Each dashboard explains how to read it in its own intro.
  </p>

  <h3 style="margin-top: 10px">Overall findings:</h3>

  <ul>
    <li>
    <p>On the benchmarked cases, <b><code>scikit-learn-intelex</code></b> on CPUs
    is the most consistently fast option among the currently benchmarked set
    of options, especially for fitting <b>tree-based
    models</b>. Other backends and builds help in narrower, workload-specific
    cases.</p>
    <p>Note that scikit-learn-intelex only accelerates a subset of estimators
    and parameters (see its
    <a href="https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html">supported algorithms</a>).
    Its behavior can also differ from scikit-learn, for instance in
    <a href="https://github.com/uxlfoundation/oneDAL/issues/3771">best-first tree growth</a>,
    <a href="https://github.com/uxlfoundation/scikit-learn-intelex/issues/3401">multiclass <code>predict_proba</code></a>
    or <a href="https://github.com/uxlfoundation/scikit-learn-intelex/issues/3356">missing values at predict time</a>.
    These gaps are actively worked on.</p>
    </li>
    <li>
    For <b>linear models</b>, a good option for improved performance is simply to pick
    the <b><dfn>MKL conda-forge build</dfn></b> of scikit-learn, instead of the <dfn>PyPI</dfn> one.
    </li>
    <li>
    Using a bigger machine doesn't help a lot for a single fit for most models
    except <b>random forests</b> and alike. But it helps a lot for multiple parallelized fits, typically
    for hyper-parameter tuning workloads.
    </li>
  </ul>
</section>"""


DASHBOARDS = [
    (module.TITLE, module, href)
    for module, href in [
        (gen_softwares_comparison, "per_hardware.html"),
        (gen_builds_comparison, "builds_comparison.html"),
        (gen_hardware_comparisons, "hardware_comparisons.html"),
        (gen_models_scalability, "models_scalability.html"),
        (gen_hptuning_scalability, "hptuning_scalability.html"),
        (gen_hgb_scalability_breakdown, "hgb_scaling.html"),
    ]
]

DASHBOARD_DESCRIPTIONS = {
    "per_hardware.html": "scikit-learn-intelex and <dfn>Array API</dfn> backends vs stock scikit-learn.",
    "builds_comparison.html": "<dfn>PyPI</dfn> vs <dfn>conda-forge</dfn> <dfn>BLAS/OpenMP builds</dfn> of scikit-learn.",
    "hardware_comparisons.html": "Same software on different machines.",
    "models_scalability.html": "How a single fit scales with more CPU cores.",
    "hptuning_scalability.html": "<dfn>Speed-up</dfn> from evaluating search <dfn>candidates</dfn> in parallel.",
    "hgb_scaling.html": "Fit time per phase as the thread count grows. This is mostly an investigation for scikit-learn developers.",
}

# Machines that are benchmarked but not presented on the index page.
INDEX_HIDDEN_HARDWARE = {"be1055", "5dcf30"}


def _hardware_overview_html() -> str:
    badges = [
        f"<h3>{escape(name)}</h3>"
        + HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash)))
        for hardware_hash, name in HARDWARE_NAMES.items()
        if hardware_hash not in INDEX_HIDDEN_HARDWARE
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
        f'<tr><td><a href="{href}">{label}</a></td>'
        f'<td>{DASHBOARD_DESCRIPTIONS[href]}</td></tr>'
        for label, _, href in DASHBOARDS
    )
    html = BASE_TEMPLATE.render(
        title="scikit-learn benchmark dashboard",
        home_url=None,
        rows=[
            ABOUT_HTML,
            f"""
            <section class="panel">
              <h2>Dashboards</h2>
              <table class="dashboard-list">
                {links}
              </table>
            </section>
            """,
            _hardware_overview_html(),
        ],
    )

    output = output_dir / "index.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
