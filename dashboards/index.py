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
    gen_per_hardware,
)
from sklbench.reporting.envs import read_env, summarize_hardware_env
from sklbench.reporting.html import BASE_TEMPLATE, HARDWARE_TEMPLATE, render_software_tabs


ABOUT_HTML = """<section class="panel">
  <p>This site tracks scikit-learn's performance across machines, builds and
  accelerated backends. It shows which workloads benefit from
  scikit-learn-intelex or an Array API backend, and when a different
  BLAS/OpenMP build or machine changes anything. Most dashboards below vary
  one of these (implementation, build or hardware) and keep the others fixed.
  The scalability ones look at how a fit speeds up with more CPU cores. Each
  dashboard explains how to read it in its own intro.</p>

  <h3 style="margin-top: 5px">Overall results:</h3>

  <ul>
    <li>
    <p>On the benchmarked cases, <code>scikit-learn-intelex</code> on CPUs
    is the most consistently fast option, especially for fitting tree-based
    models. Other backends and builds help in narrower, workload-specific
    cases.</p>
    <p>Note that scikit-learn-intelex only accelerates a subset of estimators
    and parameters (see its
    <a href="https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html">supported algorithms</a>),
    and falls back to scikit-learn otherwise. Its behavior can also differ from
    scikit-learn, for instance in
    <a href="https://github.com/uxlfoundation/oneDAL/issues/3771">best-first tree growth</a>,
    <a href="https://github.com/uxlfoundation/scikit-learn-intelex/issues/3401">multiclass <code>predict_proba</code></a>
    or <a href="https://github.com/uxlfoundation/scikit-learn-intelex/issues/3356">missing values at predict time</a>.
    These gaps are actively worked on.</p>
    </li>
    <li>
    For linear models, a good option for improved performance is simply to pick 
    the MKL conda-forge build of scikit-learn, instead of the PyPI one.
    </li>
    <li>
    Using a bigger machine doesn't help a lot for a single fit for most models
    except random forests and alike. But it helps a lot for multiple parallelized fits, typically
    for hyper-parameters tunning workloads.
    </li>
  </ul>
</section>"""


DASHBOARDS = [
    ("Software/implementations comparison", gen_per_hardware, "per_hardware.html"),
    ("Builds comparison", gen_builds_comparison, "builds_comparison.html"),
    ("Hardware comparison", gen_hardware_comparisons, "hardware_comparisons.html"),
    ("Model thread-scalability", gen_models_scalability, "models_scalability.html"),
    ("hptuning outer-parallelism scalability", gen_hptuning_scalability, "hptuning_scalability.html"),
    ("HGB thread-scalability breakdown", gen_hgb_scalability_breakdown, "hgb_scaling.html"),
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
