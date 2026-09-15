from html import escape

from dashboards import HARDWARE_NAMES, dashboard_output_path
from sklbench.reporting.envs import read_env, summarize_hardware_env
from sklbench.reporting.html import BASE_TEMPLATE, HARDWARE_TEMPLATE, render_software_tabs


DASHBOARDS = [
    ("Software/implementations comparison", "per_hardware.html"),
    ("Builds comparison", "builds_comparison.html"),
    ("Hardware comparison", "hardware_comparisons.html"),
    ("Model thread-scalability", "models_scalability.html"),
    # TODO: scikit-learn versions comparison (start when? => at least 1.8; intermediate commits?)
    # longitudinal plots: to be ran once in a while
    ("HGB thread-scalability breakdown", "hgb_scaling.html"),
    ("hptuning outer-parallelism scalability", "hptuning_scalability.html"),
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
    links = "".join(
        f'<li><a href="{href}">{label}</a></li>'
        for label, href in DASHBOARDS
    )
    html = BASE_TEMPLATE.render(
        title="sklbench dashboards",
        rows=[
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

    output = dashboard_output_path("index.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
