from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_path
from sklbench.reporting.html import BASE_TEMPLATE


DASHBOARDS = [
    ("Software/implementations comparison", "per_hardware.html"),
    ("Builds comparison", "builds_comparison.html"),
    ("Hardware comparison", "hardware_comparisons.html"),
    ("Model thread-scalability", "models_scalability.html"),
    # TODO: scikit-learn versions comparison (start when? => at least 1.8; intermediate commits?)
    # longitudinal plots: to be ran once in a while
    ("HGB thread-scalability breakdown", "hgb_scaling.html"),
    ("[dev] HGB thread-scalability breakdown", "hgb_dev_scaling.html"),
    ("[dev] HGB speed-up breakdown", "hgb_speedup_breakdown.html"),
]


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
            """
        ],
    )

    output = dashboard_output_path("index.html")
    output.write_text(html)
    print(f"Dashboard written to {output}")
