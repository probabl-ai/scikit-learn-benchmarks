from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.output import dashboard_output_path
from reporting.html import BASE_TEMPLATE


DASHBOARDS = [
    ("CPU hardware comparison", "cpu.html"),
    ("Hardware comparison", "hardware_comparisons.html"),
    ("Per-hardware implementation comparison", "per_hardware.html"),
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
