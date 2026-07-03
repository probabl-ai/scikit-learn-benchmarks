from .blocks import (
    assemble_plots_in_grid,
    render_hardware_tabs,
    render_software_hardware_tabs,
    render_software_tabs,
)
from .plot import speedup_plot_html, variant_color_map
from .table import detailed_results_table_html
from .templates import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
)

__all__ = [
    "BASE_TEMPLATE",
    "DATE_RANGE_TEMPLATE",
    "HARDWARE_TEMPLATE",
    "SOFTWARE_TEMPLATE",
    "assemble_plots_in_grid",
    "render_hardware_tabs",
    "render_software_hardware_tabs",
    "render_software_tabs",
    "speedup_plot_html",
    "detailed_results_table_html",
    "variant_color_map",
]
