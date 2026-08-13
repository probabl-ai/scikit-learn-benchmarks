from .blocks import (
    assemble_plots_in_grid,
    render_hardware_tabs,
    render_software_hardware_tabs,
    render_software_tabs,
)
from .plot import (
    PLOTLY_DEFAULT_COLORS,
    format_duration_ms,
    phase_breakdown_plot_html,
    phase_variant_speedup_plot_html,
    scaling_line_plot_html,
    speedup_plot_html,
    variant_color_map,
)
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
    "PLOTLY_DEFAULT_COLORS",
    "assemble_plots_in_grid",
    "format_duration_ms",
    "phase_breakdown_plot_html",
    "phase_variant_speedup_plot_html",
    "render_hardware_tabs",
    "render_software_hardware_tabs",
    "render_software_tabs",
    "scaling_line_plot_html",
    "speedup_plot_html",
    "detailed_results_table_html",
    "variant_color_map",
]
