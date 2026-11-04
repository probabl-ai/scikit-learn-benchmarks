from .blocks import (
    render_hardware_tabs,
    render_software_hardware_tabs,
    render_software_tabs,
)
from .plot import (
    PLOT_TEMPLATE,
    SERIES_COLORS,
    case_hover_html,
    failed_hover_html,
    format_duration_ms,
    phase_breakdown_plot_html,
    phase_variant_speedup_plot_html,
    scaling_line_plot_html,
    variant_color_map,
)
from .table import (
    detailed_results_table_html,
    export_columns,
    export_failed_row,
    export_result_row,
    n_iter_median,
)
from .templates import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    SOFTWARE_TEMPLATE,
    SOURCE_CELL_TEMPLATE,
)

__all__ = [
    "BASE_TEMPLATE",
    "DATE_RANGE_TEMPLATE",
    "HARDWARE_TEMPLATE",
    "SOFTWARE_TEMPLATE",
    "SOURCE_CELL_TEMPLATE",
    "PLOT_TEMPLATE",
    "case_hover_html",
    "export_columns",
    "export_failed_row",
    "export_result_row",
    "failed_hover_html",
    "n_iter_median",
    "SERIES_COLORS",
    "format_duration_ms",
    "phase_breakdown_plot_html",
    "phase_variant_speedup_plot_html",
    "render_hardware_tabs",
    "render_software_hardware_tabs",
    "render_software_tabs",
    "scaling_line_plot_html",
    "detailed_results_table_html",
    "variant_color_map",
]
