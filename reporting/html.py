from jinja2 import Template

from .matching import Match

BASE_TEMPLATE = Template("")
# ^ should probably include plotly.js link, etc.

DATE_RANGE_TEMPLATE = Template("")
HARDWARE_TEMPLATE = Template("")
SOFTWARE_TEMPLATE = Template("")


def plotly_colored_tabs(elements: list[str]):
    pass


def assemble_plots_in_grid(plots: list[dict], *, x, y):
    pass


def speedup_plot_html(matches: list[Match]):
    pass

