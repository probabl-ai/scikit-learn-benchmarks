from importlib.resources import files

from jinja2 import Template
from plotly.offline import get_plotlyjs_version


PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
TABULATOR_CSS = "https://unpkg.com/tabulator-tables@6.5.0/dist/css/tabulator.min.css"
TABULATOR_JS = "https://unpkg.com/tabulator-tables@6.5.0/dist/js/tabulator.min.js"
BASE_CSS = (
    files("sklbench.reporting.html")
    .joinpath("base.css")
    .read_text(encoding="utf-8")
)

BASE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{{ title|default("sklbench dashboard") }}</title>
  <link href="{{ tabulator_css }}" rel="stylesheet">
  <script src="{{ plotly_cdn }}"></script>
  <script src="{{ tabulator_js }}"></script>
  <style>
{{ base_css }}
  </style>
  <script>
    window.sklbenchTables = window.sklbenchTables || {};

    function sklbenchFormatDuration(value) {
      if (value === null || value === undefined || !Number.isFinite(value)) {
        return "N/A";
      }
      if (value < 10) {
        return `${value.toPrecision(2)}ms`;
      }
      if (value < 1000) {
        return `${Math.round(value)}ms`;
      }
      if (value < 10000) {
        return `${(value / 1000).toPrecision(2)}s`;
      }
      return `${Math.round(value / 1000)}s`;
    }

    function sklbenchDurationFormatter(cell) {
      return sklbenchFormatDuration(cell.getValue());
    }

    function sklbenchSpeedupFormatter(cell) {
      const value = cell.getValue();
      if (value === null || value === undefined || !Number.isFinite(value)) {
        return "N/A";
      }
      const label = `${value.toPrecision(2)}x`;
      if (value > 1.1) {
        return `<span class="speedup-positive">${label}</span>`;
      }
      if (value < 0.9) {
        return `<span class="speedup-negative">${label}</span>`;
      }
      return label;
    }

    function sklbenchLinkFormatter(cell, formatterParams) {
      const value = cell.getValue();
      if (!value) {
        return "N/A";
      }
      const field = cell.getColumn().getField();
      const rowLabel = cell.getData()[`${field}_label`];
      const label = rowLabel || formatterParams.label || "open";
      return `<a href="${value}" target="_blank" rel="noopener">${label}</a>`;
    }

    function sklbenchPrepareColumns(columns) {
      return columns.map((column) => {
        const prepared = {...column};
        if (prepared.formatterName === "duration") {
          prepared.formatter = sklbenchDurationFormatter;
        } else if (prepared.formatterName === "speedup") {
          prepared.formatter = sklbenchSpeedupFormatter;
        } else if (prepared.formatterName === "link") {
          prepared.formatter = sklbenchLinkFormatter;
          prepared.formatterParams = {label: prepared.linkLabel || "open"};
        }
        delete prepared.formatterName;
        delete prepared.linkLabel;
        return prepared;
      });
    }

    function sklbenchInitTable(tableId, rows, columns, resetButtonId, defaultHeaderFilters) {
      if (window.sklbenchTables[tableId]) {
        window.sklbenchTables[tableId].redraw(true);
        return;
      }
      const resetButton = document.getElementById(resetButtonId);
      const resetToolbar = resetButton
        ? resetButton.closest(".detailed-results-toolbar")
        : null;
      const initialSort = [
        {column: "estimator", dir: "asc"},
        {column: "dataset", dir: "asc"},
        {column: "variant", dir: "asc"},
      ];
      // Row click (in the table, or on a matching speed-up plot point via
      // sklbenchApplyPlotMatchSort) puts every row sharing the clicked
      // comparison_key on top instead of filtering everything else out, so
      // clicking a column header to sort by something else naturally
      // supersedes it.
      let matchSortKey = null;
      const matchFirstSorter = (a, b) => {
        const aMatches = a === matchSortKey ? 0 : 1;
        const bMatches = b === matchSortKey ? 0 : 1;
        return aMatches - bMatches;
      };
      const preparedColumns = sklbenchPrepareColumns(columns).map((column) => {
        if (column.field === "comparison_key") {
          return {...column, sorter: matchFirstSorter};
        }
        return column;
      });
      const table = new Tabulator(`#${tableId}`, {
        data: rows,
        columns: preparedColumns,
        layout: "fitDataStretch",
        pagination: "local",
        paginationSize: 25,
        paginationSizeSelector: [10, 25, 50, 100, true],
        initialSort,
        initialHeaderFilter: Object.entries(defaultHeaderFilters || {}).map(
          ([field, value]) => ({field, value})
        ),
      });
      if (resetToolbar && defaultHeaderFilters && Object.keys(defaultHeaderFilters).length) {
        resetToolbar.hidden = false;
      }
      const applyMatchSort = (comparisonKey) => {
        matchSortKey = comparisonKey;
        table.setSort([{column: "comparison_key", dir: "asc"}]);
        if (resetToolbar) {
          resetToolbar.hidden = false;
        }
      };
      table.on("rowClick", (event, row) => {
        if (event.target.closest("a, button")) {
          return;
        }
        const comparisonKey = row.getData().comparison_key;
        if (!comparisonKey) {
          return;
        }
        applyMatchSort(comparisonKey);
      });
      if (resetButton) {
        resetButton.addEventListener("click", () => {
          matchSortKey = null;
          table.setSort(initialSort);
          if (resetToolbar) {
            resetToolbar.hidden = true;
          }
        });
      }
      // Exposed so a click on a matching speed-up plot point
      // (sklbenchApplyPlotMatchSort) can trigger the same sort. Tabulator
      // builds asynchronously - table.setSort isn't safe to call until
      // "tableBuilt" fires, which a plot click (unlike a click on an
      // already-rendered row) can easily race.
      table.sklbenchRows = rows;
      table.sklbenchApplyMatchSort = applyMatchSort;
      table.sklbenchBuilt = new Promise((resolve) => table.on("tableBuilt", resolve));
      window.sklbenchTables[tableId] = table;
    }

    // A table only builds (sklbenchInitTable runs) the first time its
    // "Detailed results" <details> is opened - see the toggle listener next
    // to each sklbenchInitTable call. Opening it programmatically still goes
    // through that same listener, but the "toggle" event it relies on is
    // dispatched as a separate task rather than synchronously, so callers
    // await this before assuming the table exists.
    function sklbenchOpenDetails(details) {
      if (details.open) {
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        details.addEventListener("toggle", () => resolve(), {once: true});
        details.open = true;
      });
    }

    // Speed-up plot points carry the same comparison_key as their detailed-
    // results table row (see speedup_plot_html's `comparison_key` param) -
    // clicking one applies the same "bring matching rows to top" sort as
    // clicking the table row. `assemble_plots_in_grid` always places a row's
    // (or the whole grid's) "Detailed results" <details> right after its
    // plot(s) in document order, so the nearest one following the clicked
    // chart is its table - open that first in case it's still collapsed.
    async function sklbenchApplyPlotMatchSort(chartEl, comparisonKey) {
      if (!comparisonKey) {
        return;
      }
      for (const details of document.querySelectorAll("details.detailed-results")) {
        if (chartEl.compareDocumentPosition(details) & Node.DOCUMENT_POSITION_FOLLOWING) {
          await sklbenchOpenDetails(details);
          break;
        }
      }
      const table = Object.values(window.sklbenchTables).find(
        (t) => t.sklbenchRows && t.sklbenchRows.some((row) => row.comparison_key === comparisonKey)
      );
      if (table) {
        await table.sklbenchBuilt;
        table.sklbenchApplyMatchSort(comparisonKey);
      }
    }

    function sklbenchWirePlotClicks() {
      document.querySelectorAll(".plotly-graph-div").forEach((chart) => {
        if (chart.sklbenchClickWired || typeof chart.on !== "function") {
          return;
        }
        chart.sklbenchClickWired = true;
        chart.on("plotly_click", (eventData) => {
          const point = eventData.points && eventData.points[0];
          if (point) {
            sklbenchApplyPlotMatchSort(chart, point.customdata);
          }
        });
      });
    }
    document.addEventListener("DOMContentLoaded", sklbenchWirePlotClicks);

    function sklbenchRelativeTime(isoString, nowMs) {
      const seconds = (nowMs - new Date(isoString).getTime()) / 1000;
      if (seconds < 60) {
        return "just now";
      }
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) {
        return `${minutes}m ago`;
      }
      const hours = Math.floor(seconds / 3600);
      if (hours < 24) {
        return `${hours}h ago`;
      }
      const days = Math.floor(seconds / 86400);
      return `${days}d ago`;
    }

    function sklbenchFormatDateRanges() {
      const nowMs = Date.now();
      document.querySelectorAll(".date-range").forEach((el) => {
        const start = sklbenchRelativeTime(el.dataset.start, nowMs);
        const end = sklbenchRelativeTime(el.dataset.end, nowMs);
        const range = start === end ? start : `${start} to ${end}`;
        el.textContent = `${el.dataset.count} benchmark records from ${range}`;
      });
    }
    document.addEventListener("DOMContentLoaded", sklbenchFormatDateRanges);
    setInterval(sklbenchFormatDateRanges, 60000);

    function sklbenchSetPredictCollapsed(collapsed) {
      document.body.classList.toggle("predict-collapsed", collapsed);
      document.querySelectorAll(".predict-collapse-toggle").forEach((button) => {
        button.setAttribute("aria-expanded", String(!collapsed));
      });
      document.querySelectorAll(".plotly-graph-div").forEach((chart) => {
        if (chart.offsetParent !== null) {
          Plotly.Plots.resize(chart);
        }
      });
    }
    document.addEventListener("click", (event) => {
      const button = event.target.closest(".predict-collapse-toggle");
      if (!button) {
        return;
      }
      sklbenchSetPredictCollapsed(!document.body.classList.contains("predict-collapsed"));
    });

    document.addEventListener("change", (event) => {
      if (!event.target.classList.contains("preprocessing-toggle-checkbox")) {
        return;
      }
      document.body.classList.toggle("with-preprocessing", event.target.checked);
      if (event.target.checked) {
        sklbenchSetPredictCollapsed(true);
      }
      document.querySelectorAll(".plotly-graph-div").forEach((chart) => {
        if (chart.offsetParent !== null) {
          Plotly.Plots.resize(chart);
        }
      });
    });
  </script>
</head>
<body>
  <h1>{{ title|default("sklbench dashboard") }}</h1>
  {% for row in rows %}
  <div class="page-row">{{ row }}</div>
  {% endfor %}
</body>
</html>
""")
BASE_TEMPLATE.globals["plotly_cdn"] = PLOTLY_CDN
BASE_TEMPLATE.globals["tabulator_css"] = TABULATOR_CSS
BASE_TEMPLATE.globals["tabulator_js"] = TABULATOR_JS
BASE_TEMPLATE.globals["base_css"] = BASE_CSS

DATE_RANGE_TEMPLATE = Template("""<section class="panel">
  {% if empty %}
  <h2>No results</h2>
  {% else %}
  <h2 class="date-range" data-count="{{ count }}" data-start="{{ start }}" data-end="{{ end }}"></h2>
  {% endif %}
</section>""")

HARDWARE_TEMPLATE = Template("""<section class="panel">
  <h2>Hardware</h2>
  <div class="summary-grid">
    <div>
      <h3>CPU</h3>
      <p>{{ cpu_name }}</p>
      <p class="muted">{{ architecture }}, {{ physical_cores }} physical cores, {{ logical_cpus }} logical CPUs</p>
      <p class="muted">{{ ram_gb }} GB RAM</p>
    </div>
    <div>
      <h3>GPU(s)</h3>
      {% if gpus %}
      <ul class="compact">
      {% for gpu in gpus %}
        <li><code>{{ gpu.id }}</code>: {{ gpu.name }} <span class="muted">({{ gpu.memory_gb }} GB)</span></li>
      {% endfor %}
      </ul>
      {% else %}
      <p class="muted">No GPU detected.</p>
      {% endif %}
    </div>
  </div>
</section>""")

SOFTWARE_TEMPLATE = Template("""<section class="software-details">
  <div style="min-width: 150px">
    <h3>{{ name }}</h3>
    <p> Python {{ python_version }}</p>
    {% if array_api_docs_url %}
    <p><a href="{{ array_api_docs_url }}">Array API</a> active</p>
    {% endif %}
  </div>
  <div>
    <h3>Packages</h3>
    <ul class="package-list">
    {% for package in packages %}
      <li><code>{{ package.name }}</code> {{ package.version }}{% if package.kind %} <span class="muted">({{ package.kind }})</span>{% endif %}</li>
    {% endfor %}
    </ul>
  </div>
  <div>
    <h3>Threadpools</h3>
    <ul class="compact">
    {% for threadpool in threadpools %}
      <li>{{ threadpool }}</li>
    {% endfor %}
    </ul>
  </div>
  {% if openmp %}
  <div>
    <h3>OpenMP</h3>
    <ul class="compact">
    {% for line in openmp %}
      <li>{{ line }}</li>
    {% endfor %}
    </ul>
  </div>
  {% endif %}
  <div>
    <h3>Full environment</h3>
    <p><a href="{{ software_env_json_url }}">view pixi env JSON</a></p>
  </div>
</section>""")

SOFTWARE_TABS_TEMPLATE = Template("""<section class="tabs" id="{{ tabs_id }}">
  <div class="tab-buttons">
  {% for button in buttons %}
    <button class="tab-button{% if button.active %} active{% endif %}" type="button" data-tab-target="{{ button.marker }}"{% if button.style %} style="{{ button.style }}"{% endif %}>{{ button.label|e }}</button>
  {% endfor %}
  </div>
  {% for panel in panels %}
  <div id="{{ panel.marker }}" class="tab-panel{% if panel.active %} active{% endif %}">{{ panel.html }}</div>
  {% endfor %}
  <script>
    document.querySelectorAll("#{{ tabs_id }} .tab-button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("#{{ tabs_id }} .tab-button").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll("#{{ tabs_id }} .tab-panel").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.tabTarget).classList.add("active");
      });
    });
  </script>
</section>""")

HARDWARE_TABS_TEMPLATE = Template("""<section class="tabs" id="{{ tabs_id }}">
  <div class="tab-buttons">
  {% for button in buttons %}
    <button class="tab-button{% if button.active %} active{% endif %}" type="button" data-tab-target="{{ button.marker }}">{{ button.label|e }}</button>
  {% endfor %}
  </div>
  {% for panel in panels %}
  <div id="{{ panel.marker }}" class="tab-panel{% if panel.active %} active{% endif %}">{{ panel.html }}</div>
  {% endfor %}
  <script>
    const hardwareTabButtons = document.querySelectorAll("#{{ tabs_id }} > .tab-buttons .tab-button");
    function activateHardwareTab(button, updateHash = false) {
      document.querySelectorAll("#{{ tabs_id }} > .tab-buttons .tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll("#{{ tabs_id }} > .tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const panel = document.getElementById(button.dataset.tabTarget);
      panel.classList.add("active");
      panel.querySelectorAll(".plotly-graph-div").forEach((chart) => Plotly.Plots.resize(chart));
      if (updateHash) {
        history.replaceState(null, "", `#${button.dataset.tabTarget}`);
      }
    }
    hardwareTabButtons.forEach((button) => {
      button.addEventListener("click", () => {
        activateHardwareTab(button, true);
      });
    });
    const hardwareTabFromHash = Array.from(hardwareTabButtons).find(
      (button) => button.dataset.tabTarget === window.location.hash.slice(1)
    );
    if (hardwareTabFromHash) {
      activateHardwareTab(hardwareTabFromHash);
    }
  </script>
</section>""")

PLOT_NOTES_TEMPLATE = Template("""<div class="plot-notes">
  <div class="plot-note">
    <span class="plot-marker plot-marker-circle">●</span>
    <span>Metrics and benchmark setup match the baseline</span>
  </div>
  {% if warnings %}
  <div class="plot-note">
    <span class="plot-marker plot-marker-square">■</span>
    <span>
      Metrics match the baseline, but some comparison details are worth reporting:
      <ul>
      {% for warning in warnings %}
        <li class="warning-note">
          <span class="warning-icon">{{ warning.icon }}</span>
          <span>{{ warning.message|e }} - {{ warning.estimator_counts }}</span>
        </li>
      {% endfor %}
      </ul>
    </span>
  </div>
  {% endif %}
  {% if metric_mismatch_count %}
  <div class="plot-note">
    <span class="plot-marker plot-marker-diamond">◇</span>
    <span>Metrics differ from the baseline ({{ metric_mismatch_label }})</span>
  </div>
  {% endif %}
  {% if fallback_count %}
  <div class="plot-note">
    <span class="plot-marker plot-marker-circle" style="color: #9e9e9e;">●</span>
    <span>Ran via sklearnex but fell back to stock scikit-learn, no oneDAL acceleration ({{ fallback_label }})</span>
  </div>
  {% endif %}
  {% if failed_count %}
  <div class="plot-note">
    <span class="plot-marker plot-marker-x">✕</span>
    <span>Benchmark run failed, shown at the bottom of its column ({{ failed_label }})</span>
  </div>
  {% endif %}
</div>""")
