from importlib.resources import files

from jinja2 import Template


PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
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
  <script src="{{ plotly_cdn }}"></script>
  <style>
{{ base_css }}
  </style>
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
BASE_TEMPLATE.globals["base_css"] = BASE_CSS

DATE_RANGE_TEMPLATE = Template("""<section class="panel">
  <h2>{{ label }}</h2>
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
      <li><code>{{ package.name }}</code> {{ package.version }} <span class="muted">({{ package.kind }})</span></li>
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
  <div>
    <h3>Full environment</h3>
    <p><a href="{{ software_env_json_url }}">download pixi env JSON</a></p>
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
      panel.querySelectorAll(".chart").forEach((chart) => Plotly.Plots.resize(chart));
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
</div>""")
