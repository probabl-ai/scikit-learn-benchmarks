// Software, builds, hardware and cross comparison pages: loads the data files of the
// variants on screen (sklbench/reporting/comparison.py), then draws what
// speedup_core.js computes from them. Rendering every combination up front would
// take tens of MB per page.
(function () {
  const config = JSON.parse(document.getElementById("comparison-config").textContent);
  const core = window.sklbenchSpeedupCore(config);
  const {escapeHtml, sortVariants, sum, unique} = core;
  const toggle = document.querySelector(".preprocessing-toggle-checkbox");
  const content = document.getElementById("comparison-content");

  let nextId = 0;
  let renderToken = 0;
  let view = null;

  // Data files are scripts so that the page also works from file://.
  window.sklbenchComparisonData = {};
  const loading = {};
  function loadVariant(key) {
    if (!loading[key]) {
      loading[key] = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = `${config.dataDir}/${encodeURIComponent(key)}.js`;
        script.onload = () => resolve(window.sklbenchComparisonData[key]);
        script.onerror = () => reject(new Error(`Cannot load ${script.src}`));
        document.head.appendChild(script);
      });
    }
    return loading[key];
  }

  function tableHtml(table, pending) {
    if (!table) {
      return "";
    }
    const id = `comparison-table-${(nextId += 1)}`;
    pending.tables.push({id, table});
    return `<section class="plot-detail-row"><details class="detailed-results"><summary>Detailed results</summary><div id="${id}" class="detailed-results-table"></div></details></section>`;
  }

  // --- Grid of plots. ---

  function plotCellHtml(title, method, plot, pending) {
    const toggleHtml = method === "predict"
      ? '<button type="button" class="predict-collapse-toggle" aria-expanded="true" aria-label="Collapse predict plots" title="Collapse predict plots">&rsaquo;</button>'
      : "";
    let body = '<div class="empty">No matches for this group.</div>';
    if (plot) {
      const id = `comparison-chart-${(nextId += 1)}`;
      pending.charts.push({id, plot});
      body = `<div id="${id}" class="plotly-graph-div" style="height:540px; width:100%;"></div>\n${plot.notes}`;
    }
    return `<section class="plot-cell" data-method="${method}"><div class="plot-cell-head"><h3 class="plot-cell-title">${escapeHtml(title)}</h3>${toggleHtml}</div><div class="plot-cell-body">${body}</div></section>`;
  }

  function gridHtml(grid, pending) {
    if (grid.emptyMessage) {
      return `<section class="empty">${escapeHtml(grid.emptyMessage)}</section>`;
    }
    const cells = grid.rows.flatMap((row) => {
      const hasPlot = row.cells.some((cell) => !cell.empty);
      return [
        hasPlot && row.category === "tree-based" && view.treeNote ? `<p class="plot-row-note">${view.treeNote}</p>` : "",
        ...row.cells.map((cell) =>
          cell.empty
            ? `<section class="plot-cell empty" data-method="${cell.method}">No data</section>`
            : plotCellHtml(cell.title, cell.method, cell.plot, pending)
        ),
        tableHtml(row.table, pending),
      ];
    });
    return `<section class="plot-grid method-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); --collapsed-columns: 1fr 44px;">${cells.join("")}</section>`;
  }

  function searchHtml(search, pending) {
    if (!search) {
      return "";
    }
    const cell = plotCellHtml(search.title, "fit", search.plot, pending);
    const table = tableHtml(search.table, pending);
    return `<section class="plot-grid" style="grid-template-columns: repeat(1, minmax(0, 1fr));"><p class="plot-row-note">${config.searchNote}</p>${cell}${table}</section>`;
  }

  function drawPending(pending) {
    pending.charts.forEach(({id, plot}) => {
      Plotly.newPlot(id, plot.traces, plot.layout, {responsive: true});
    });
    pending.tables.forEach(({id, table}) => {
      document.getElementById(id).closest("details").addEventListener("toggle", (event) => {
        if (event.target.open) {
          sklbenchInitTable(id, table.rows, table.columns, {});
        }
      }, {once: true});
    });
    sklbenchWirePlotClicks();
  }

  function clearGrids() {
    const container = document.getElementById("comparison-grids");
    container.querySelectorAll(".plotly-graph-div").forEach((chart) => Plotly.purge(chart));
    container.innerHTML = "";
    return container;
  }

  function drawGrids() {
    const container = clearGrids();
    window.sklbenchTables = {};
    const pending = {charts: [], tables: []};
    const grid = core.computeGrid(view, Boolean(toggle && toggle.checked));
    container.innerHTML = `<div class="page-row">${gridHtml(grid, pending)}</div><div class="page-row">${searchHtml(core.computeSearch(view), pending)}</div>`;
    drawPending(pending);
  }

  // --- Page parts. ---

  function dateRangeHtml(allData) {
    const start = allData.map((data) => data.start).sort()[0];
    const end = allData.map((data) => data.end).sort().pop();
    const count = sum(allData.map((data) => data.count));
    const sources = new Map();
    allData.flatMap((data) => data.sources).forEach((source) => {
      const key = `${source.hardware}|${source.cell}`;
      const merged = sources.get(key);
      if (merged) {
        merged.count += source.count;
        merged.start = merged.start < source.start ? merged.start : source.start;
        merged.end = merged.end > source.end ? merged.end : source.end;
        merged.envs = unique([...merged.envs, ...source.envs]).sort();
        merged.configs = unique([...merged.configs, ...source.configs]).sort();
      } else {
        sources.set(key, {...source});
      }
    });
    const codes = (values) => values.map((value) => `<code>${escapeHtml(value)}</code>`).join(", ");
    const rows = Array.from(sources.values())
      .sort((a, b) => (a.end < b.end ? 1 : a.end > b.end ? -1 : a.hardware < b.hardware ? 1 : -1))
      .map((source) => {
        const startDay = source.start.slice(0, 10);
        const endDay = source.end.slice(0, 10);
        const dates = startDay === endDay ? startDay : `${startDay} to ${endDay}`;
        return `<tr><td>${dates}</td><td>${escapeHtml(source.hardware)}</td><td>${source.count}</td><td>${source.cell}</td><td>${codes(source.envs)}</td><td>${codes(source.configs)}</td></tr>`;
      })
      .join("");
    return `<section class="panel"><h2 class="date-range" data-count="${count}" data-start="${start}" data-end="${end}"></h2>
      <details class="about-section result-sources"><summary>Where these results come from (${sources.size} source${sources.size > 1 ? "s" : ""})</summary>
      <table class="sources-table"><thead><tr><th>Recorded</th><th>Hardware</th><th>Records</th><th>Source</th><th>Envs</th><th>Config</th></tr></thead><tbody>${rows}</tbody></table>
      </details></section>`;
  }

  // `tabs` are {label, html, color, isBaseline}.
  function innerTabsHtml(tabs) {
    const buttons = tabs
      .map((tab, index) => {
        const style = tab.color ? ` style="background: ${tab.color}1F;"` : "";
        return `<button class="tab-button${index === 0 ? " active" : ""}${tab.isBaseline ? " is-baseline" : ""}" type="button" data-tab-index="${index}" data-label="${escapeHtml(tab.label)}"${style}>${escapeHtml(tab.label)}</button>`;
      })
      .join("");
    const panels = tabs
      .map((tab, index) => `<div class="tab-panel${index === 0 ? " active" : ""}" data-tab-index="${index}">${tab.html}</div>`)
      .join("");
    return `<section class="tabs" id="comparison-inner-tabs"><div class="tab-buttons">${buttons}</div>${panels}</section>`;
  }

  function wireInnerTabs() {
    const tabs = document.getElementById("comparison-inner-tabs");
    tabs.addEventListener("click", (event) => {
      const button = event.target.closest(".tab-button");
      if (!button) {
        return;
      }
      tabs.querySelectorAll(".tab-button, .tab-panel").forEach((element) => {
        element.classList.toggle("active", element.dataset.tabIndex === button.dataset.tabIndex);
      });
    });
  }

  // --- Software and hardware pages: outer tabs with a pick-baseline button. ---

  function initBaselineTabsPage() {
    const outerTabs = config.outerTabs;
    let pickedBaseline = null;
    let outerIndex = 0;

    const fromHash = outerTabs.findIndex((tab) => tab.slug === window.location.hash.slice(1));
    if (fromHash !== -1) {
      outerIndex = fromHash;
    }

    function outerHtml() {
      const buttons = outerTabs
        .map((tab, index) => `<button class="tab-button${index === outerIndex ? " active" : ""}" type="button" data-outer-index="${index}">${escapeHtml(tab.label)}</button>`)
        .join("");
      return `<section class="tabs"><div class="tab-buttons" id="comparison-outer-tabs">${buttons}</div><div class="tab-panel active" id="comparison-outer-panel"></div></section>`;
    }

    function setBaselineButtons(baselineLabel) {
      document.querySelectorAll(".baseline-pick").forEach((button) => {
        const picked = button.dataset.baseline === baselineLabel;
        button.disabled = picked;
        button.textContent = picked ? "✓ Current baseline" : "Pick as baseline";
      });
      document.querySelectorAll("#comparison-inner-tabs > .tab-buttons .tab-button").forEach((button) => {
        button.classList.toggle("is-baseline", button.dataset.label === baselineLabel);
      });
    }

    async function showOuter() {
      const token = (renderToken += 1);
      const tab = outerTabs[outerIndex];
      const panel = document.getElementById("comparison-outer-panel");
      panel.innerHTML = '<section class="empty">Loading results...</section>';
      let dataList;
      try {
        dataList = await Promise.all(tab.variants.map((variant) => loadVariant(variant.key)));
      } catch (error) {
        panel.innerHTML = `<section class="empty">${escapeHtml(error.message)}</section>`;
        return;
      }
      if (token !== renderToken) {
        return;
      }
      const loaded = new Map(tab.variants.map((variant, index) => [variant.key, dataList[index]]));
      const hasLabel = (label) => tab.variants.some((variant) => variant.label === label);
      const baselineLabel = hasLabel(pickedBaseline) ? pickedBaseline : tab.variants[0].label;
      const tabs = tab.variants.map((variant) => ({
        label: variant.label,
        color: variant.tabColor,
        isBaseline: variant.label === baselineLabel,
        html: (variant.pickHtml || "") + (loaded.get(variant.key).hardwareHtml || "") + loaded.get(variant.key).envHtml,
      }));
      panel.innerHTML =
        `<div class="page-row">${dateRangeHtml(dataList)}</div>` +
        (tab.topHtml ? `<div class="page-row">${tab.topHtml}</div>` : "") +
        `<div class="page-row">${innerTabsHtml(tabs)}</div>` +
        '<div id="comparison-grids"></div>';
      wireInnerTabs();
      sklbenchFormatDateRanges();
      setBaselineButtons(baselineLabel);
      view = {...core.tabsView(tab, loaded, baselineLabel), loaded, tab};
      drawGrids();
    }

    content.innerHTML = outerHtml();
    document.getElementById("comparison-outer-tabs").addEventListener("click", (event) => {
      const button = event.target.closest(".tab-button");
      if (!button) {
        return;
      }
      outerIndex = Number(button.dataset.outerIndex);
      document.querySelectorAll("#comparison-outer-tabs .tab-button").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      history.replaceState(null, "", `#${outerTabs[outerIndex].slug}`);
      showOuter();
    });
    content.addEventListener("click", (event) => {
      const pick = event.target.closest(".baseline-pick");
      if (!pick || pick.disabled) {
        return;
      }
      pickedBaseline = pick.dataset.baseline;
      const {loaded, tab} = view;
      setBaselineButtons(pickedBaseline);
      view = {...core.tabsView(tab, loaded, pickedBaseline), loaded, tab};
      drawGrids();
    });
    if (toggle) {
      toggle.addEventListener("change", () => {
        if (view) {
          drawGrids();
        }
      });
    }
    showOuter();
  }

  // --- Cross page: the picked combination lives in the URL hash. ---

  function initCrossPage() {
    const softwareSelect = document.getElementById("cross-software");
    const machineSelect = document.getElementById("cross-machine");
    const baselineEnv = document.getElementById("cross-baseline-env");
    const targetPickers = {
      software: {
        select: document.getElementById("cross-target-software"),
        radio: document.getElementById("cross-target-software-radio"),
      },
      hardware: {
        select: document.getElementById("cross-target-hardware"),
        radio: document.getElementById("cross-target-hardware-radio"),
      },
    };
    const softwareSuffix = document.getElementById("cross-target-software-suffix");
    let viewKey = null;

    function readState() {
      const params = new URLSearchParams(window.location.hash.slice(1));
      let baseline = params.get("baseline");
      let target = params.get("target");
      if (!config.pages[baseline]) {
        baseline = config.defaultBaseline;
      }
      if (!config.pages[baseline].includes(target)) {
        target = baseline === config.defaultBaseline && config.pages[baseline].includes(config.defaultTarget)
          ? config.defaultTarget
          : config.pages[baseline][0];
      }
      return {baseline, target, preprocessing: params.get("preprocessing") === "1"};
    }

    function writeState(state) {
      const params = new URLSearchParams({baseline: state.baseline, target: state.target});
      if (state.preprocessing) {
        params.set("preprocessing", "1");
      }
      window.location.hash = params.toString();
    }

    function setOptions(select, options, selected) {
      select.innerHTML = options
        .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`)
        .join("");
      select.value = selected;
    }

    function ownValue(kind, baseline) {
      return kind === "software" ? baseline.software : baseline.hardwareHash;
    }

    // The picker that isn't in use shows the baseline's own value when it can.
    function updateTargetPicker(kind, state) {
      const {select, radio} = targetPickers[kind];
      const prefix = `${kind}-`;
      const values = config.pages[state.baseline]
        .filter((target) => target.startsWith(prefix))
        .map((target) => target.slice(prefix.length));
      const own = ownValue(kind, config.variants[state.baseline]);
      const active = state.target.startsWith(prefix);
      const selected = active ? state.target.slice(prefix.length) : values.includes(own) ? own : values[0];
      setOptions(
        select,
        values.map((value) => [value, kind === "software" ? value : config.machineNames[value]]),
        selected
      );
      radio.checked = active;
      radio.disabled = select.disabled = !values.length;
    }

    function updateSelectors(state) {
      const baseline = config.variants[state.baseline];
      const baselines = Object.keys(config.pages).map((key) => config.variants[key]);
      setOptions(softwareSelect, unique(baselines.map((v) => v.software)).map((s) => [s, s]), baseline.software);
      setOptions(
        machineSelect,
        baselines
          .filter((v) => v.software === baseline.software)
          .map((v) => [v.hardwareHash, config.machineNames[v.hardwareHash]]),
        baseline.hardwareHash
      );
      updateTargetPicker("software", state);
      updateTargetPicker("hardware", state);
      // Another software also runs on the baseline's machine.
      softwareSuffix.textContent =
        targetPickers.software.select.value === baseline.software ? "on all other machines" : "on all machines";
    }

    function closestTarget(baselineKey, target) {
      const targets = config.pages[baselineKey];
      if (targets.includes(target)) {
        return target;
      }
      const kind = target.slice(0, target.indexOf("-"));
      const own = `${kind}-${ownValue(kind, config.variants[baselineKey])}`;
      if (targets.includes(own)) {
        return own;
      }
      return targets.find((candidate) => candidate.startsWith(`${kind}-`)) || targets[0];
    }

    function onBaselineChange() {
      const software = softwareSelect.value;
      const keys = Object.keys(config.pages).filter((key) => config.variants[key].software === software);
      const baseline = keys.find((key) => config.variants[key].hardwareHash === machineSelect.value) || keys[0];
      writeState({
        baseline,
        target: closestTarget(baseline, readState().target),
        preprocessing: toggle.checked,
      });
    }

    function softwareHtml(data) {
      return `<section class="panel"><h2>Software</h2>${data.envHtml}</section>`;
    }

    // A tab describes what varies between tabs, and the other side only when it
    // isn't the baseline's.
    function comparedEnvHtml(baselineKey, target, key, data) {
      const baseline = config.variants[baselineKey];
      const variant = config.variants[key];
      const showHardware = target.startsWith("software-") || variant.hardwareHash !== baseline.hardwareHash;
      const showSoftware = target.startsWith("hardware-") || variant.software !== baseline.software;
      return `<div class="cross-env">${showSoftware ? softwareHtml(data) : ""}${showHardware ? data.hardwareHtml : ""}</div>`;
    }

    async function render(state) {
      const token = (renderToken += 1);
      updateSelectors(state);
      const candidates = core.crossCandidates(state.baseline, state.target);
      clearGrids();
      content.innerHTML = '<section class="empty">Loading results...</section>';
      const keys = [state.baseline, ...candidates];
      let dataList;
      try {
        dataList = await Promise.all(keys.map(loadVariant));
      } catch (error) {
        content.innerHTML = `<section class="empty">${escapeHtml(error.message)}</section>`;
        return;
      }
      if (token !== renderToken) {
        return;
      }
      const [baseData, ...candidateData] = dataList;
      viewKey = `${state.baseline}|${state.target}`;
      view = core.crossView(state.baseline, state.target, new Map(keys.map((key, index) => [key, dataList[index]])));
      const labelOfVariant = view.labelOfVariant;
      const tabs = sortVariants(candidates.map(labelOfVariant)).map((label) => {
        const index = candidates.findIndex((key) => labelOfVariant(key) === label);
        return {
          label,
          color: view.colors[label],
          html: comparedEnvHtml(state.baseline, state.target, candidates[index], candidateData[index]),
        };
      });
      baselineEnv.innerHTML = softwareHtml(baseData) + baseData.hardwareHtml;
      content.innerHTML =
        `<div class="page-row">${innerTabsHtml(tabs)}</div>` +
        dateRangeHtml([baseData, ...candidateData]);
      wireInnerTabs();
      sklbenchFormatDateRanges();
      drawGrids();
    }

    function onHashChange() {
      const state = readState();
      if (toggle.checked !== state.preprocessing) {
        toggle.checked = state.preprocessing;
        toggle.dispatchEvent(new Event("change", {bubbles: true}));
      }
      if (viewKey === `${state.baseline}|${state.target}`) {
        drawGrids();
      } else {
        render(state);
      }
    }

    [softwareSelect, machineSelect].forEach((select) => {
      select.addEventListener("change", onBaselineChange);
    });
    Object.entries(targetPickers).forEach(([kind, {select, radio}]) => {
      const pick = () => writeState({...readState(), target: `${kind}-${select.value}`});
      select.addEventListener("change", pick);
      radio.addEventListener("change", pick);
    });
    toggle.addEventListener("change", () => {
      const state = readState();
      if (state.preprocessing !== toggle.checked) {
        writeState({...state, preprocessing: toggle.checked});
      }
    });
    window.addEventListener("hashchange", onHashChange);
    onHashChange();
  }

  if (config.page === "cross") {
    initCrossPage();
  } else {
    initBaselineTabsPage();
  }
})();
