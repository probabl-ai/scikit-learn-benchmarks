// Fit and parallelized fits comparison pages: loads the data files of the
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

  function plotCellHtml(title, plot, pending) {
    let body = '<div class="empty">No matches for this group.</div>';
    if (plot) {
      const id = `comparison-chart-${(nextId += 1)}`;
      pending.charts.push({id, plot});
      body = `<div id="${id}" class="plotly-graph-div" style="height:540px; width:100%;"></div>\n${plot.notes}`;
    }
    return `<section class="plot-cell"><div class="plot-cell-head"><h3 class="plot-cell-title">${escapeHtml(title)}</h3></div><div class="plot-cell-body">${body}</div></section>`;
  }

  function gridHtml(grid, pending) {
    if (grid.emptyMessage) {
      return `<section class="empty">${escapeHtml(grid.emptyMessage)}</section>`;
    }
    const cells = grid.rows.flatMap((row) => {
      const hasPlot = row.cells.some((cell) => !cell.empty);
      return [
        hasPlot && row.category === "tree-based" && view.treeNote ? `<p class="plot-row-note">${view.treeNote}</p>` : "",
        ...row.cells.filter((cell) => !cell.empty).map((cell) => plotCellHtml(cell.title, cell.plot, pending)),
        tableHtml(row.table, pending),
      ];
    });
    return `<section class="plot-grid" style="grid-template-columns: repeat(1, minmax(0, 1fr));">${cells.join("")}</section>`;
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
    // The cross page can filter out every candidate.
    if (!view) {
      return;
    }
    window.sklbenchTables = {};
    const pending = {charts: [], tables: []};
    const grid = core.computeGrid(view, Boolean(toggle && toggle.checked));
    container.innerHTML = `<div class="page-row">${gridHtml(grid, pending)}</div>`;
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

  // --- Parallelized fits page: outer tabs with a pick-baseline button. ---

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
    const machineTabs = document.getElementById("cross-machine-tabs");
    const machineHardware = document.getElementById("cross-machine-hardware");
    const softwareSelect = document.getElementById("cross-software");
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
    const groupBoxes = Array.from(document.querySelectorAll(".cross-variants input"));
    const useCaseButtons = Array.from(document.querySelectorAll(".use-case-pick"));
    const defaultSoftware = config.variants[config.defaultBaseline].software;
    let viewKey = null;

    machineTabs.innerHTML = config.baselineMachines
      .map((hash) => `<button class="tab-button" type="button" data-machine="${hash}">${escapeHtml(config.machineNames[hash])}</button>`)
      .join("");

    function readState() {
      const params = new URLSearchParams(window.location.hash.slice(1));
      let baseline = params.get("baseline");
      let target = params.get("target");
      if (!config.pages[baseline]) {
        baseline = config.defaultBaseline;
      }
      if (!config.pages[baseline].includes(target)) {
        target = closestTarget(baseline, ownMachineTarget(baseline));
      }
      const softwareGroups = params.has("variants")
        ? groupBoxes.map((box) => box.dataset.group).filter((group) => params.get("variants").split(",").includes(group))
        : config.defaultSoftwareGroups;
      return {baseline, target, softwareGroups, preprocessing: params.get("preprocessing") === "1"};
    }

    function hasDefaultGroups(state) {
      return state.softwareGroups.join() === config.defaultSoftwareGroups.join();
    }

    function writeState(state) {
      const params = new URLSearchParams({baseline: state.baseline, target: state.target});
      if (!hasDefaultGroups(state)) {
        params.set("variants", state.softwareGroups.join());
      }
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

    function ownMachineTarget(baselineKey) {
      return `hardware-${config.variants[baselineKey].hardwareHash}`;
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
      machineTabs.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.machine === baseline.hardwareHash);
      });
      const softwares = Object.keys(config.pages)
        .map((key) => config.variants[key])
        .filter((variant) => variant.hardwareHash === baseline.hardwareHash)
        .map((variant) => variant.software);
      setOptions(softwareSelect, softwares.map((s) => [s, s]), baseline.software);
      updateTargetPicker("software", state);
      updateTargetPicker("hardware", state);
      // Another software also runs on the baseline's machine.
      softwareSuffix.textContent =
        targetPickers.software.select.value === baseline.software ? "on all other machines" : "on all machines";
      updateGroupBoxes(state);
      updateUseCaseButtons(state);
    }

    function machineRan(hash, softwareGroups) {
      return Object.values(config.variants).some(
        (variant) => variant.hardwareHash === hash && softwareGroups.includes(config.softwareGroups[variant.software])
      );
    }

    // What the button of a use case switches to. Not every machine ran every software:
    // the builds, for one, only ran on a few.
    function useCaseState(button, state) {
      const softwareGroups = button.dataset.softwareGroups.split(",");
      const acrossMachines = "acrossMachines" in button.dataset;
      const current = config.variants[state.baseline].hardwareHash;
      const machine = acrossMachines || machineRan(current, softwareGroups)
        ? current
        : config.baselineMachines.find((hash) => machineRan(hash, softwareGroups));
      const baseline = baselineOn(machine, defaultSoftware);
      const target = acrossMachines ? `software-${defaultSoftware}` : ownMachineTarget(baseline);
      return {...state, baseline, target: closestTarget(baseline, target), softwareGroups};
    }

    function updateUseCaseButtons(state) {
      useCaseButtons.forEach((button) => {
        const shown = viewKeyOf(useCaseState(button, state)) === viewKeyOf(state);
        button.disabled = shown;
        button.textContent = shown ? "✓ Currently shown" : "Show this comparison";
      });
    }

    // Greyed out when the box wouldn't change anything on screen.
    function updateGroupBoxes(state) {
      const onMachine = targetPickers.hardware.radio.checked;
      const machine = targetPickers.hardware.select.value;
      groupBoxes.forEach((box) => {
        const group = box.dataset.group;
        box.checked = state.softwareGroups.includes(group);
        box.disabled = !onMachine || !Object.keys(config.variants).some((key) => {
          const variant = config.variants[key];
          return key !== state.baseline && variant.hardwareHash === machine && config.softwareGroups[variant.software] === group;
        });
      });
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

    function baselineOn(hash, software) {
      const keys = Object.keys(config.pages).filter((key) => config.variants[key].hardwareHash === hash);
      const withSoftware = (name) => keys.find((key) => config.variants[key].software === name);
      return withSoftware(software) || withSoftware(defaultSoftware) || keys[0];
    }

    function onSoftwareChange() {
      const state = readState();
      const baseline = baselineOn(config.variants[state.baseline].hardwareHash, softwareSelect.value);
      writeState({...state, baseline, target: closestTarget(baseline, state.target)});
    }

    // The default comparison, with the softwares of the baseline's machine, follows the tab.
    function onMachineTab(hash) {
      const state = readState();
      const baseline = baselineOn(hash, config.variants[state.baseline].software);
      const target = state.target === ownMachineTarget(state.baseline) ? ownMachineTarget(baseline) : state.target;
      writeState({...state, baseline, target: closestTarget(baseline, target)});
    }

    function softwareHtml(data) {
      return `<section class="panel"><h2>Software</h2>${data.envHtml}</section>`;
    }

    // Machine tabs keep the software: its threadpools and versions depend on the machine.
    function comparedEnvHtml(target, data) {
      return target.startsWith("software-")
        ? `<div class="cross-env">${data.hardwareHtml}${softwareHtml(data)}</div>`
        : softwareHtml(data);
    }

    // All the tabs of a machine share its hardware.
    function sharedHardwareHtml(state, candidateData) {
      const isOtherMachine = state.target !== `hardware-${config.variants[state.baseline].hardwareHash}`;
      return state.target.startsWith("hardware-") && isOtherMachine
        ? `<div class="page-row">${candidateData[0].hardwareHtml}</div>`
        : "";
    }

    async function render(state) {
      const token = (renderToken += 1);
      updateSelectors(state);
      const candidates = core.crossCandidates(state.baseline, state.target, state.softwareGroups);
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
      viewKey = viewKeyOf(state);
      machineHardware.innerHTML = baseData.hardwareHtml;
      baselineEnv.innerHTML = softwareHtml(baseData);
      if (!candidates.length) {
        view = null;
        content.innerHTML = '<section class="empty">Nothing left to compare: include more software variants in the advanced comparison settings.</section>';
        return;
      }
      view = core.crossView(state.baseline, state.target, state.softwareGroups, new Map(keys.map((key, index) => [key, dataList[index]])));
      const labelOfVariant = view.labelOfVariant;
      const tabs = sortVariants(candidates.map(labelOfVariant)).map((label) => {
        const index = candidates.findIndex((key) => labelOfVariant(key) === label);
        return {
          label,
          color: view.colors[label],
          html: comparedEnvHtml(state.target, candidateData[index]),
        };
      });
      content.innerHTML =
        sharedHardwareHtml(state, candidateData) +
        `<div class="page-row">${innerTabsHtml(tabs)}</div>` +
        dateRangeHtml([baseData, ...candidateData]);
      wireInnerTabs();
      sklbenchFormatDateRanges();
      drawGrids();
    }

    function viewKeyOf(state) {
      return `${state.baseline}|${state.target}|${state.softwareGroups.join()}`;
    }

    function onHashChange() {
      const state = readState();
      if (toggle.checked !== state.preprocessing) {
        toggle.checked = state.preprocessing;
        toggle.dispatchEvent(new Event("change", {bubbles: true}));
      }
      if (viewKey === viewKeyOf(state)) {
        drawGrids();
      } else {
        render(state);
      }
    }

    machineTabs.addEventListener("click", (event) => {
      const button = event.target.closest(".tab-button");
      if (button) {
        onMachineTab(button.dataset.machine);
      }
    });
    softwareSelect.addEventListener("change", onSoftwareChange);
    useCaseButtons.forEach((button) => {
      button.addEventListener("click", () => {
        writeState(useCaseState(button, readState()));
        machineTabs.scrollIntoView({behavior: "smooth"});
      });
    });
    groupBoxes.forEach((box) => {
      box.addEventListener("change", () => {
        const softwareGroups = groupBoxes.filter((item) => item.checked).map((item) => item.dataset.group);
        writeState({...readState(), softwareGroups});
      });
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
    // A shared link to another comparison shows how it was set up.
    const initial = readState();
    document.getElementById("cross-advanced").open =
      initial.target !== ownMachineTarget(initial.baseline) || !hasDefaultGroups(initial);
    onHashChange();
  }

  if (config.outerTabs) {
    initBaselineTabsPage();
  } else {
    initCrossPage();
  }
})();
