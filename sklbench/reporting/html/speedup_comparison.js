// Software, hardware and cross comparison pages: loads the data files of the
// variants on screen (sklbench/reporting/comparison.py), then pairs results and
// draws the plots and tables in the browser. Rendering every combination up front
// would take tens of MB per page.
(function () {
  const config = JSON.parse(document.getElementById("comparison-config").textContent);
  const toggle = document.querySelector(".preprocessing-toggle-checkbox");
  const content = document.getElementById("comparison-content");

  const CATEGORIES = ["linear", "tree-based", "clustering"];
  const METHODS = ["fit", "predict"];
  const SEARCH_ROW = "hyper-parameter search";
  const HEADER_FILTER = {clearable: true, sort: "asc", valuesLookup: true};
  const FAILED_MARKER_GAP = 0.4;
  const OFFSET_STEP = 0.14;
  const FALLBACK_COLOR = "#9e9e9e";
  const REFERENCE_LINE_COLOR = "#48566b";

  let nextId = 0;
  let renderToken = 0;
  let view = null;

  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#x27;"}[c]));
  const unique = (values) => Array.from(new Set(values));
  const sum = (values) => values.reduce((total, value) => total + value, 0);
  const mean = (values) => sum(values) / values.length;
  const compareStrings = (a, b) => (a < b ? -1 : a > b ? 1 : 0);

  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function stdev(values) {
    const center = mean(values);
    return Math.sqrt(sum(values.map((value) => (value - center) ** 2)) / (values.length - 1));
  }

  // statistics.quantiles(values, n=4, method="inclusive")
  function quartiles(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const at = (fraction) => {
      const position = (sorted.length - 1) * fraction;
      const low = Math.floor(position);
      const high = Math.min(low + 1, sorted.length - 1);
      return sorted[low] + (position - low) * (sorted[high] - sorted[low]);
    };
    return [at(0.25), at(0.75)];
  }

  function groupBy(items, keyOf) {
    const groups = new Map();
    items.forEach((item) => {
      const key = keyOf(item);
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key).push(item);
    });
    return groups;
  }

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

  // --- Pairing results. ---

  const entryTimes = (entry, method, preprocessing) => {
    const times = entry.methods[method].times;
    if (preprocessing && method === "fit" && entry.preprocessingTimes) {
      return times.map((time, index) => time + entry.preprocessingTimes[index]);
    }
    return times;
  };

  function methodResults(entries, method, preprocessing) {
    return entries
      .filter((entry) => entry.methods[method])
      .map((entry) => ({
        entry,
        method,
        times: entryTimes(entry, method, preprocessing),
        hover: entry.methods[method].hover,
        binned: entry.methods[method].binned,
      }));
  }

  function appendIterationsWarning(base, candidate, warnings) {
    const baseValues = base.entry.nIter.map((value) => JSON.stringify(value));
    const values = candidate.entry.nIter.map((value) => JSON.stringify(value));
    if (baseValues.some((value) => values.includes(value))) {
      return;
    }
    if (!baseValues.length && !values.length) {
      return;
    }
    const baseFirst = base.entry.nIter[0];
    const first = candidate.entry.nIter[0];
    if (!values.length) {
      warnings.push({icon: "🔁", short_message: `(${baseFirst} vs ?)`, message: "Number iteration of iteration not reported for this variant"});
    } else if (!baseValues.length) {
      warnings.push({icon: "🔁", short_message: `(? vs ${first})`, message: "Number iteration of iteration not reported for the baseline"});
    } else {
      warnings.push({icon: "🔁", short_message: `(${baseFirst} vs ${first})`, message: "Number of iteration differs: this might mean algorithms differ"});
    }
  }

  function pairWarnings(base, candidate) {
    const warnings = [];
    if (base.entry.sklearnexTree && candidate.entry.sklearnexTree) {
      // sklearnex tree cases are run both with the default max_bins and with
      // exact splits: pair each with its counterpart.
      if (base.entry.maxBins !== candidate.entry.maxBins) {
        return null;
      }
    } else if (candidate.entry.sklearnexTree) {
      if (base.entry.library === "sklearn" && candidate.binned) {
        warnings.push(config.warnings.sklearnexBins);
      }
    } else if (base.entry.sklearnexTree && !base.entry.fallback) {
      if (candidate.entry.library === "sklearn" && base.binned) {
        warnings.push(config.warnings.sklearnexBins);
      }
    }
    if (Boolean(base.entry.cumlRf) !== Boolean(candidate.entry.cumlRf)) {
      warnings.push(config.warnings.cumlRfBins);
    }
    appendIterationsWarning(base, candidate, warnings);
    if (base.entry.solver && candidate.entry.solver && base.entry.solver !== candidate.entry.solver) {
      warnings.push({icon: "🧮", short_message: `(${base.entry.solver} vs ${candidate.entry.solver})`, message: "Solver differs"});
    }
    [base, candidate].forEach((side) => {
      if (side.entry.cpuFallback) {
        warnings.push(config.warnings.cpuFallback);
      }
    });
    return warnings;
  }

  function shortMetric(value) {
    return String(Number(value.toPrecision(3)));
  }

  function metricTolerance(baseValues) {
    return Math.max(
      config.metricTolerance.stdFactor * (baseValues.length >= 2 ? stdev(baseValues) : 0),
      config.metricTolerance.absFloor,
      config.metricTolerance.relFloor * Math.abs(mean(baseValues))
    );
  }

  const isNumericList = (values) =>
    values.length > 0 && values.every((value) => typeof value === "number");

  function metricDifference(method, name, baseValues, values) {
    if (!isNumericList(baseValues) || !isNumericList(values)) {
      if (JSON.stringify(baseValues) !== JSON.stringify(values)) {
        return {method, name, base: JSON.stringify(baseValues), target: JSON.stringify(values)};
      }
      return null;
    }
    if (Math.abs(mean(values) - mean(baseValues)) > metricTolerance(baseValues)) {
      return {method, name, base: shortMetric(mean(baseValues)), target: shortMetric(mean(values))};
    }
    return null;
  }

  // Match.metrics_differences. A bigger forest scores differently, so metrics
  // aren't compared across forest sizes.
  function metricsDifferences(base, candidate) {
    if (base.method !== "fit" || base.entry.nEstimators !== candidate.entry.nEstimators) {
      return [];
    }
    const differences = [];
    const baseMetrics = base.entry.metrics;
    const metrics = candidate.entry.metrics;
    unique([...Object.keys(baseMetrics), ...Object.keys(metrics)]).sort().forEach((method) => {
      if (!baseMetrics[method]) {
        differences.push({method, name: "*", base: "(missing)", target: "(present)"});
      } else if (!metrics[method]) {
        differences.push({method, name: "*", base: "(present)", target: "(missing)"});
      } else {
        unique([...Object.keys(baseMetrics[method]), ...Object.keys(metrics[method])]).sort().forEach((name) => {
          const baseValues = baseMetrics[method][name];
          const values = metrics[method][name];
          if (baseValues === undefined) {
            differences.push({method, name, base: "(missing)", target: JSON.stringify(values)});
          } else if (values === undefined) {
            differences.push({method, name, base: JSON.stringify(baseValues), target: "(missing)"});
          } else {
            const difference = metricDifference(method, name, baseValues, values);
            if (difference) {
              differences.push(difference);
            }
          }
        });
      }
    });
    return differences;
  }

  function makeMatch(base, candidate, warnings) {
    const candidateMedian = median(candidate.times);
    return {
      base,
      candidate,
      // A fallback runs the same code as scikit-learn, so the warnings about
      // oneDAL's implementation choices would be misleading.
      warnings: candidate.entry.fallback ? [] : warnings,
      speedup: candidateMedian === 0 ? Infinity : median(base.times) / candidateMedian,
      differences: metricsDifferences(base, candidate),
    };
  }

  function findMatches(baseResults, others) {
    const matches = [];
    // A sklearnex baseline has one tree result per max_bins value for the same case.
    groupBy(baseResults, (result) => result.entry.maxBins ?? null).forEach((group) => {
      // A software env rebuild can leave two baseline runs of one case: keep the
      // most recent.
      const baseById = new Map();
      group.forEach((result) => {
        const current = baseById.get(result.entry.id);
        if (!current || result.entry.timestamp > current.entry.timestamp) {
          baseById.set(result.entry.id, result);
        }
      });
      others.forEach((candidate) => {
        const base = baseById.get(candidate.entry.id);
        if (base) {
          const warnings = pairWarnings(base, candidate);
          if (warnings) {
            matches.push(makeMatch(base, candidate, warnings));
          }
        }
      });
    });
    return matches;
  }

  // --- Plots. ---

  function sortVariants(variants) {
    const rank = (variant) => {
      if (!config.labelOrder) {
        return 0;
      }
      const index = config.labelOrder.indexOf(variant.split(" / ")[0]);
      return index === -1 ? config.labelOrder.length : index;
    };
    return [...variants].sort((a, b) => rank(a) - rank(b) || compareStrings(a, b));
  }

  // Within a variant, tree cases without binning go left. Across machines, both
  // sides of a sklearnex comparison can bin the same way, which no warning tells.
  function columnOf(label, category, binned, warnings) {
    if (category !== "tree-based") {
      return label;
    }
    const warned = warnings.some((warning) => config.binnedWarningKinds.includes(warning.kind));
    const isBinned = warned || (config.splitsFromParams && binned);
    return `${label} / ${isBinned ? "1-binned" : "0-exact"}`;
  }

  function formatDuration(time) {
    if (time < 10) {
      return `${Number(time.toPrecision(2))}ms`;
    }
    if (time < 1000) {
      return `${Math.round(time)}ms`;
    }
    if (time < 10000) {
      return `${Number((time / 1000).toPrecision(2))}s`;
    }
    return `${Math.round(time / 1000)}s`;
  }

  function differenceLines(match) {
    const differences = match.differences;
    if (!differences.length) {
      return [];
    }
    const methodLabel = (method) => ({fit: "train", predict: "test"}[method] || method);
    const lineOf = (d) => escapeHtml(`${d.name}: ${d.base} (base) vs ${d.target} (variant)`);
    const fit = differences.filter((d) => d.method === "fit");
    const other = differences.filter((d) => d.method !== "fit");
    if (other.length) {
      const lines = [`<b>${escapeHtml(methodLabel(other[0].method))} metrics differ:</b>`, ...other.map(lineOf)];
      if (fit.length) {
        lines.push(`<i>${methodLabel("fit")} metrics also mismatch</i>`);
      }
      return lines;
    }
    return [`<b>${methodLabel("fit")} metrics differ:</b>`, ...fit.map(lineOf)];
  }

  function hoverText(match) {
    const lines = [
      `<b>speed-up: ${Number(match.speedup.toPrecision(2))}x</b> ` +
        `(${formatDuration(median(match.base.times))} vs ${formatDuration(median(match.candidate.times))})`,
    ];
    if (match.base.entry.searchNJobs !== undefined) {
      lines.push(`outer n_jobs: ${match.base.entry.searchNJobs} vs ${match.candidate.entry.searchNJobs}`);
    }
    lines.push(...differenceLines(match));
    if (match.candidate.entry.fallback) {
      lines.push("<i>fell back to scikit-learn</i>");
    }
    match.warnings.forEach((warning) => {
      lines.push(escapeHtml(warning.short_message ? `${warning.icon} ${warning.short_message}` : warning.icon));
    });
    lines.push(match.candidate.hover);
    return lines.join("<br>");
  }

  function markerSymbol(match) {
    if (match.differences.length) {
      return "diamond-open";
    }
    return match.warnings.length ? "square" : "circle";
  }

  function markerNotesHtml(matches, failedCount) {
    const mismatchCount = matches.filter((match) => match.differences.length).length;
    const fallbackCount = matches.filter((match) => match.candidate.entry.fallback).length;
    const points = (count) => (count === 1 ? `${count} point` : `${count} points`);
    const warnings = new Map();
    matches
      .filter((match) => !match.differences.length)
      .forEach((match) => match.warnings.forEach((w) => warnings.set(`${w.icon}${w.message}`, w)));
    const markers = {"plot-marker-x": "✕", "plot-marker-square": "■", "plot-marker-diamond": "◇", "plot-marker-circle": "●"};
    const note = (markerClass, style, text) =>
      `<div class="plot-note"><span class="plot-marker ${markerClass}"${style}>${markers[markerClass]}</span><span>${text}</span></div>`;
    const parts = [note("plot-marker-circle", "", "Metrics and benchmark setup match the baseline")];
    if (warnings.size) {
      const items = Array.from(warnings.values())
        .map((w) => `<li class="warning-note"><span class="warning-icon">${w.icon}</span><span>${escapeHtml(w.message)}</span></li>`)
        .join("");
      parts.push(note("plot-marker-square", "", `Metrics match the baseline, but some comparison details are worth reporting:<ul>${items}</ul>`));
    }
    if (mismatchCount) {
      parts.push(note("plot-marker-diamond", "", `Metrics differ from the baseline (${points(mismatchCount)})`));
    }
    if (fallbackCount) {
      parts.push(note("plot-marker-circle", ' style="color: #9e9e9e;"', `Ran via sklearnex but fell back to stock scikit-learn, no oneDAL acceleration (${points(fallbackCount)})`));
    }
    if (failedCount) {
      parts.push(note("plot-marker-x", "", `Benchmark run failed, shown at the bottom of its column (${points(failedCount)})`));
    }
    return `<div class="plot-notes">${parts.join("")}</div>`;
  }

  function buildPlot(matches, failed) {
    const items = matches.map((match) => ({
      match,
      entry: match.candidate.entry,
      variant: view.labelOf(match.candidate.entry),
      column: match.column,
      estimator: match.candidate.entry.estimator,
    }));
    const failedItems = failed.map((entry) => ({
      entry,
      variant: view.labelOf(entry),
      column: columnOf(view.labelOf(entry), entry.category, entry.binned, []),
      estimator: entry.estimator,
    }));
    const allItems = [...items, ...failedItems];
    const estimators = unique(allItems.map((item) => item.estimator)).sort();
    const positions = new Map(estimators.map((estimator, index) => [estimator, index]));
    const columns = sortVariants(unique(allItems.map((item) => item.column)));
    const center = (columns.length - 1) / 2;
    const offsets = new Map(
      columns.map((column, index) => [column, columns.length > 1 ? (index - center) * OFFSET_STEP : 0])
    );

    const columnMin = new Map();
    items.forEach((item) => {
      const key = `${item.estimator}|${item.column}`;
      const y = Math.log2(item.match.speedup);
      columnMin.set(key, Math.min(columnMin.get(key) ?? y, y));
    });
    const values = items.map((item) => Math.log2(item.match.speedup));
    const fallbackMin = values.length ? Math.min(...values) : 0;
    const failedY = (item) =>
      (columnMin.get(`${item.estimator}|${item.column}`) ?? fallbackMin) - FAILED_MARKER_GAP;

    const traces = [];
    sortVariants(unique(allItems.map((item) => item.variant))).forEach((variant) => {
      const color = view.colors[variant];
      const common = {mode: "markers", type: "scatter", name: variant, legendgroup: variant};
      traces.push({...common, showlegend: true, hoverinfo: "skip", x: [null], y: [null], marker: {size: 10, symbol: "circle", color}});
      const rank = (item) => (markerSymbol(item.match) !== "circle" ? 1 : 0);
      const variantItems = items
        .filter((item) => item.variant === variant)
        .sort((a, b) => rank(a) - rank(b) || compareStrings(a.estimator, b.estimator));
      if (variantItems.length) {
        traces.push({
          ...common,
          showlegend: false,
          x: variantItems.map((item) => positions.get(item.estimator) + offsets.get(item.column)),
          y: variantItems.map((item) => Math.log2(item.match.speedup)),
          text: variantItems.map((item) => hoverText(item.match)),
          customdata: variantItems.map((item) => item.entry.id),
          hovertemplate: "%{text}<extra></extra>",
          marker: {
            size: 10,
            symbol: variantItems.map((item) => markerSymbol(item.match)),
            color: variantItems.map((item) => (item.entry.fallback ? FALLBACK_COLOR : color)),
          },
        });
      }
      const variantFailed = failedItems.filter((item) => item.variant === variant);
      if (variantFailed.length) {
        traces.push({
          ...common,
          showlegend: false,
          x: variantFailed.map((item) => positions.get(item.estimator) + offsets.get(item.column)),
          y: variantFailed.map(failedY),
          text: variantFailed.map((item) => item.entry.hover),
          customdata: variantFailed.map((item) => item.entry.id),
          hovertemplate: "%{text}<extra></extra>",
          marker: {size: 10, symbol: "x", color},
        });
      }
    });

    const allValues = [...values, ...failedItems.map(failedY)];
    const minTick = Math.floor(Math.min(...allValues, 0));
    const maxTick = Math.ceil(Math.max(...allValues, 0));
    const ticks = Array.from({length: maxTick - minTick + 1}, (_, index) => minTick + index);
    const layout = {
      template: config.plotTemplate,
      xaxis: {
        tickmode: "array",
        tickvals: estimators.map((_, index) => index),
        ticktext: estimators.map((estimator) =>
          config.estimatorAbbreviations.reduce((name, [full, short]) => name.replace(full, short), estimator)
        ),
        range: [-0.5, estimators.length - 0.5],
      },
      yaxis: {
        title: {text: `speed-up vs ${view.baselineLabel}`},
        tickmode: "array",
        tickvals: ticks,
        ticktext: ticks.map((tick) => (tick >= 0 ? `${2 ** tick}x` : `1/${2 ** -tick}x`)),
      },
      shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: 0, y1: 0, line: {color: REFERENCE_LINE_COLOR, width: 1, dash: "dash"}}],
      annotations: [{
        xref: "paper", x: 0, xanchor: "left", yref: "y", y: 0, yanchor: "bottom",
        text: `1x = ${view.baselineLabel}`, showarrow: false,
        font: {size: 11, color: REFERENCE_LINE_COLOR}, bgcolor: "rgba(255, 255, 255, 0.7)",
      }],
      margin: {l: 70, r: 20, t: 20, b: 110},
      showlegend: true,
      legend: {orientation: "h"},
    };
    return {traces, layout, notes: markerNotesHtml(matches, failedItems.length)};
  }

  // --- Detailed results tables. ---

  function speedupInterval(base, candidate) {
    if (base.times.length < 3 || candidate.times.length < 3) {
      return null;
    }
    const [baseQ1, baseQ3] = quartiles(base.times);
    const [q1, q3] = quartiles(candidate.times);
    return q1 <= 0 || baseQ1 <= 0 ? null : [baseQ1 / q3, baseQ3 / q1];
  }

  function formatNIter(entry, baseEntry) {
    if (entry.nIterMedian === undefined) {
      return undefined;
    }
    let label = entry.nIterMedian.toFixed(0);
    const baseNIter = baseEntry ? baseEntry.nIterMedian : undefined;
    if (baseNIter && entry.nIterMedian !== baseNIter) {
      const ratio = entry.nIterMedian / baseNIter;
      label += ratio >= 10 ? ` (${ratio.toFixed(0)}x)` : ` (${Number(ratio.toPrecision(2))}x)`;
    }
    return label;
  }

  function addScores(row, entry, baseEntry) {
    const metrics = (entry.metrics || {}).predict || {};
    const baseMetrics = baseEntry ? (baseEntry.metrics || {}).predict || {} : {};
    Object.entries(config.scoreMetrics).forEach(([field, name]) => {
      const values = (metrics[name] || []).filter((value) => typeof value === "number");
      if (!values.length) {
        return;
      }
      row[field] = mean(values);
      if (!baseEntry || baseEntry === entry || !baseMetrics[name] || baseEntry.nEstimators !== entry.nEstimators) {
        return;
      }
      if (!isNumericList(baseMetrics[name])) {
        return;
      }
      const difference = mean(values) - mean(baseMetrics[name]);
      if (Math.abs(difference) > metricTolerance(baseMetrics[name])) {
        row[`${field}_cmp`] = difference > 0 ? "better" : "worse";
      }
    });
  }

  function addResultMethod(rows, result, base, variant, isBaseline) {
    const key = `${variant}:${result.entry.path}`;
    if (!rows.has(key)) {
      rows.set(key, {...result.entry.row, variant, row_id: key, is_baseline: false});
    }
    const row = rows.get(key);
    if (isBaseline) {
      row.is_baseline = true;
    }
    addScores(row, result.entry, base ? base.entry : null);
    const method = result.method;
    if (method === "fit") {
      row.n_samples = result.entry.methods.fit.samples;
      row.n_features = result.entry.methods.fit.features;
      const nIter = formatNIter(result.entry, base ? base.entry : null);
      if (nIter !== undefined) {
        row.n_iter = nIter;
      }
    }
    row[`${method}_time`] = median(result.times);
    row[`${method}_speedup`] = base ? median(base.times) / median(result.times) : null;
    const interval = base && base !== result ? speedupInterval(base, result) : null;
    if (interval) {
      row[`${method}_speedup_low`] = interval[0];
      row[`${method}_speedup_high`] = interval[1];
    }
  }

  function addGroupRanks(rows) {
    const compare = (a, b) => {
      for (let index = 0; index < a.length; index += 1) {
        if (a[index] < b[index]) return -1;
        if (a[index] > b[index]) return 1;
      }
      return 0;
    };
    const groups = Array.from(groupBy(rows, (row) => row.comparison_key).entries());
    const groupKey = ([key, group]) => [
      group[0].estimator,
      group[0].dataset,
      Math.min(...group.map((row) => row.n_samples || -1)),
      Math.min(...group.map((row) => row.n_features || -1)),
      key,
    ];
    groups.sort((a, b) => compare(groupKey(a), groupKey(b)));
    groups.forEach(([, group], groupRank) => {
      const baselineVariants = new Set(group.filter((row) => row.is_baseline).map((row) => row.variant));
      const within = (row) => [
        baselineVariants.has(row.variant) ? 0 : 1, row.variant, row.is_baseline ? 0 : 1, row.env || "", row.row_id,
      ];
      [...group].sort((a, b) => compare(within(a), within(b))).forEach((row, withinRank) => {
        row.group_rank = groupRank;
        row.within_rank = withinRank;
      });
    });
  }

  const formatValue = (value) => {
    if (value === null || value === undefined) return "N/A";
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  };
  const variesAcross = (rows, field) => new Set(rows.map((row) => formatValue(row[field]))).size > 1;

  function isColumnVisible(spec, rows) {
    const field = spec.column.field;
    if (spec.visibility === "custom") {
      if (field === "status") {
        return rows.some((row) => row.status !== "ok");
      }
      const hgbRows = rows.filter((row) => String(row.estimator || "").startsWith("HistGradientBoosting"));
      return hgbRows.length > 0 && variesAcross(hgbRows, field);
    }
    if (spec.visibility === "if_varies") {
      return variesAcross(rows, field);
    }
    if (spec.visibility === "if_any") {
      return rows.some((row) => row[field]);
    }
    return true;
  }

  function buildColumns(rows, variantTitle) {
    const ompSpec = config.columns.find((spec) => spec.column && spec.column.field === "omp_active_wait");
    const hiddenEnv = isColumnVisible(ompSpec, rows) ? ["GOMP_SPINCOUNT", "KMP_BLOCKTIME"] : [];
    const columns = [];
    config.columns.forEach((spec) => {
      if (spec.group) {
        const groupKey = `__group_${spec.group}`;
        const names = spec.allowedKeys || unique(rows.flatMap((row) => Object.keys(row[groupKey] || {}))).sort();
        names
          .filter((name) => !(spec.group === "env" && hiddenEnv.includes(name)))
          .filter((name) => new Set(rows.map((row) => formatValue((row[groupKey] || {})[name]))).size > 1)
          .forEach((name, index) => {
            const field = `${spec.group}_${index}`;
            rows.forEach((row) => {
              row[field] = formatValue((row[groupKey] || {})[name]);
            });
            columns.push({title: name, field, headerFilter: "list", headerFilterParams: HEADER_FILTER, sorter: "string"});
          });
      } else if (isColumnVisible(spec, rows)) {
        const column = {...spec.column};
        if (column.field === "variant") {
          column.title = variantTitle;
        }
        if (column.field === "dataset" && new Set(rows.map((row) => Boolean(row.real_dataset))).size > 1) {
          const names = unique(rows.map((row) => row.dataset)).sort((a, b) => compareStrings(a.toLowerCase(), b.toLowerCase()));
          column.headerFilterParams = {clearable: true, values: [...Object.keys(config.datasetKindFilters), ...names]};
          column.datasetKindFilters = config.datasetKindFilters;
        }
        columns.push(column);
      }
    });
    return columns;
  }

  function buildTable(allMatches, failed, unmatchedBase, unmatchedCandidate) {
    const rows = new Map();
    allMatches.forEach((match) => {
      addResultMethod(rows, match.base, match.base, view.baselineLabel, true);
      addResultMethod(rows, match.candidate, match.base, view.labelOf(match.candidate.entry), false);
    });
    failed.forEach((entry) => {
      const variant = view.isBaseline(entry) ? view.baselineLabel : view.labelOf(entry);
      const key = `${variant}:${entry.path}`;
      rows.set(key, {...entry.row, variant, row_id: key, is_baseline: false});
    });
    unmatchedBase.forEach((result) => addResultMethod(rows, result, null, view.baselineLabel, true));
    unmatchedCandidate.forEach((result) => addResultMethod(rows, result, null, view.labelOf(result.entry), false));
    if (!rows.size) {
      return null;
    }
    const rowList = Array.from(rows.values());
    const columns = buildColumns(rowList, config.columnTitle);
    addGroupRanks(rowList);
    rowList.sort((a, b) => a.group_rank - b.group_rank || a.within_rank - b.within_rank);
    return {rows: rowList, columns};
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

  const cellTitle = (category, method, count) =>
    `${category} - ${method} time - ${count} ${count === 1 ? "case" : "cases"}`;

  function buildGrid(pending) {
    const preprocessing = toggle.checked;
    const keep = (entry) => !preprocessing || entry.real;
    const baseEntries = view.baseData.results.filter(keep);
    const candidateEntries = view.candidateData.flatMap((data) => data.results).filter(keep);
    const failed = [...view.baseData.failed, ...view.candidateData.flatMap((data) => data.failed)].filter(keep);
    if (!baseEntries.length) {
      return `<section class="empty">No ${escapeHtml(view.baselineLabel)} baseline results for ${escapeHtml(view.emptyScope)}.</section>`;
    }
    const baseByMethod = Object.fromEntries(METHODS.map((m) => [m, methodResults(baseEntries, m, preprocessing)]));
    const candidateByMethod = Object.fromEntries(METHODS.map((m) => [m, methodResults(candidateEntries, m, preprocessing)]));
    const failedCandidates = failed.filter((entry) => !view.isBaseline(entry));

    const cells = [];
    CATEGORIES.forEach((category) => {
      const inCategory = (result) => result.entry.category === category;
      const categoryFailed = failed.filter((entry) => entry.category === category);
      const plotFailed = failedCandidates.filter((entry) => entry.category === category);
      const allMatches = [];
      const rowCells = [];
      METHODS.forEach((method) => {
        const base = baseByMethod[method].filter(inCategory);
        if (!base.length) {
          rowCells.push(`<section class="plot-cell empty" data-method="${method}">No data</section>`);
          return;
        }
        const matches = findMatches(base, candidateByMethod[method].filter(inCategory));
        matches.forEach((match) => {
          match.column = columnOf(view.labelOf(match.candidate.entry), category, match.candidate.binned, match.warnings);
        });
        allMatches.push(...matches);
        const caseCount = new Set([...matches.map((m) => m.base.entry.id), ...plotFailed.map((e) => e.id)]).size;
        const plot = matches.length || plotFailed.length ? buildPlot(matches, plotFailed) : null;
        rowCells.push(plotCellHtml(cellTitle(category, method, caseCount), method, plot, pending));
      });
      const hasPlot = METHODS.some((method) => baseByMethod[method].some(inCategory));
      if (hasPlot && category === "tree-based") {
        cells.push(`<p class="plot-row-note">${view.treeNote}</p>`);
      }
      cells.push(...rowCells);

      // Without this the side that succeeded would vanish along with its failed counterpart.
      const baseResults = METHODS.flatMap((m) => baseByMethod[m].filter(inCategory));
      const candidateResults = METHODS.flatMap((m) => candidateByMethod[m].filter(inCategory));
      const unmatchedBase = [];
      const unmatchedCandidate = [];
      categoryFailed.forEach((entry) => {
        if (view.isBaseline(entry)) {
          unmatchedCandidate.push(...candidateResults.filter((result) => result.entry.id === entry.id));
        } else {
          unmatchedBase.push(...baseResults.filter((result) => result.entry.id === entry.id));
        }
      });
      cells.push(tableHtml(buildTable(allMatches, categoryFailed, unmatchedBase, unmatchedCandidate), pending));
    });
    return `<section class="plot-grid method-grid" style="grid-template-columns: repeat(2, minmax(0, 1fr)); --collapsed-columns: 1fr 44px;">${cells.join("")}</section>`;
  }

  function buildSearchGrid(pending) {
    const toResults = (entries) =>
      entries.map((entry) => ({entry, method: "fit", times: entry.methods.fit.times, hover: entry.methods.fit.hover, binned: false}));
    const base = new Map(toResults(view.baseData.search).map((result) => [result.entry.id, result]));
    const matches = toResults(view.candidateData.flatMap((data) => data.search))
      .filter((candidate) => base.has(candidate.entry.id))
      .map((candidate) => {
        const match = makeMatch(base.get(candidate.entry.id), candidate, []);
        match.differences = [];
        match.column = view.labelOf(candidate.entry);
        return match;
      });
    if (!matches.length) {
      return "";
    }
    const caseCount = new Set(matches.map((match) => match.base.entry.id)).size;
    const cell = plotCellHtml(cellTitle(SEARCH_ROW, "fit", caseCount), "fit", buildPlot(matches, []), pending);
    const table = tableHtml(buildTable(matches, [], [], []), pending);
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

  function drawGrids() {
    const container = document.getElementById("comparison-grids");
    container.querySelectorAll(".plotly-graph-div").forEach((chart) => Plotly.purge(chart));
    window.sklbenchTables = {};
    const pending = {charts: [], tables: []};
    container.innerHTML = `<div class="page-row">${buildGrid(pending)}</div><div class="page-row">${buildSearchGrid(pending)}</div>`;
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

    function setView(tab, loaded, baselineLabel) {
      const labels = new Map(tab.variants.map((variant) => [variant.key, variant.label]));
      const baselineKey = tab.variants.find((variant) => variant.label === baselineLabel).key;
      const candidates = tab.variants.filter((variant) => variant.key !== baselineKey && variant.plotted);
      view = {
        baselineLabel,
        baseData: loaded.get(baselineKey),
        candidateData: candidates.map((variant) => loaded.get(variant.key)),
        colors: config.colors,
        emptyScope: tab.emptyScope,
        treeNote: config.treeNote,
        labelOf: (entry) => labels.get(entry.variant),
        isBaseline: (entry) => entry.variant === baselineKey,
      };
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
        html: (variant.pickHtml || "") + loaded.get(variant.key).envHtml,
      }));
      panel.innerHTML =
        `<div class="page-row">${dateRangeHtml(dataList)}</div>` +
        (tab.topHtml ? `<div class="page-row">${tab.topHtml}</div>` : "") +
        `<div class="page-row">${innerTabsHtml(tabs)}</div>` +
        '<div id="comparison-grids"></div>';
      wireInnerTabs();
      sklbenchFormatDateRanges();
      setBaselineButtons(baselineLabel);
      setView(tab, loaded, baselineLabel);
      view.loaded = loaded;
      view.tab = tab;
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
      setView(tab, loaded, pickedBaseline);
      view.loaded = loaded;
      view.tab = tab;
      drawGrids();
    });
    toggle.addEventListener("change", () => {
      if (view) {
        drawGrids();
      }
    });
    showOuter();
  }

  // --- Cross page: the picked combination lives in the URL hash. ---

  function initCrossPage() {
    const softwareSelect = document.getElementById("cross-software");
    const machineSelect = document.getElementById("cross-machine");
    const targetSelect = document.getElementById("cross-target");
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

    function targetLabel(target) {
      return target.startsWith("hardware-")
        ? config.machineNames[target.slice("hardware-".length)]
        : target.slice("software-".length);
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
      const targets = config.pages[state.baseline];
      targetSelect.innerHTML = [
        ["One software, on every machine", "software-"],
        ["One machine, with every software", "hardware-"],
      ]
        .map(([label, prefix]) => {
          const options = targets
            .filter((target) => target.startsWith(prefix))
            .map((target) => `<option value="${escapeHtml(target)}">${escapeHtml(targetLabel(target))}</option>`)
            .join("");
          return options ? `<optgroup label="${label}">${options}</optgroup>` : "";
        })
        .join("");
      targetSelect.value = state.target;
    }

    function onSelectorChange() {
      const software = softwareSelect.value;
      const keys = Object.keys(config.pages).filter((key) => config.variants[key].software === software);
      const baseline = keys.find((key) => config.variants[key].hardwareHash === machineSelect.value) || keys[0];
      const targets = config.pages[baseline];
      writeState({
        baseline,
        target: targets.includes(targetSelect.value) ? targetSelect.value : targets[0],
        preprocessing: toggle.checked,
      });
    }

    async function render(state) {
      const token = (renderToken += 1);
      updateSelectors(state);
      const baseline = config.variants[state.baseline];
      const isSoftwareTarget = state.target.startsWith("software-");
      const value = state.target.slice(state.target.indexOf("-") + 1);
      const candidates = Object.keys(config.variants).filter((key) => {
        const variant = config.variants[key];
        return key !== state.baseline && (isSoftwareTarget ? variant.software === value : variant.hardwareHash === value);
      });
      content.innerHTML = '<section class="empty">Loading results...</section>';
      let baseData;
      let candidateData;
      try {
        [baseData, ...candidateData] = await Promise.all([state.baseline, ...candidates].map(loadVariant));
      } catch (error) {
        content.innerHTML = `<section class="empty">${escapeHtml(error.message)}</section>`;
        return;
      }
      if (token !== renderToken) {
        return;
      }
      const labelOfVariant = (key) =>
        isSoftwareTarget ? config.variants[key].hardwareLabel : config.variants[key].software;
      viewKey = `${state.baseline}|${state.target}`;
      view = {
        baselineLabel: `${baseline.software} on ${baseline.hardwareLabel}`,
        baseData,
        candidateData,
        colors: isSoftwareTarget ? config.hardwareColors : config.softwareColors,
        emptyScope: "real datasets",
        treeNote: config.treeNotes[isSoftwareTarget ? "machine" : "software"],
        labelOf: (entry) => labelOfVariant(entry.variant),
        isBaseline: (entry) => entry.variant === state.baseline,
      };
      config.columnTitle = isSoftwareTarget ? "Hardware" : "Software";
      const tabs = [
        {label: view.baselineLabel, html: baseData.envHtml, isBaseline: true},
        ...sortVariants(candidates.map(labelOfVariant)).map((label) => ({
          label,
          color: view.colors[label],
          html: candidateData[candidates.findIndex((key) => labelOfVariant(key) === label)].envHtml,
        })),
      ];
      content.innerHTML =
        `<div class="page-row">${dateRangeHtml([baseData, ...candidateData])}</div>` +
        `<div class="page-row">${innerTabsHtml(tabs)}</div>` +
        '<div id="comparison-grids"></div>';
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

    [softwareSelect, machineSelect, targetSelect].forEach((select) => {
      select.addEventListener("change", onSelectorChange);
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
