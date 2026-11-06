// Runs sklbench/reporting/html/speedup_core.js on a dashboard's generated data files,
// the way speedup_comparison.js does in the browser, and prints what it computes.
// Input on stdin: {corePath, dataDir, config, views}, see tests/test_speedup_comparison.py.
const fs = require("fs");
const path = require("path");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const window = {sklbenchComparisonData: {}};
new Function("window", fs.readFileSync(input.corePath, "utf8"))(window);
for (const file of fs.readdirSync(input.dataDir)) {
  new Function("window", fs.readFileSync(path.join(input.dataDir, file), "utf8"))(window);
}
const loaded = new Map(Object.entries(window.sklbenchComparisonData));
const core = window.sklbenchSpeedupCore(input.config);

function buildView(spec) {
  const softwareGroups = spec.softwareGroups || input.config.defaultSoftwareGroups;
  return core.crossView(spec.baseline, spec.target, softwareGroups, loaded);
}

const output = input.views.map((spec) => {
  const view = buildView(spec);
  return {
    baselineLabel: view.baselineLabel,
    grid: core.computeGrid(view, Boolean(spec.preprocessing)),
  };
});
// The plot template is large and irrelevant here.
process.stdout.write(JSON.stringify(output, (key, value) => (key === "template" ? undefined : value)));
