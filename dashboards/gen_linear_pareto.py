"""Fit time vs suboptimality of L2-penalized linear models, scikit-learn vs
glum, from `configs/linear_pareto.py`.

One tab per machine, one cell per problem (dataset and loss), with two plots:
the gap to the best training objective found, and the test loss. One series
per (library, solver). A scikit-learn run from another env than `glm` (e.g. a
`sklearn-dev` branch) is its own series.
"""
from html import escape
from pathlib import Path
from statistics import median

from dashboards import HARDWARE_NAMES
from sklbench.reporting.envs import read_env, software_build_name, summarize_software_env
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    SERIES_COLORS,
    SOFTWARE_TEMPLATE,
    ladder_plot_html,
    render_hardware_tabs,
    render_software_tabs,
)
from sklbench.reporting.matching import (
    BenchmarkRecord, Implementation, date_range, matches_source_configs,
    read_benchmark_records,
)


TITLE = "Linear models: fit time vs suboptimality"
SOURCE_CONFIGS = ["configs/linear_pareto.py"]
SOURCE_ENVS = ["glm"]

GLM_ENV = "glm"
LIBRARY_LABELS = {"sklearn": "scikit-learn", "glum": "glum"}
# Series order and colors: scikit-learn's solvers first.
SOLVER_ORDER = [
    ("sklearn", "lbfgs"),
    ("sklearn", "newton-cg"),
    ("sklearn", "newton-cholesky"),
    ("sklearn", "sag"),
    ("sklearn", "saga"),
    ("glum", "irls-ls"),
    ("glum", "irls-cd"),
]
LOSS_LABELS = {
    "binomial": "logistic regression",
    "multinomial": "multinomial logistic regression",
    "poisson": "Poisson GLM",
    "gamma": "Gamma GLM",
}
PROBLEM_ORDER = [
    "bank_marketing",
    "kick",
    "kddcup09_churn",
    "letters",
    "covtype",
    "fremtpl2_freq",
    "fremtpl2_sev",
]
# Relative gaps below this are rounding noise of the objective's evaluation,
# shown at this value.
SUBOPTIMALITY_FLOOR = 1e-12
# Lowest test loss shown at first, above each plot's best point, relative to
# that best point.
TEST_LOSS_ZOOM = 0.05

ABOUT_HTML = """<section class="panel">
  <p>This dashboard compares the <dfn>solvers</dfn> of scikit-learn's
  <b>linear models</b> with those of <b>glum</b>, a library specialized in
  <dfn>GLMs</dfn>, on real datasets: logistic regression, and Poisson and Gamma
  GLMs, all with an L2 penalty. For a given dataset and penalty, every solver
  that converges finds the same model, so the libraries don't differ in
  accuracy but in how fast they get close to the optimum. So each solver fits
  each problem at a ladder of <dfn>tolerances</dfn>, from loose to tight, and
  each plot shows how far from the optimum it stops, and how long it takes.</p>
  <details class="about-section">
    <summary>How to read</summary>
    <p>There is one tab per machine, and two plots per problem. In both, the
    x-axis is the median <code><dfn>fit</dfn></code> time over 3 train/test
    splits (log scale), and each line joins one solver's points, from its
    loosest tolerance to its tightest. The star is the solver at its default
    tolerance and <code>max_iter</code>, which is what users get. The top plot
    shows the <dfn>suboptimality</dfn> of the fit (log scale): lower and to the
    left is better. The bottom plot shows the test loss: the
    <dfn>log loss</dfn> for logistic regression, half the mean
    <dfn>deviance</dfn> for the GLMs. It shows how much precision matters for
    predictions, and is zoomed on the best points: double-click it to see
    all of them. Hover a point to see its tolerance and number of
    iterations.</p>
    <p>All libraries minimize the same objective: the mean loss on the
    training set plus <code>alpha / 2</code> times the squared L2 norm of the
    coefficients, without penalizing the intercept. scikit-learn's
    <code>LogisticRegression</code> takes <code>C = 1 / (alpha *
    n_samples)</code>, its GLMs and glum take <code>alpha</code> directly.
    <code>alpha</code> is about the one with the best test loss on each
    dataset. The tolerances are each solver's own stopping criterion
    (<code>tol</code> in scikit-learn, <code>gradient_tol</code> in glum),
    from 0.1 to 1e-9, with a <code>max_iter</code> of 10,000 so that the
    tolerance is what stops the solver. Their meaning differs between
    solvers, so a point's tolerance only places it on its own solver's
    line.</p>
    <p>Every library gets the same features: one-hot and
    <dfn>target encoding</dfn> of the categorical columns, and splines of the
    numeric ones. glum runs on this dense matrix, not on its own categorical
    matrix type. The Poisson problem predicts the number of claims of each
    car insurance policy, with the log of its exposure (the fraction of the
    year it was in force) as a feature. The Gamma problem predicts the mean
    amount of a policy's claims. Some solvers are left out: liblinear, which
    penalizes the intercept and so solves another problem, glum's lbfgs,
    which fails with recent SciPy versions, and glum's trust-constr, meant
    for constrained problems. glum has no multinomial model.</p>
    <p>Both libraries come from <dfn>conda-forge</dfn> and share the
    <dfn>MKL</dfn> build of <dfn>BLAS</dfn>, the fastest one for
    scikit-learn's linear models. Every library uses its default number of
    <dfn>threads</dfn>. On the Xeon server, the cases run in 8 processes at
    once, each pinned to 21 cores and the memory of one of its 8
    <dfn>NUMA nodes</dfn>.</p>
  </details>
  <details class="about-section">
    <summary>Findings</summary>
    <p>On the benchmarked cases, on one NUMA node of the Xeon server:</p>
    <ul>
      <li><b>glum</b>'s default solver (irls-ls) reaches a suboptimality
      below 1e-8 at its default tolerance, and is the fastest to get there on
      3 of the 4 binary and Poisson problems: ~1.4x to 2.6x faster than
      <b>scikit-learn</b>'s newton-cholesky, its fastest solver on them. On
      kddcup09_churn (~2,200 features), newton-cholesky is ~1.2x faster.</li>
      <li><b>scikit-learn</b>'s default solver, lbfgs, stops at a
      suboptimality of ~3e-4 to ~6e-3 at its default tolerance (apart from
      the Gamma problem below). That costs up to 0.14% of test loss on the
      binary and Poisson problems, and 0.5% to 1% on the multiclass ones.
      Reaching 1e-8 with lbfgs takes ~1.3x to ~40x longer than with
      <b>glum</b>'s irls-ls on the binary and Poisson problems.</li>
      <li>On the Gamma problem, <b>scikit-learn</b>'s lbfgs stops after 1
      iteration at every tolerance, ~2% above the optimum, without a warning.
      Its first line search step overflows the loss on the target-encoded
      columns (in the thousands), and L-BFGS-B then reports convergence.
      newton-cholesky and <b>glum</b> converge in ~0.01s.</li>
      <li>For multinomial logistic regression (<b>glum</b> has none),
      newton-cg is the fastest <b>scikit-learn</b> solver on letters (26
      classes), and newton-cholesky is ~15x slower, as its Hessian grows with
      the square of the number of classes. On covtype, sag, saga, newton-cg
      and newton-cholesky are within ~1.2x of each other, and lbfgs is ~3x
      slower.</li>
      <li><b>scikit-learn</b>'s sag and saga are ~4x to ~140x slower than
      newton-cholesky to reach 1e-4 on the binary problems.</li>
    </ul>
  </details>
</section>"""


def _dataset(record: BenchmarkRecord) -> str:
    return record.case["data"]["dataset"]


def _objective(record: BenchmarkRecord) -> dict:
    return record.case["algorithm"]["objective"]


def _solver(record: BenchmarkRecord) -> str:
    return record.case["metadata"]["solver"]


def _tol(record: BenchmarkRecord) -> float | str:
    return record.case["metadata"]["tol"]


def _series_label(record: BenchmarkRecord) -> str:
    library = record.implementation.library
    label = f"{LIBRARY_LABELS.get(library, library)} {_solver(record)}"
    build = software_build_name(record.software_hash)
    if build != GLM_ENV:
        label += f" ({build})"
    return label


def _base_labels() -> list[str]:
    return [f"{LIBRARY_LABELS[library]} {solver}" for library, solver in SOLVER_ORDER]


def _label_sort_key(label: str) -> tuple:
    base = label.split(" (")[0]
    order = _base_labels()
    return (order.index(base) if base in order else len(order), label)


def _series_colors(labels: set[str]) -> dict[str, str]:
    """Each solver keeps its color across tabs and problems; another build of
    the same solver gets the same color, with a dashed line."""
    base_colors = {
        label: SERIES_COLORS[index % len(SERIES_COLORS)]
        for index, label in enumerate(_base_labels())
    }
    extra = sorted({label.split(" (")[0] for label in labels} - set(base_colors))
    for index, label in enumerate(extra, start=len(base_colors)):
        base_colors[label] = SERIES_COLORS[index % len(SERIES_COLORS)]
    return {label: base_colors[label.split(" (")[0]] for label in labels}


def _line_dashes(labels: set[str]) -> dict[str, str]:
    return {label: "dash" for label in labels if " (" in label}


def _dedup_latest(records: list[BenchmarkRecord]) -> list[BenchmarkRecord]:
    latest: dict[tuple, BenchmarkRecord] = {}
    for record in records:
        key = (
            record.hardware_hash,
            record.software_hash,
            record.implementation.library,
            _dataset(record),
            _solver(record),
            str(_tol(record)),
        )
        current = latest.get(key)
        if current is None or record.timestamp_recorded > current.timestamp_recorded:
            latest[key] = record
    return list(latest.values())


def _run_key(record: BenchmarkRecord, run: dict) -> tuple:
    """Runs whose objectives are comparable: same machine, problem, and
    training data (split and preprocessing)."""
    objective = _objective(record)
    return (
        record.hardware_hash,
        _dataset(record),
        objective["loss"],
        objective["alpha"],
        run["random_state"]["data"],
        run["data_desc"]["fit"]["hash"],
    )


def _has_objective(run: dict) -> bool:
    return "objective" in run["metrics"].get("fit", {})


def best_objectives(records: list[BenchmarkRecord]) -> dict[tuple, float]:
    best: dict[tuple, float] = {}
    for record in records:
        for run in record.runs:
            if _has_objective(run):
                key = _run_key(record, run)
                value = run["metrics"]["fit"]["objective"]
                best[key] = min(best.get(key, value), value)
    return best


def suboptimality(record: BenchmarkRecord, run: dict, best: dict[tuple, float]) -> float:
    """Relative gap to the best objective of the same training data, floored
    at SUBOPTIMALITY_FLOOR."""
    reference = best[_run_key(record, run)]
    gap = (run["metrics"]["fit"]["objective"] - reference) / abs(reference)
    return max(gap, SUBOPTIMALITY_FLOOR)


def _n_iter_text(runs: list[dict]) -> str:
    values = []
    for run in runs:
        n_iter = run.get("attributes", {}).get("n_iter")
        if isinstance(n_iter, list):
            n_iter = max(n_iter) if n_iter else None
        if isinstance(n_iter, (int, float)):
            values.append(int(n_iter))
    if not values:
        return "?"
    return str(values[0]) if min(values) == max(values) else f"{min(values)} to {max(values)}"


def _points(record: BenchmarkRecord, best: dict[tuple, float]) -> tuple | None:
    """`(fit time, suboptimality, test loss, hover)`, medians over the runs."""
    runs = [run for run in record.runs if _has_objective(run)]
    if not runs:
        return None
    fit_s = median(run["time_ms"]["fit"] for run in runs) / 1000
    gap = median(suboptimality(record, run, best) for run in runs)
    test_loss = median(run["metrics"]["predict"]["mean loss"] for run in runs)
    tol = _tol(record)
    gap_text = f"≤ {SUBOPTIMALITY_FLOOR:.0e}" if gap <= SUBOPTIMALITY_FLOOR else f"{gap:.2e}"
    hover = "<br>".join([
        f"tol: {tol if tol == 'default' else f'{tol:.0e}'}",
        f"fit time: {fit_s:.3g}s (median of {len(runs)})",
        f"suboptimality: {gap_text}",
        f"test loss: {test_loss:.6g}",
        f"iterations: {_n_iter_text(runs)}",
    ])
    return fit_s, gap, test_loss, hover


def _shape_line(records: list[BenchmarkRecord]) -> str:
    objective = _objective(records[0])
    for record in records:
        for run in record.runs:
            fit = run.get("data_desc", {}).get("fit", {})
            if fit.get("samples") and fit.get("features"):
                line = f"Train: {fit['samples']:,} x {fit['features']}"
                if fit.get("n_classes"):
                    line += f", {fit['n_classes']} classes"
                return f"{line}. {LOSS_LABELS[objective['loss']]}, alpha = {objective['alpha']:g}"
    return ""


def _problem_cell_html(
    dataset: str, records: list[BenchmarkRecord], best: dict, colors: dict, dashes: dict
) -> str:
    gap_series: dict[str, dict] = {}
    loss_series: dict[str, dict] = {}
    ladders: dict[str, list] = {}
    for record in records:
        point = _points(record, best)
        if point is None:
            continue
        fit_s, gap, test_loss, hover = point
        label = _series_label(record)
        tol = _tol(record)
        if tol == "default":
            gap_series.setdefault(label, {}).setdefault("highlight", []).append((fit_s, gap, hover))
            loss_series.setdefault(label, {}).setdefault("highlight", []).append((fit_s, test_loss, hover))
        else:
            ladders.setdefault(label, []).append((-tol, fit_s, gap, test_loss, hover))
    for label, ladder in ladders.items():
        ladder.sort()
        gap_series.setdefault(label, {})["points"] = [(p[1], p[2], p[4]) for p in ladder]
        loss_series.setdefault(label, {})["points"] = [(p[1], p[3], p[4]) for p in ladder]
    if not gap_series:
        return ""
    order = sorted(gap_series, key=_label_sort_key)
    gap_series = {label: gap_series[label] for label in order}
    loss_series = {label: loss_series[label] for label in order}

    losses = [
        point[1] for data in loss_series.values()
        for point in data.get("points", []) + data.get("highlight", [])
    ]
    lowest = min(losses)
    highest = min(max(losses), lowest + TEST_LOSS_ZOOM * abs(lowest))
    padding = 0.05 * (highest - lowest or 1e-3 * abs(lowest))
    loss_range = [lowest - padding, highest + padding]
    gap_plot = ladder_plot_html(
        gap_series, colors=colors, line_dashes=dashes,
        y_title="suboptimality", y_log=True, height="360px",
    )
    loss_plot = ladder_plot_html(
        loss_series, colors=colors, line_dashes=dashes,
        y_title="test loss", y_range=loss_range, height="260px", show_legend=False,
    )
    subtitle = f'<div class="plot-subtitle">{escape(_shape_line(records))}</div>'
    return (
        f'<section class="plot-cell"><h3>{escape(dataset)}</h3>{subtitle}'
        f"{gap_plot}{loss_plot}</section>"
    )


def _software_tabs_html(tab_records: list[BenchmarkRecord]) -> str:
    cards = []
    for software_hash in sorted({record.software_hash for record in tab_records}):
        build = software_build_name(software_hash)
        summary = summarize_software_env(
            read_env("software", software_hash),
            Implementation(library="sklearn", device=None, data_library=None),
            software_hash=software_hash,
            extra_packages=["glum", "tabmat"] if build == GLM_ENV else (),
        )
        summary["name"] = build
        cards.append(SOFTWARE_TEMPLATE.render(**summary))
    return render_software_tabs(cards)


def _failure_reason(record: BenchmarkRecord) -> str:
    logs = (record.failed_case or {}).get("logs", {}) or {}
    stderr_lines = str(logs.get("stderr", "")).strip().splitlines()
    if stderr_lines:
        return stderr_lines[-1]
    return_code = (record.failed_case or {}).get("return_code")
    return f"return code {return_code}" if return_code is not None else "failed"


def _failed_cases_html(hw_records: list[BenchmarkRecord]) -> str:
    failed = [record for record in hw_records if record.failed_case is not None]
    if not failed:
        return ""
    key = lambda r: (_label_sort_key(_series_label(r)), _dataset(r), str(_tol(r)))
    items = "".join(
        f"<li>{escape(_series_label(record))} / {escape(_dataset(record))} "
        f"(tol {escape(str(_tol(record)))}, {len(record.runs)} runs done): "
        f"{escape(_failure_reason(record))}</li>"
        for record in sorted(failed, key=key)
    )
    return f'<section class="panel"><h3>Failed cases</h3><ul class="compact">{items}</ul></section>'


def _problem_sort_key(dataset: str) -> tuple:
    return (
        PROBLEM_ORDER.index(dataset) if dataset in PROBLEM_ORDER else len(PROBLEM_ORDER),
        dataset,
    )


def render_tab_page(tab_records: list[BenchmarkRecord], best: dict, colors: dict, dashes: dict) -> str:
    sections = [
        f'<div class="page-row">{DATE_RANGE_TEMPLATE.render(**date_range(tab_records))}</div>',
        f'<div class="page-row">{_software_tabs_html(tab_records)}</div>',
    ]
    failed_html = _failed_cases_html(tab_records)
    if failed_html:
        sections.append(f'<div class="page-row">{failed_html}</div>')

    by_dataset: dict[str, list[BenchmarkRecord]] = {}
    for record in tab_records:
        # A case that timed out keeps the runs it finished, on fewer splits
        # than the others, so its medians aren't comparable.
        if record.runs and record.failed_case is None:
            by_dataset.setdefault(_dataset(record), []).append(record)
    cells = [
        cell_html
        for dataset in sorted(by_dataset, key=_problem_sort_key)
        if (cell_html := _problem_cell_html(dataset, by_dataset[dataset], best, colors, dashes))
    ]
    if cells:
        grid = (
            '<section class="plot-grid" '
            'style="grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));">'
            + "".join(cells)
            + "</section>"
        )
        sections.append(f'<div class="page-row">{grid}</div>')
    return "".join(sections)


def _hardware_sort_index(hardware_hash: str) -> int:
    try:
        return list(HARDWARE_NAMES).index(hardware_hash)
    except ValueError:
        return len(HARDWARE_NAMES)


def generate(output_dir: Path) -> None:
    records = _dedup_latest(
        [
            record for record in read_benchmark_records()
            if matches_source_configs(record.case, SOURCE_CONFIGS)
        ]
    )
    best = best_objectives(records)
    tabs: dict[str, list[BenchmarkRecord]] = {}
    for record in records:
        tabs.setdefault(record.hardware_hash, []).append(record)

    labels = {_series_label(record) for record in records}
    colors = _series_colors(labels)
    dashes = _line_dashes(labels)
    pages = [
        (HARDWARE_NAMES.get(hardware_hash, hardware_hash),
         render_tab_page(tab_records, best, colors, dashes))
        for hardware_hash, tab_records in sorted(
            tabs.items(), key=lambda item: _hardware_sort_index(item[0])
        )
    ]
    html = BASE_TEMPLATE.render(
        title=TITLE,
        rows=[ABOUT_HTML, render_hardware_tabs(pages)],
    )
    output = output_dir / "linear_pareto.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
