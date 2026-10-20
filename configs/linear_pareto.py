"""
Fit time vs suboptimality of L2-penalized linear models, scikit-learn vs glum,
on real datasets: logistic regression (binary and multiclass), Poisson and
Gamma GLMs.

For a fixed problem (dataset, loss and L2 penalty `alpha`), every solver that
converges finds the same model, so what differs is how fast it gets close to
the optimum. Each solver fits the problem at a ladder of tolerances, from
loose to tight, and at its default tolerance. The runner records the
training objective in one convention for every library (see
`sklbench/runners/estimator/objective.py`), and the dashboard plots the fit
time against the gap to the best objective found by any solver.

Each `alpha` is about the one with the best test loss on the dataset, from a
sweep over powers of 10.

In the `glm` env, scikit-learn and glum run. In the other scikit-learn envs
(`sklearn-dev`, `sklearn-pypi`, ...), only scikit-learn runs, to place a
scikit-learn branch or build on the same plots.

Every library uses its default number of threads.
"""
import math
import os

from _utils.implementations import implementations_for_pixi_env

BENCH = {"n_runs": 3, "py_spy_profiling": False, "time_limit": 900}

# loss: dataset -> alpha
PROBLEMS = {
    "binomial": {
        # 45k x 16, low-cardinality categoricals.
        "bank_marketing": 1e-6,
        # 73k x 32, mixed numeric and categorical.
        "kick": 1e-5,
        # 50k x 212, many missing values: ~2,200 features after splines.
        "kddcup09_churn": 1e-4,
    },
    "multinomial": {
        # 20k x 16, 26 classes.
        "letters": 1e-5,
        # 581k x 12, 7 classes. The test loss is slightly lower at 1e-6
        # (0.5925 vs 0.5959), but lbfgs then needs ~3,000 iterations.
        "covtype": 1e-5,
    },
    "poisson": {
        # 678k x 10, claim counts.
        "fremtpl2_freq": 1e-6,
    },
    "gamma": {
        # 25k x 9, claim amounts.
        "fremtpl2_sev": 1e-1,
    },
}

TOLS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
# sag and saga already take ~1 to 2 min per fit at 1e-6 on bank_marketing
# (36k rows), and each tighter tolerance multiplies that.
MIN_TOL = {"sag": 1e-6, "saga": 1e-6}
# Large enough that the tolerance is what stops every solver. The default
# tolerance cases keep the default `max_iter` too.
MAX_ITER = 10_000

ESTIMATORS = {
    "sklearn": {
        "binomial": "LogisticRegression",
        "multinomial": "LogisticRegression",
        "poisson": "PoissonRegressor",
        "gamma": "GammaRegressor",
    },
    "glum": {
        "binomial": "GeneralizedLinearClassifier",
        "poisson": "GeneralizedLinearRegressor",
        "gamma": "GeneralizedLinearRegressor",
    },
}

# liblinear is left out: it penalizes the intercept, so it solves another
# problem. glum's lbfgs solver fails with scipy>=1.18 (it passes the removed
# `iprint` argument to `fmin_l_bfgs_b`), and trust-constr is for constrained
# problems.
SOLVERS = {
    "sklearn": {
        "binomial": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        "multinomial": ["lbfgs", "newton-cg", "newton-cholesky", "sag", "saga"],
        "poisson": ["lbfgs", "newton-cholesky"],
        "gamma": ["lbfgs", "newton-cholesky"],
    },
    "glum": {
        "binomial": ["irls-ls", "irls-cd"],
        "poisson": ["irls-ls", "irls-cd"],
        "gamma": ["irls-ls", "irls-cd"],
    },
}


def _sklearn(loss: str, alpha: float, n_train: int, solver: str, tol: float | None) -> dict:
    if loss in ("binomial", "multinomial"):
        params = {"C": 1 / (alpha * n_train)}
    else:
        params = {"alpha": alpha}
    params["solver"] = solver
    if tol is not None:
        params |= {"tol": tol, "max_iter": MAX_ITER}
    return params


def _glum(loss: str, alpha: float, n_train: int, solver: str, tol: float | None) -> dict:
    params = {"family": loss, "alpha": alpha, "solver": solver}
    if tol is not None:
        params |= {"gradient_tol": tol, "max_iter": MAX_ITER}
    return params


ESTIMATOR_PARAMS = {"sklearn": _sklearn, "glum": _glum}

# Training samples after the loaders' default 80/20 split, for
# LogisticRegression's `C`.
N_TRAIN = {
    "bank_marketing": 36_168,
    "kick": 58_386,
    "kddcup09_churn": 40_000,
    "letters": 16_000,
    "covtype": 464_809,
    "fremtpl2_freq": 542_410,
    "fremtpl2_sev": 19_955,
}
# Features after the linear preprocessing and classes (1 for the other
# losses), only to balance the shards.
DATASET_SIZES = {
    dataset: (N_TRAIN[dataset], n_features, n_classes)
    for dataset, n_features, n_classes in [
        ("bank_marketing", 126, 1),
        ("kick", 296, 1),
        ("kddcup09_churn", 2_199, 1),
        ("letters", 176, 26),
        ("covtype", 138, 7),
        ("fremtpl2_freq", 98, 1),
        ("fremtpl2_sev", 87, 1),
    ]
}


def _cases(implem: dict) -> list[dict]:
    library = implem["library"]
    cases = []
    for loss, datasets in PROBLEMS.items():
        if loss not in ESTIMATORS[library]:
            continue
        task = "classification" if loss in ("binomial", "multinomial") else "regression"
        for dataset, alpha in datasets.items():
            for solver in SOLVERS[library][loss]:
                tols = [tol for tol in TOLS if tol >= MIN_TOL.get(solver, 0)]
                for tol in [None, *tols]:
                    params = ESTIMATOR_PARAMS[library](loss, alpha, N_TRAIN[dataset], solver, tol)
                    cases.append({
                        "bench": BENCH,
                        "implementation": implem,
                        "metadata": {
                            "task": task,
                            "solver": solver,
                            "tol": "default" if tol is None else tol,
                        },
                        "algorithm": {
                            "estimator": ESTIMATORS[library][loss],
                            "estimator_params": params,
                            "objective": {"loss": loss, "alpha": alpha},
                        },
                        "data": {"dataset": dataset, "preprocessing_kind": "linear"},
                    })
    return cases


def _cost(case: dict) -> float:
    """Rough relative cost of a case, only to balance the shards."""
    n_rows, n_features, n_classes = DATASET_SIZES[case["data"]["dataset"]]
    tol = case["metadata"]["tol"]
    n_digits = 4 if tol == "default" else -math.log10(tol)
    return n_rows * n_features * n_classes * (1 + n_digits)


def _shard(cases: list[dict], index: int, count: int) -> list[dict]:
    """Longest-processing-time-first assignment: each case, most expensive
    first, goes to the least loaded shard."""
    loads = [0.0] * count
    assignment = [[] for _ in range(count)]
    for position in sorted(range(len(cases)), key=lambda i: (-_cost(cases[i]), i)):
        target = min(range(count), key=lambda j: (loads[j], j))
        loads[target] += _cost(cases[position])
        assignment[target].append(position)
    return [cases[i] for i in sorted(assignment[index])]


def generate_cases() -> list[dict]:
    cases = []
    for implem in implementations_for_pixi_env():
        # Skips sklearnex and the Array API implementations.
        if implem.get("device") is None and implem["library"] in ESTIMATOR_PARAMS:
            cases += _cases(implem)
    # LINEAR_PARETO_SHARD="i/n" keeps the i-th of n shards (from 0), to run
    # the shards at once, e.g. one per NUMA node. It filters the cases
    # without changing them.
    shard = os.environ.get("LINEAR_PARETO_SHARD")
    if shard is not None:
        index, count = (int(x) for x in shard.split("/"))
        cases = _shard(cases, index, count)
    return cases
