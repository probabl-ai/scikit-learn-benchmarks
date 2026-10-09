"""Hover definitions for the technical terms of the dashboard prose.

Mark a term as ``<dfn>BLAS</dfn>``, or as ``<dfn term="smt">SMT sibling</dfn>``
when the displayed text is neither a key nor an alias of ``TERMS``.
``expand_terms`` (applied to every page row by ``BASE_TEMPLATE``) turns it into
a tooltip. An unknown term raises, so a typo fails the dashboard build instead
of shipping a word without its definition.
"""

from dataclasses import dataclass
from html import escape
import re


GLOSSARY_URL = "https://scikit-learn.org/stable/glossary.html#term-"


@dataclass(frozen=True)
class Term:
    definition: str
    aliases: tuple[str, ...] = ()
    # Anchor in the scikit-learn glossary, without the "term-" prefix.
    glossary: str | None = None
    # Documentation link for terms that aren't in the scikit-learn glossary.
    url: str | None = None


TERMS = {
    # Libraries and builds
    "blas": Term(
        "Basic Linear Algebra Subprograms: the library that runs the matrix and "
        "vector products of NumPy and SciPy. OpenBLAS and MKL are two "
        "implementations of it.",
    ),
    "openmp": Term(
        "Standard for running the loops of compiled code on several threads. "
        "scikit-learn's Cython code (HistGradientBoosting, KMeans, ...) uses it.",
        aliases=("openmp runtime", "openmp threads"),
    ),
    "openblas": Term(
        "Open-source BLAS implementation. The NumPy and SciPy wheels on PyPI "
        "each ship their own copy of it.",
    ),
    "mkl": Term(
        "Intel's Math Kernel Library (oneMKL): a BLAS implementation tuned for "
        "Intel CPUs, available as a conda-forge build option.",
        aliases=("mkl conda-forge build",),
    ),
    "libgomp": Term("GCC's OpenMP runtime, used by the PyPI scikit-learn wheels on Linux."),
    "libomp": Term("LLVM's OpenMP runtime, used by the PyPI scikit-learn wheels on macOS."),
    "llvm/intel openmp": Term(
        "LLVM's OpenMP runtime (libomp) and Intel's (libiomp5), which share "
        "most of their code.",
    ),
    "pypi": Term(
        "The Python Package Index, where pip installs packages from. Its "
        "scikit-learn wheels bundle their own OpenMP runtime, and NumPy and "
        "SciPy bundle OpenBLAS.",
        aliases=("pypi wheel",),
    ),
    "conda-forge": Term(
        "Community-maintained channel of conda packages. Its scikit-learn can "
        "be installed with different BLAS and OpenMP libraries.",
    ),
    "build": Term(
        "One packaging of the same scikit-learn version. Here, builds differ "
        "by their BLAS and OpenMP libraries.",
        aliases=("builds", "blas/openmp build", "blas/openmp builds"),
    ),
    "blas/openmp runtime": Term(
        "The BLAS and OpenMP libraries loaded at run time. They do the "
        "multithreaded number crunching for scikit-learn.",
    ),
    "software environment": Term(
        "One Pixi environment of this repository: a fixed set of package "
        "versions and builds.",
        aliases=("software environments", "environment", "pixi envs"),
    ),

    "onedal": Term(
        "oneAPI Data Analytics Library: Intel's C++ library that implements the "
        "algorithms of scikit-learn-intelex.",
    ),
    "onetbb": Term(
        "oneAPI Threading Building Blocks: Intel's C++ library for parallel "
        "tasks. oneDAL uses it to run on several threads, with its own thread "
        "pool, separate from OpenMP and BLAS.",
        url="https://uxlfoundation.github.io/oneTBB/",
    ),
    "array api": Term(
        "Standard API shared by array libraries (NumPy, PyTorch, CuPy, dpnp). "
        "With it, some scikit-learn estimators run directly on these "
        "libraries' arrays, including on GPU.",
    ),
    "pytorch": Term(
        "Deep learning library. scikit-learn can take its tensors as Array API "
        "inputs, on CPU or GPU.",
        url="https://pytorch.org/",
    ),
    "dpnp": Term(
        "Intel's NumPy-like array library, running on Intel GPUs and CPUs.",
        url="https://intelpython.github.io/dpnp/",
    ),
    "cupy": Term(
        "NumPy-like array library running on NVIDIA GPUs.",
        url="https://cupy.dev/",
    ),
    "joblib": Term(
        "The library scikit-learn uses to run tasks in parallel, in worker "
        "processes or threads.",
        glossary="joblib",
    ),
    "loky": Term(
        "joblib's default backend. It runs tasks in separate worker processes "
        "that are reused across calls.",
    ),
    "threading backend": Term(
        "joblib backend that runs tasks in threads of the same process. It only "
        "helps when the code releases the GIL, as tree building does.",
    ),
    # scikit-learn concepts
    "implementation": Term(
        "Code that runs a given estimator: stock scikit-learn, "
        "scikit-learn-intelex or an Array API backend.",
        aliases=("implementations",),
    ),
    "fit": Term(
        "Training a model on data.",
        aliases=(".fit()",),
        glossary="fit",
    ),
    "predict": Term(
        "Making predictions with a fitted model.",
        glossary="predict",
    ),
    "hyperparameters": Term(
        "Parameters set before fit that control how an estimator learns, such "
        "as max_depth or the regularization strength.",
        aliases=("hyperparameter", "hyper-parameter", "hyper-parameters"),
        glossary="hyperparameter",
    ),
    "candidates": Term(
        "Hyperparameter combinations tried by the search. Each one is fitted "
        "and scored on every cross-validation split.",
        aliases=("candidate",),
    ),
    "n_jobs": Term(
        "Parameter setting how many jobs scikit-learn runs in parallel with "
        "joblib. -1 means one per CPU.",
        glossary="n_jobs",
    ),
    "metrics": Term(
        "Scores of the fitted model on held-out data (ROC AUC, R2 score). They "
        "check that the compared implementations learn equivalent models.",
        glossary="evaluation-metric",
    ),
    "solver": Term(
        "The optimization algorithm used by fit, for instance lbfgs or "
        "newton-cholesky for LogisticRegression.",
    ),
    "preprocessing": Term(
        "Steps run before the model, such as encoding categorical columns and "
        "imputing missing values.",
    ),
    "fell back": Term(
        "scikit-learn-intelex ran the stock scikit-learn code, because it "
        "doesn't support this case's parameters or data.",
    ),
    # Algorithms
    "binning": Term(
        "Mapping each feature to a small number of integer bins (a few "
        "hundred at most) before building trees.",
        aliases=("bin features",),
    ),
    "histogram-based splits": Term(
        "Splits chosen among bin edges instead of among every distinct feature "
        "value. Faster, and usually as accurate.",
    ),
    "exact splits": Term("Splits chosen among every distinct value of the feature."),
    "max_bins": Term("Maximum number of bins per feature for histogram-based splits."),
    "histogram computation": Term(
        "For each tree node, summing the gradients of the samples in each bin "
        "of each feature.",
    ),
    "split finding": Term(
        "Scanning the histograms to find the best feature and threshold to "
        "split each node.",
    ),
    "max_leaf_nodes": Term(
        "Tree parameter capping the number of leaves. With it, trees grow "
        "best-first: the most promising node is split first.",
    ),
    "l-bfgs": Term(
        "Quasi-Newton optimization algorithm, behind LogisticRegression's "
        "default lbfgs solver. L-BFGS-B is its variant with bounds.",
        aliases=("l-bfgs-b",),
    ),
    "svd": Term(
        "Singular value decomposition: a stable but costly way to solve Ridge's "
        "least squares problem.",
    ),
    "cholesky": Term(
        "Matrix factorization used to solve Ridge's normal equations. Faster "
        "than SVD.",
    ),
    "vectorization": Term("Using CPU instructions that process several numbers at once (SIMD)."),
    # Hardware and parallelism
    "threads": Term(
        "Units of execution that share the memory of their process. Each one "
        "can run on its own CPU core.",
        aliases=("thread", "threading"),
    ),
    "worker processes": Term(
        "Separate Python processes running tasks in parallel. Unlike threads, "
        "they don't share memory.",
        aliases=("worker process", "outer workers"),
    ),
    "logical cpu": Term(
        "What the operating system sees as one CPU. With SMT, each physical "
        "core shows up as two logical CPUs.",
        aliases=("logical cpus", "logical cores"),
    ),
    "physical core": Term(
        "One hardware core of the CPU. With SMT, it shows up as two logical CPUs.",
        aliases=("physical cores", "cpu cores", "cores"),
    ),
    "smt": Term(
        "Simultaneous multithreading, called hyper-threading by Intel: one "
        "physical core runs two threads at once by sharing its execution units.",
        aliases=("hyper-threading",),
    ),
    "smt sibling": Term("The other logical CPU of the same physical core."),
    "e-cores": Term(
        "Efficiency cores: smaller and slower cores that recent Intel laptop "
        "CPUs have next to their fast performance cores.",
    ),
    "thread affinity": Term(
        "Pinning threads to given CPU cores, so the operating system doesn't "
        "move them around.",
    ),
    "omp_num_threads": Term("Environment variable setting how many threads OpenMP code uses."),
    "active wait": Term(
        "Idle OpenMP threads spin on the CPU for a while before sleeping, so "
        "they start the next parallel loop faster, at the cost of CPU time.",
    ),
    "oversubscription": Term(
        "More busy threads than CPU cores. They compete for the cores and the "
        "operating system keeps switching between them, which slows "
        "everything down.",
    ),
    "cpu load": Term(
        "Mean share of all the machine's CPUs that were busy during the run. "
        "100% means all cores busy.",
    ),
    "memory bound": Term(
        "Limited by how fast data moves from memory rather than by "
        "computation, so more cores don't help much.",
    ),
    "wall-clock time": Term(
        "Elapsed real time, as opposed to CPU time, which sums the time of "
        "all threads.",
    ),
    # Reading the plots
    "speed-up": Term(
        "Baseline time divided by the compared time: 2x is twice as fast as "
        "the baseline, 0.5x twice as slow. In the detailed results, a "
        "noisy speed-up is shown as ~1.5x (5-15% spread over the repeats) "
        "or as a range like 1.2-1.9x (more than 15%).",
        aliases=("speed-ups", "speedup", "fit speedup", "predict speedup"),
    ),
    "baseline": Term("The reference that speed-ups are computed against."),

    "pareto front": Term(
        "The points that no other point beats on both axes: nothing is both "
        "faster and more accurate. Every other point is a worse trade-off.",
        aliases=("pareto fronts",),
    ),
    "roc auc": Term(
        "Probability that the model ranks a random positive sample above a "
        "random negative one. 0.5 is random guessing, 1 is perfect. For "
        "several classes, averaged over one-vs-rest problems.",
    ),
    "r2": Term(
        "Coefficient of determination: 1 minus the squared error of the "
        "predictions divided by the variance of the target. 1 is perfect, 0 "
        "is as good as always predicting the mean.",
    ),
    "perfect-scaling line": Term(
        "The time the fit would take if n times more cores made it n times "
        "faster.",
    ),
}

_LOOKUP = {}
for _key, _term in TERMS.items():
    for _name in (_key, *_term.aliases):
        assert _name not in _LOOKUP, f"duplicate glossary name {_name!r}"
        _LOOKUP[_name] = _term

_DFN = re.compile(r'<dfn(?: term="([^"]+)")?>(.*?)</dfn>', re.DOTALL)
_TAG = re.compile(r"<[^>]+>")


def _render(match: re.Match) -> str:
    text = match[2]
    name = match[1] or " ".join(_TAG.sub("", text).split()).lower()
    try:
        term = _LOOKUP[name]
    except KeyError:
        raise KeyError(f"no glossary entry for <dfn> term {name!r}") from None
    more = ""
    if term.glossary is not None:
        more = (
            f' <a href="{GLOSSARY_URL}{term.glossary}" target="_blank" rel="noopener">'
            "More in the scikit-learn glossary</a>"
        )
    elif term.url is not None:
        more = f' <a href="{escape(term.url)}" target="_blank" rel="noopener">More</a>'
    return (
        f'<span class="term" tabindex="0">{text}'
        f'<span class="term-tip" role="tooltip">{escape(term.definition)}{more}</span>'
        "</span>"
    )


def expand_terms(html: str) -> str:
    if "<dfn" not in html:
        return html
    return _DFN.sub(_render, html)
