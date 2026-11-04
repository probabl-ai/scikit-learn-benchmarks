# Drop the shared sklearn-dev env's editable hooks (scikit-learn and sklbench),
# so this venv's scikit-learn and the sklbench on PYTHONPATH are imported.
import sys
sys.meta_path[:] = [
    f for f in sys.meta_path
    if type(f).__module__ != "_scikit_learn_editable_loader"
    and not type(f).__module__.startswith("__editable___scikit_learn_benchmarks")
]
