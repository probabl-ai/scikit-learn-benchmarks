"""What a benchmark repeat measures, and on which data. Timings are only comparable
if these hold, and a dashboard can't tell when they don't."""

import numpy as np
import pytest

from sklbench.config import EstimatorCase
from sklbench.runners.datasets import load_raw_data, preprocess_data
from sklbench.runners.datasets.preprocessing import split_data
from sklbench.runners.estimator import __main__ as runner


GENERATION = {"n_samples": 300, "n_features": 4}


def _case(implementation=None, generation_kwargs=GENERATION):
    return EstimatorCase(
        algorithm={"estimator": "Ridge"},
        data={"source": "make_regression", "generation_kwargs": generation_kwargs},
        implementation=implementation or {"library": "sklearn"},
    )


def _subsets(case, repeat=0):
    raw, description = load_raw_data(case, random_state=repeat)
    (x_train, x_test, y_train, y_test), _ = preprocess_data(case, raw, description, random_state=repeat)
    return x_train, x_test, y_train, y_test


def test_times_only_cover_what_they_are_named_after(monkeypatch):
    """With a clock that only moves inside each step, including the quality metrics
    that call `predict` again after it was timed."""
    clock = [0.0]

    class SlowEstimator:
        def fit(self, x, y):
            clock[0] += 7
            return self

        def predict(self, x):
            clock[0] += 3
            return np.zeros(len(x))

    real_preprocess = runner.preprocess_data

    def slow_preprocess(*args, **kwargs):
        clock[0] += 5
        return real_preprocess(*args, **kwargs)

    monkeypatch.setattr(runner, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "preprocess_data", slow_preprocess)
    case = _case()
    raw, description = load_raw_data(case, random_state=0)

    row = runner.run_case_once(case, SlowEstimator(), raw, description, data_random_state=0)

    assert row["time_ms"] == {"preprocessing": 5000, "fit": 7000, "predict": 3000}


def test_every_implementation_gets_the_same_data():
    sklearn = _subsets(_case())
    sklearnex = _subsets(_case({"library": "sklearnex", "device": "cpu"}))

    for subset, other in zip(sklearn, sklearnex):
        np.testing.assert_array_equal(subset, other)


def test_repeats_differ_but_are_reproducible():
    first = _subsets(_case(), repeat=0)

    np.testing.assert_array_equal(first[0], _subsets(_case(), repeat=0)[0])
    assert not np.array_equal(first[0], _subsets(_case(), repeat=1)[0])


def test_each_repeat_gets_its_own_seed_unless_the_case_sets_one():
    case = _case()
    assert {runner.data_random_state_for_repeat(case, repeat) for repeat in range(3)} == {0, 1, 2}
    pinned = _case(generation_kwargs={**GENERATION, "random_state": 7})
    assert {runner.data_random_state_for_repeat(pinned, repeat) for repeat in range(3)} == {7}

    class Model:
        def __init__(self, random_state=None):
            pass

    seeds = {runner.estimator_params_for_repeat(Model, {}, repeat)["random_state"] for repeat in range(3)}
    assert seeds == {0, 1, 2}
    assert runner.estimator_params_for_repeat(Model, {"random_state": 7}, 2) == {"random_state": 7}


@pytest.mark.parametrize("split_kwargs", [None, {"test_size": 0.3}])
def test_train_and_test_partition_the_data(split_kwargs):
    rows = np.arange(100)
    data = {"x": rows.reshape(-1, 1), "y": rows}

    split = split_data(data, split_kwargs=split_kwargs, default_split=None, random_state=1)

    train, test = set(split["x_train"].ravel()), set(split["x_test"].ravel())
    assert not train & test
    assert train | test == set(rows)
    # Features and target stay aligned through the split.
    np.testing.assert_array_equal(split["x_train"].ravel(), split["y_train"])


def test_synthetic_train_and_test_share_no_rows():
    x_train, x_test, _, _ = _subsets(_case())

    assert not {tuple(row) for row in x_train} & {tuple(row) for row in x_test}
