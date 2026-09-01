"""
Time just the preprocessing step (not dataset loading) for every distinct
dataset/preprocessing_kind pair used in `configs/real_datasets.py`.

Cases with `preprocessing_kind=None` (the clustering/KMeans cases, plus a
few linear/tree cases on already-well-conditioned datasets like `susy`) have
nothing to time and are skipped.
"""

import os
import time

import pandas as pd

from sklbench.config import load_cases_from_script
from sklbench.runners.datasets.loaders import dataset_loading_functions
from sklbench.runners.datasets.loading import load_from_cache_or_compute
from sklbench.runners.datasets.preprocessing import split_and_preprocess_data

CONFIG_PATH = "configs/real_datasets.py"


def freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def main():
    cases = load_cases_from_script(CONFIG_PATH)

    # dedupe: several estimators (e.g. Ridge and RandomForestRegressor) share
    # the same dataset/preprocessing_kind/preprocessing_kwargs combination.
    pairs = {}
    for case in cases:
        data = case.data
        if data.preprocessing_kind is None:
            continue
        key = (
            data.dataset,
            data.preprocessing_kind,
            freeze(data.split_kwargs),
            freeze(data.preprocessing_kwargs),
        )
        pairs.setdefault(key, data)

    data_cache = os.environ.get("SKLBENCH_DATA_CACHE", "data_cache")
    raw_data_cache = os.path.join(data_cache, "raw")

    for key, data_params in sorted(pairs.items(), key=lambda kv: kv[0][:2]):
        dataset, preprocessing_kind, _split, _kwargs = key

        raw_data, data_desc = load_from_cache_or_compute(
            dataset_loading_functions[dataset],
            data_name=dataset,
            data_cache=data_cache,
            raw_data_cache=raw_data_cache,
        )
        if isinstance(raw_data["x"], pd.DataFrame):
            object_columns = raw_data["x"].select_dtypes(include=["object"]).columns.tolist()
            assert not object_columns, (
                f"{dataset}: categorical columns must use `category` dtype, "
                f"found `object`-dtype columns instead: {object_columns}"
            )
        preprocessing_defaults = data_desc.get("preprocessing_defaults", {}).get(
            preprocessing_kind, {}
        )
        preprocessing_kwargs = preprocessing_defaults | data_params.preprocessing_kwargs

        start = time.perf_counter()
        split_and_preprocess_data(
            raw_data,
            split_kwargs=data_params.split_kwargs,
            default_split=data_desc.get("default_split"),
            preprocessing_kind=preprocessing_kind,
            preprocessing_kwargs=preprocessing_kwargs,
        )
        elapsed = time.perf_counter() - start

        kwargs_note = f" {preprocessing_kwargs}" if preprocessing_kwargs else ""
        print(f"{dataset:26s} {preprocessing_kind:8s}{kwargs_note}: {elapsed:.3f}s")

        if preprocessing_kwargs.get("nystroem") is not None:
            no_nystroem_kwargs = preprocessing_kwargs | {"nystroem": None}

            start = time.perf_counter()
            split_and_preprocess_data(
                raw_data,
                split_kwargs=data_params.split_kwargs,
                default_split=data_desc.get("default_split"),
                preprocessing_kind=preprocessing_kind,
                preprocessing_kwargs=no_nystroem_kwargs,
            )
            elapsed_no_nystroem = time.perf_counter() - start

            print(
                f"{dataset:26s} {preprocessing_kind:8s} (no nystroem): "
                f"{elapsed_no_nystroem:.3f}s"
            )


if __name__ == "__main__":
    main()
