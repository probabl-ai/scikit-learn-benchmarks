# WIP!!!!

from dataclasses import dataclass

from .utils import stable_json, without_keys


@dataclass
class Implementation:
    library: str
    device: str | None
    data_library: str | None

    @property
    def short_name(self):
        if self.device in (None, "default"):
            return self.library
        if self.library == "sklearn":
            if self.data_library is not None:
                return f"{self.library}-{self.data_library}-{self.device}"
        return f"{self.library}-{self.device}"
    

@dataclass
class Result:
    # No support for functions for now.

    hardware: str  # sklearn
    hardware_hash: str  # ...
    software: str  # pixi env name
    software_hash: str  # ...
    method: str  # fit/predict
    timestamp_recorded: int | float  # based on filename
    case: dict  # case without the "bench" key
    # results:
    metrics: dict
    times: list[float]  # in ms
    data_desc: dict
    attributes: dict = {}
    # TODO: collect attributes? e.g. solver, tree-structure, ...
    # first we need to record meaningful attributes in results
    # => any int/str, arrays with only a few elements should be transofrmed to list

    @property
    def implementation(self) -> Implementation:
        implementation = self.case["implementation"]
        library = implementation.get("library")
        device = implementation.get("device")
        if device in (None, "default"):
            return library
        if library == "sklearn":
            data_library = implementation.get("data_library")
            if data_library is not None:
                return f"{library}-{data_library}-{device}"
        return f"{library}-{device}"

    @property
    def category(self) -> str:
        algo = self.case["algorithm"]["estimator"]
        if "Forest" in algo or "Tree" in algo:
            return "tree-based"
        elif algo == "KMeans" or algo == "DBSCAN":
            return "clustering"
        else:
            return "linear"

    @property
    def is_sklearnex_tree(self) -> bool:
        return False # TODO
    
    @property
    def minimal_match_key(self) -> str:
        """
        If two results don't share the same minimal_match_key, it will
        never makes sense to compare them
        """
        case = without_keys(
            self.case,
            excluded_names=("implementation", "max_bins"),
        )
        case['method'] = self.method
        return stable_json(case)

    @property
    def full_match_key(self) -> str:
        """
        If two results share the same full_match_key, it's not useful to include
        both in a given plot
        """
        return stable_json({
            **self.case,
            "method": self.method,
            "hardware": self.hardware_hash,
            "software": self.software_hash
        })


def read_all_results(path=None) -> list[Result]:
    """
    path defaults to ./results/

    Read all available results and de-duplicates redundant results, i.e.
    results with the same `full_match_key` (keep the latest)
    """
    pass


class MatchWarning:
    icon: str
    message: str


class Match:
    base_result: Result
    matched_result: Result
    warnings: list[MatchWarning]

    @property
    def metrics_match(self) -> bool:
        # TODO: we should actually store both fit & predict metrics in a Result
        # and compare both here
        base_metrics = self.base_result.metrics
        matched_metrics = self.matched_result.metrics
        if set(base_metrics) != set(matched_metrics):
            raise ValueError()

        # for regression, we pop RMSE, as it's redundant with R2 but
        # depends on the variance of y, which makes the comparison harder to do
        if "R2" in base_metrics and "RMSE" in base_metrics:
            base_metrics = {**base_metrics}
            base_metrics.pop("RMSE")

        # maybe improve? e.g. for ROC AUC or R2,
        # 0.991 vs 0.999 is quite different I'd say, but we consider
        # them the same

        for k, v in base_metrics.items():
            if abs(v - matched_metrics[k]) > 0.01:
                return False
        
        return True

    @property
    def speedup(self) -> float:
        pass


MAX_BINS_WARNING = MatchWarning(
    icon="🧺",
    message="Scikit-learn intelex uses binning & histogram-based splits while scikit-learn doesn't",
)

def append_max_bins_warning(sklearn_res: Result, sklearnex_res: Result, warnings: list):
    assert sklearn_res.implementation.library == "sklearn"
    assert sklearnex_res.implementation.library == "sklearnex"

    max_bins = sklearnex_res.case['algorithm_params'].pop('max_bins', 255)
    if max_bins < sklearnex_res.data_desc['n_samples']:
        warnings.append(MAX_BINS_WARNING)


def find_matches(
    base_results: list[Result],
    results_to_match: list[Result],
    match_function
) -> list[Match]:
    # should assert all result in results_to_match match at most one base_results
    pass


def date_range(results: list[Result]) -> dict:
    pass

