from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import json
import re
from statistics import median

from .utils import stable_json, without_keys


RESULT_FILE_RE = re.compile(r"^(?:.+_)?(\d{8}T\d{6}(?:\d{6})?Z)\.json$")


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

    hardware_hash: str  # ...
    software: str  # pixi env name
    software_hash: str  # ...
    method: str  # fit/predict
    timestamp_recorded: datetime  # based on filename
    case: dict  # case without the "bench" key
    # results:
    metrics: dict[str, dict]
    times: list[float]  # in ms
    data_desc: dict
    attributes: dict = field(default_factory=dict)
    logs: dict = field(default_factory=dict)
    # TODO: collect attributes? e.g. solver, tree-structure, ...
    # first we need to record meaningful attributes in results
    # => any int/str, arrays with only a few elements should be transofrmed to list

    @property
    def implementation(self) -> Implementation:
        implementation = self.case["implementation"]
        return Implementation(
            library=implementation.get("library"),
            device=implementation.get("device"),
            data_library=implementation.get("data_library"),
        )

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
        return self.implementation.library == "sklearnex" and self.category == "tree-based"
    
    @property
    def minimal_match_key(self) -> str:
        """
        If two results don't share the same minimal_match_key, it will
        never makes sense to compare them
        """
        case = without_keys(
            self.case,
            excluded_names={"implementation", "max_bins"},
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


def _parse_result_timestamp(path: Path) -> datetime:
    match = RESULT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Result filename '{path}' does not end with _<datetime>.json")
    timestamp = match.group(1)
    date_format = "%Y%m%dT%H%M%S%fZ" if len(timestamp) > 16 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(timestamp, date_format).replace(tzinfo=timezone.utc)


def _split_metrics_attributes(metrics: dict) -> tuple[dict, dict]:
    metrics = dict(metrics)
    attributes = {}
    if "iterations" in metrics:
        attributes["iterations"] = metrics.pop("iterations")
    return metrics, attributes


def _split_method_metrics_attributes(
    method_metrics: dict[str, dict]
) -> tuple[dict[str, dict], dict]:
    all_metrics = {}
    all_attributes = {}
    for method, metrics in method_metrics.items():
        clean_metrics, attributes = _split_metrics_attributes(metrics)
        all_metrics[method] = clean_metrics
        if method == "fit":
            all_attributes = attributes
    return all_metrics, all_attributes


def read_all_results(path=None) -> list[Result]:
    """
    path defaults to ./results/

    Read all available results and de-duplicates redundant results, i.e.
    results with the same `full_match_key` (keep the latest)
    """
    results_root = Path(path) if path is not None else Path("results")
    results: list[Result] = []

    for result_path in sorted(results_root.glob("*.json")):
        if not RESULT_FILE_RE.match(result_path.name):
            continue

        with open(result_path, "r") as f:
            result_file = json.load(f)

        hardware_hash = result_file["hardware_hash"]
        software_hash = result_file["software_hash"]
        timestamp = _parse_result_timestamp(result_path)

        for bench_case in result_file.get("bench_cases", []):
            case = without_keys(bench_case.get("case", {}), excluded_names={"bench"})
            metrics, attributes = _split_method_metrics_attributes(
                bench_case.get("metrics", {})
            )
            for method, times in bench_case.get("time[ms]", {}).items():
                if not isinstance(times, list) or len(times) == 0:
                    continue
                results.append(
                    Result(
                        hardware_hash=hardware_hash,
                        software=software_hash,
                        software_hash=software_hash,
                        method=method,
                        timestamp_recorded=timestamp,
                        case=case,
                        metrics=metrics,
                        times=[float(time) for time in times],
                        data_desc=bench_case.get("data_desc", {}).get(method, {}),
                        attributes=attributes,
                        logs=bench_case.get("logs", {}),
                    )
                )

    latest_by_key: dict[str, Result] = {}
    for result in results:
        current = latest_by_key.get(result.full_match_key)
        if current is None or result.timestamp_recorded > current.timestamp_recorded:
            latest_by_key[result.full_match_key] = result
    return list(latest_by_key.values())


@dataclass(frozen=True)
class MatchWarning:
    icon: str
    message: str
    short_message: str | None = None


@dataclass(frozen=True)
class Match:
    base_result: Result
    matched_result: Result
    warnings: list[MatchWarning]

    @staticmethod
    def _comparable_metric_values(metrics: dict) -> dict:
        metrics = dict(metrics)

        # for regression, we pop RMSE, as it's redundant with R2 but
        # depends on the variance of y, which makes the comparison harder to do
        if "R2" in metrics and "RMSE" in metrics:
            metrics.pop("RMSE")
        return metrics

    def _comparable_metrics(self) -> tuple[dict, dict]:
        base_metrics = {
            method: self._comparable_metric_values(metrics)
            for method, metrics in self.base_result.metrics.items()
        }
        matched_metrics = {
            method: self._comparable_metric_values(metrics)
            for method, metrics in self.matched_result.metrics.items()
        }

        if set(base_metrics) != set(matched_metrics):
            raise ValueError("Sets of metric methods differ")

        return base_metrics, matched_metrics

    @property
    def metrics_differences(self) -> list[str]:
        if self.base_result.method != "fit":
            return []

        differences = []
        base_metrics, matched_metrics = (
            self._comparable_metrics()
        )

        for method, base_method_metrics in base_metrics.items():
            matched_method_metrics = matched_metrics[method]
            if set(base_method_metrics) != set(matched_method_metrics):
                raise ValueError(f"Sets of {method} metrics differ")

            for k, v in base_method_metrics.items():
                if not math.isclose(v, matched_method_metrics[k], rel_tol=0.01, abs_tol=0.01):
                    differences.append(
                        f"{method}.{k}: base={v:.3g}, target={matched_method_metrics[k]:.3g}"
                    )
        return differences

    @property
    def metrics_match(self) -> bool:
        return not self.metrics_differences

    @property
    def speedup(self) -> float:
        matched_time = median(self.matched_result.times)
        if matched_time == 0:
            return math.inf
        return median(self.base_result.times) / matched_time


MAX_BINS_WARNING = MatchWarning(
    icon="🧺",
    short_message="histogram-based splits",
    message="Scikit-learn intelex uses binning & histogram-based splits while scikit-learn doesn't",
)

CPU_FALLBACK_WARNING = MatchWarning(
    icon="↩",
    short_message="CPU fallback",
    message="Some operations fell back to CPU according to benchmark logs",
)


def _logs_text(result: Result) -> str:
    return "\n".join(str(result.logs.get(stream, "")) for stream in ("stdout", "stderr"))


def has_cpu_fallback_warning(result: Result) -> bool:
    text = _logs_text(result).lower()
    return (
        "fallback from xpu to cpu" in text
        or ("aten op fallback" in text and "cpu" in text)
    )


def append_cpu_fallback_warning(result: Result, warnings: list):
    if result.implementation.device not in {"cuda", "gpu", "xpu"}:
        return
    if has_cpu_fallback_warning(result):
        warnings.append(CPU_FALLBACK_WARNING)


def append_max_bins_warning(sklearn_res: Result, sklearnex_res: Result, warnings: list):
    assert sklearn_res.implementation.library == "sklearn"
    assert sklearnex_res.implementation.library == "sklearnex"

    estimator_params = sklearnex_res.case.get("algorithm", {}).get("estimator_params", {})
    max_bins = estimator_params.get("max_bins", 255)
    n_samples = (
        sklearnex_res.case.get("data", {})
        .get("generation_kwargs", {})
        .get("n_samples")
    )
    if n_samples is None:
        n_samples = sklearnex_res.data_desc.get("samples")

    if n_samples is not None and max_bins < n_samples:
        warnings.append(MAX_BINS_WARNING)


def append_iterations_warning(base_res: Result, candidate: Result, warnings: list):
    base_iterations = base_res.attributes.get("iterations")
    candidate_iterations = candidate.attributes.get("iterations")
    if base_iterations == candidate_iterations:
        return
    if base_iterations is None and candidate_iterations is None:
        return

    warnings.append(
        MatchWarning(
            icon="🔁",
            short_message=f"({base_iterations} vs {candidate_iterations})",
            message="Number of iteration differs: this might mean algorithms are different",
        )
    )


def find_matches(
    base_results: list[Result],
    results_to_match: list[Result],
    match_function,
    match_key=None,
) -> list[Match]:
    # should assert all result in results_to_match match at most one base_results
    if match_key is None:
        match_key = lambda result: result.minimal_match_key
    base_by_minimal_key: dict[str, list[Result]] = {}
    for base_result in base_results:
        base_by_minimal_key.setdefault(match_key(base_result), []).append(base_result)

    matches: list[Match] = []
    for result in results_to_match:
        candidate_matches = []
        for base_result in base_by_minimal_key.get(match_key(result), []):
            match_result = match_function(base_result, result)
            if isinstance(match_result, tuple):
                does_match, warnings = match_result
            else:
                does_match, warnings = bool(match_result), []
            if does_match:
                candidate_matches.append(Match(base_result, result, list(warnings)))

        if len(candidate_matches) > 1:
            raise ValueError(
                "A result matched several base results: "
                f"{result.implementation.short_name} {result.method} "
                f"{result.case.get('algorithm', {}).get('estimator', 'unknown')}"
            )
        if candidate_matches:
            matches.append(candidate_matches[0])

    return matches


def date_range(results: list[Result]) -> dict:
    if not results:
        return {"label": "No results"}
    start = min(result.timestamp_recorded for result in results)
    end = max(result.timestamp_recorded for result in results)
    if start.date() == end.date():
        date_range = start.strftime("%Y-%m-%d")
    else:
        date_range = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    return {"label": f"{len(results)} benchmark records from {date_range}"}
