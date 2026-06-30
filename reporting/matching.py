from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import json
import re
from statistics import mean, median, stdev
from typing import Any

from .utils import stable_json, without_keys


RESULT_FILE_RE = re.compile(r"^(?:.+_)?(\d{8}T\d{6}(?:\d{6})?Z)\.json$")
METRIC_ABS_TOLERANCE_FLOOR = 0.01
METRIC_REL_TOLERANCE_FLOOR = 0.01
METRIC_STD_TOLERANCE_FACTOR = 3  # TODO: should be based on number of runs:
# 3 std tol over the mean of e.g. 7 runs is quite a lot
# 2 should be enough for 7 runs; 2.5 for1.959963984540 3 runs; 3 for 2 runs
# this risks hiding bad matches


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
class BenchmarkRecord:
    hardware_hash: str
    software_hash: str
    timestamp_recorded: datetime
    case: dict
    runs: list[dict]

    @property
    def implementation(self) -> Implementation:
        implementation = self.case["implementation"]
        return Implementation(
            library=implementation.get("library"),
            device=implementation.get("device"),
            data_library=implementation.get("data_library"),
        )


@dataclass
class MethodResult:
    # No support for functions for now.

    hardware_hash: str
    software: str  # pixi env name
    software_hash: str
    method: str  # fit/predict
    timestamp_recorded: datetime
    case: dict  # case without the "bench" key
    times: list[float]  # in ms
    data_desc: dict
    metrics: dict[str, dict[str, list[Any]]]
    attributes: dict = field(default_factory=dict)
    logs: dict = field(default_factory=dict)
    record: BenchmarkRecord | None = None

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
        return (
            self.implementation.library == "sklearnex"
            and self.category == "tree-based"
        )

    @property
    def minimal_match_key(self) -> str:
        """
        If two results don't share the same minimal_match_key, it will
        never makes sense to compare them.
        """
        case = without_keys(
            self.case,
            excluded_names={"implementation", "max_bins"},
        )
        case["method"] = self.method
        return stable_json(case)

    @property
    def full_match_key(self) -> str:
        """
        If two results share the same full_match_key, it's not useful to include
        both in a given plot.
        """
        return stable_json(
            {
                **self.case,
                "method": self.method,
                "hardware": self.hardware_hash,
                "software": self.software_hash,
            }
        )


def _parse_result_timestamp(path: Path) -> datetime:
    match = RESULT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Result filename '{path}' does not end with _<datetime>.json")
    timestamp = match.group(1)
    date_format = "%Y%m%dT%H%M%S%fZ" if len(timestamp) > 16 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(timestamp, date_format).replace(tzinfo=timezone.utc)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_values(values: list[Any]) -> list[float] | None:
    if not values or not all(_is_number(value) for value in values):
        return None
    return [float(value) for value in values]


def _metric_tolerance(base_values: list[float]) -> float:
    base_mean = mean(base_values)
    base_std = stdev(base_values) if len(base_values) >= 2 else 0.0
    return max(
        METRIC_STD_TOLERANCE_FACTOR * base_std,
        METRIC_ABS_TOLERANCE_FLOOR,
        METRIC_REL_TOLERANCE_FLOOR * abs(base_mean),
    )


def _short_metric_value(value: float) -> str:
    return f"{value:.3g}"


def _first_non_empty_logs(rows: list[dict]) -> dict:
    empty_logs = {"stdout": "", "stderr": ""}
    for row in rows:
        logs = row.get("logs", {}) or {}
        stdout = str(logs.get("stdout", ""))
        stderr = str(logs.get("stderr", ""))
        if stdout or stderr:
            return {"stdout": stdout, "stderr": stderr}
    return empty_logs


def _runs_to_values(runs: list[dict]) -> dict:
    values = {
        "time_ms": {},
        "data_desc": {},
        "metrics": {},
        "attributes": {},
    }
    for run in runs:
        for method, time_ms in run.get("time_ms", {}).items():
            values["time_ms"].setdefault(method, []).append(float(time_ms))

        for method, data_desc in run.get("data_desc", {}).items():
            if method in values["data_desc"]:
                if stable_json(data_desc) != stable_json(values["data_desc"][method]):
                    raise ValueError(f"Inconsistent data_desc across repeats for {method}")
            else:
                values["data_desc"][method] = data_desc

        for method, method_metrics in run.get("metrics", {}).items():
            metric_values = values["metrics"].setdefault(method, {})
            for metric_name, metric_value in method_metrics.items():
                metric_values.setdefault(metric_name, []).append(metric_value)

        for name, value in (run.get("attributes", {}) or {}).items():
            values["attributes"].setdefault(name, []).append(value)

    return values


def read_benchmark_records(path=None) -> list[BenchmarkRecord]:
    """
    Read new-format benchmark result files from `path`, defaulting to ./results/.
    """
    results_root = Path(path) if path is not None else Path("results")
    records: list[BenchmarkRecord] = []

    for result_path in sorted(results_root.glob("*.json")):
        if not RESULT_FILE_RE.match(result_path.name):
            continue

        with open(result_path, "r") as f:
            result_file = json.load(f)

        hardware_hash = result_file["hardware_hash"]
        software_hash = result_file["software_hash"]
        timestamp = _parse_result_timestamp(result_path)

        for bench_case in result_file.get("bench_cases", []):
            if "results" not in bench_case:
                raise ValueError(
                    f"{result_path} contains legacy aggregated benchmark results"
                )
            records.append(
                BenchmarkRecord(
                    hardware_hash=hardware_hash,
                    software_hash=software_hash,
                    timestamp_recorded=timestamp,
                    case=without_keys(
                        bench_case.get("case", {}), excluded_names={"bench"}
                    ),
                    runs=bench_case.get("results", []),
                )
            )

    return records


def method_results_from_records(records: list[BenchmarkRecord]) -> list[MethodResult]:
    results: list[MethodResult] = []
    for record in records:
        values = _runs_to_values(record.runs)
        logs = _first_non_empty_logs(record.runs)

        for method, times in values["time_ms"].items():
            results.append(
                MethodResult(
                    hardware_hash=record.hardware_hash,
                    software=record.software_hash,
                    software_hash=record.software_hash,
                    method=method,
                    timestamp_recorded=record.timestamp_recorded,
                    case=record.case,
                    times=times,
                    data_desc=values["data_desc"][method],
                    metrics=values["metrics"],
                    attributes=values["attributes"],
                    logs=logs,
                    record=record,
                )
            )
    return results


def read_all_results(path=None) -> list[MethodResult]:
    """
    path defaults to ./results/

    Read all available results and de-duplicates redundant results, i.e.
    results with the same `full_match_key` (keep the latest).
    """
    results = method_results_from_records(read_benchmark_records(path))

    latest_by_key: dict[str, MethodResult] = {}
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
    base_result: MethodResult
    matched_result: MethodResult
    warnings: list[MatchWarning]

    @staticmethod
    def _comparable_metrics(metrics: dict) -> dict:
        metrics = {
            method: dict(method_metrics)
            for method, method_metrics in metrics.items()
        }

        # for regression, we pop RMSE, as it's redundant with R2 but
        # depends on the variance of y, which makes the comparison harder to do.
        for method_metrics in metrics.values():
            if "R2" in method_metrics and "RMSE" in method_metrics:
                method_metrics.pop("RMSE")
        return metrics

    def _metric_difference(
        self,
        method: str,
        metric_name: str,
        base_values: list[Any],
        matched_values: list[Any],
    ) -> str | None:
        numeric_base = _numeric_values(base_values)
        numeric_matched = _numeric_values(matched_values)
        if numeric_base is None or numeric_matched is None:
            if base_values != matched_values:
                return (
                    f"{method}.{metric_name}: "
                    f"base={base_values}, target={matched_values}"
                )
            return None

        base_mean = mean(numeric_base)
        matched_mean = mean(numeric_matched)
        tolerance = _metric_tolerance(numeric_base)
        if abs(matched_mean - base_mean) > tolerance:
            return (
                f"{method}.{metric_name}: "
                f"base_mean={_short_metric_value(base_mean)}, "
                f"target_mean={_short_metric_value(matched_mean)}, "
                f"tolerance={_short_metric_value(tolerance)}"
            )
        return None

    @property
    def metrics_differences(self) -> list[str]:
        if self.base_result.method != "fit":
            return []

        differences = []
        base_metrics = self._comparable_metrics(self.base_result.metrics)
        matched_metrics = self._comparable_metrics(self.matched_result.metrics)

        for method in sorted(set(base_metrics) | set(matched_metrics)):
            if method not in base_metrics:
                differences.append(f"{method}: missing in base")
                continue
            if method not in matched_metrics:
                differences.append(f"{method}: missing in target")
                continue

            base_method_metrics = base_metrics[method]
            matched_method_metrics = matched_metrics[method]
            for metric_name in sorted(
                set(base_method_metrics) | set(matched_method_metrics)
            ):
                if metric_name not in base_method_metrics:
                    differences.append(f"{method}.{metric_name}: missing in base")
                    continue
                if metric_name not in matched_method_metrics:
                    differences.append(f"{method}.{metric_name}: missing in target")
                    continue

                difference = self._metric_difference(
                    method,
                    metric_name,
                    base_method_metrics[metric_name],
                    matched_method_metrics[metric_name],
                )
                if difference is not None:
                    differences.append(difference)
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
    message=(
        "Scikit-learn intelex uses binning & histogram-based splits "
        "while scikit-learn doesn't"
    ),
)

CPU_FALLBACK_WARNING = MatchWarning(
    icon="↩",
    short_message="CPU fallback",
    message="Some operations fell back to CPU according to benchmark logs",
)


def _logs_text(result: MethodResult) -> str:
    return "\n".join(str(result.logs.get(stream, "")) for stream in ("stdout", "stderr"))


def has_cpu_fallback_warning(result: MethodResult) -> bool:
    text = _logs_text(result).lower()
    return (
        "fallback from xpu to cpu" in text
        or ("aten op fallback" in text and "cpu" in text)
    )


def append_cpu_fallback_warning(result: MethodResult, warnings: list):
    if result.implementation.device not in {"cuda", "gpu", "xpu"}:
        return
    if has_cpu_fallback_warning(result):
        warnings.append(CPU_FALLBACK_WARNING)


def append_max_bins_warning(
    sklearn_res: MethodResult, sklearnex_res: MethodResult, warnings: list
):
    assert sklearn_res.implementation.library == "sklearn"
    assert sklearnex_res.implementation.library == "sklearnex"

    estimator_params = sklearnex_res.case.get("algorithm", {}).get(
        "estimator_params", {}
    )
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


def append_iterations_warning(
    base_res: MethodResult, candidate: MethodResult, warnings: list
):
    # expect .attributes to be in the form:
    # {'n_iter': [...] (values for all the runs), ...}
    base_iterations = base_res.attributes.get("n_iter", [])
    candidate_iterations = candidate.attributes.get("n_iter", [])
    if set(base_iterations) == set(candidate_iterations):
        return

    if set(base_iterations).intersection(set(candidate_iterations)):
        # this is quite permissive, but for now let's do that
        return

    if len(candidate_iterations) == 0:
        warnings.append(
            MatchWarning(
                icon="🔁",
                short_message=f"({base_iterations[0]} vs ?)",
                message="Number iteration of iteration not reported for this variant",
            )
        )
    elif len(base_iterations) == 0:
        warnings.append(
            MatchWarning(
                icon="🔁",
                short_message=f"(? vs {candidate_iterations[0]})",
                message="Number iteration of iteration not reported for the baseline",
            )
        )
    else:
        warnings.append(
            MatchWarning(
                icon="🔁",
                short_message=f"({base_iterations[0]} vs {candidate_iterations[0]})",
                message="Number of iteration differs: this might mean algorithms differ",
            )
        )


def find_matches(
    base_results: list[MethodResult],
    results_to_match: list[MethodResult],
    match_function,
    match_key=None,
) -> list[Match]:
    # should assert all result in results_to_match match at most one base_results
    if match_key is None:
        match_key = lambda result: result.minimal_match_key
    base_by_minimal_key: dict[str, list[MethodResult]] = {}
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


def date_range(results: list[MethodResult]) -> dict:
    if not results:
        return {"label": "No results"}
    start = min(result.timestamp_recorded for result in results)
    end = max(result.timestamp_recorded for result in results)
    if start.date() == end.date():
        date_range = start.strftime("%Y-%m-%d")
    else:
        date_range = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
    return {"label": f"{len(results)} benchmark records from {date_range}"}
