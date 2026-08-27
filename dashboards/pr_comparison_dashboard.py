"""Simple table-based before/after comparison between a PR branch's
sklearn-dev build and the scikit-learn `main` sklearn-dev baseline, both
benchmarked on the same self-hosted runner in one CI job.

Deliberately named without the "gen_" prefix the other dashboards/gen_*.py
scripts share, so it's excluded from the loops that regenerate every
gen_*.py dashboard from the repo's full committed results/ (dashboard-pages.yml,
dashboard-preview-build.yml, watch_dashboards.py, CONTRIBUTING.md's local
preview instructions) - this script's `results/` is instead an ephemeral,
CI-produced directory (see .github/workflows/pr-comparison.yml) containing
only the two runs being compared, and it would fail loudly if pointed at
the real one (see below).

Unlike those other dashboards, there's exactly one hardware_hash, one
config, and exactly two sklearn-dev builds. This script assumes that shape
rather than defending against the general multi-hardware/multi-build case
the other dashboards handle: its only caller is that one CI job, so a shape
mismatch here means the pipeline itself is broken, not "no data yet".
"""

from html import escape
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboards.output import dashboard_output_dir
from sklbench.reporting.utils import groupby, stable_json, without_keys
from sklbench.reporting.matching import (
    append_iterations_warning,
    read_all_results,
    read_failed_records,
    find_matches,
    date_range,
    BenchmarkRecord,
    MatchWarning,
    MethodResult,
)
from sklbench.reporting.envs import (
    FLAMEGRAPH_VIEWER_BASE_URL,
    JSON_VIEWER_BASE_URL,
    external_viewer_url,
    json_viewer_url,
    profile_viewer_url,
    read_env,
    software_build_name,
    summarize_hardware_env,
)
from sklbench.reporting.html import (
    BASE_TEMPLATE,
    DATE_RANGE_TEMPLATE,
    HARDWARE_TEMPLATE,
    detailed_results_table_html,
)


SKLEARN_DEV_PIXI_ENV = "sklearn-dev"
# Matches "sklearn-dev@..." as well as pixi-env variants of it, e.g.
# "sklearn-dev-libomp@..." (see configs/_implementations.py). Re-derived
# here rather than imported from gen_hgb_speedup_breakdown.py, per this
# codebase's convention of each gen_*.py dashboard owning its own such
# helpers instead of importing another dashboard module's internals.
_SKLEARN_DEV_BUILD_RE = re.compile(rf"^{re.escape(SKLEARN_DEV_PIXI_ENV)}-?.*@")


def _is_sklearn_dev_build(build_name: str) -> bool:
    return bool(_SKLEARN_DEV_BUILD_RE.match(build_name))


def _base_build(builds: list[str]) -> str | None:
    """The `sklearn-dev@<owner>:main` build among `builds` - picked
    dynamically since `<owner>` varies with whichever fork/remote this
    comparison's baseline used (see CONTRIBUTING.md's `env@owner:ref`
    workflow), same heuristic as gen_hgb_speedup_breakdown.py's
    `_base_build`."""
    return next((build for build in builds if build.rsplit(":", 1)[-1] == "main"), None)


def _branch_label(build_name: str) -> str:
    """Strip the `sklearn-dev@owner:` prefix off a build name, leaving just
    the ref/branch (e.g. "sklearn-dev@cakedev0:ridge/optim_cholesky" ->
    "ridge/optim_cholesky") - the owner is noise in a table whose whole point
    is comparing two branches of the same PR."""
    return build_name.rsplit(":", 1)[-1]


def _case_key(case: dict) -> str:
    """Case identity ignoring implementation/max_bins - shared by a base
    result and the candidate(s) it would be compared against."""
    return stable_json(without_keys(case, excluded_names={"implementation", "max_bins"}))


def result_matches(
    base_res: MethodResult, candidate: MethodResult
) -> tuple[bool, list[MatchWarning]]:
    warnings = []
    append_iterations_warning(base_res, candidate, warnings)
    return base_res.minimal_match_key == candidate.minimal_match_key, warnings


def _sklearn_commit_url(software_hash: str) -> str | None:
    """A GitHub commit-view URL for the scikit-learn checkout `software_hash`
    was built from, or None if that info isn't there (e.g. a failed-only
    build whose env capture never ran). `sklearn-dev` builds always source
    scikit-learn from a git checkout (see scripts/setup_sklearn_ref.sh), so
    `git_info` should always resolve to remote+commit in practice."""
    git_info = read_env("software", software_hash).get("runtime_imports", {}).get("sklearn", {}).get("git")
    if not git_info:
        return None
    commit = git_info.get("commit")
    owner_repo_match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", git_info.get("remote", ""))
    if not commit or not owner_repo_match:
        return None
    owner, repo = owner_repo_match.groups()
    return f"https://github.com/{owner}/{repo}/commit/{commit}"


def _hosted_url_fn(
    viewer_base_url: str,
    fallback: Callable[[Path], str],
    site_base_url: str | None,
    output_dir: Path,
):
    """Build a `json_url_fn`/`profile_url_fn` for `detailed_results_table_html`.

    PR-comparison results are ephemeral (never committed to `results/`, see
    .github/workflows/pr-comparison.yml), so `json_viewer_url`/
    `profile_viewer_url`'s GitHub-raw-URL links 404 - the underlying file
    doesn't exist at any repo ref. When the CI job tells us where this run's
    site will be deployed (`SKLBENCH_PR_COMPARE_SITE_URL`), copy each
    referenced record/profile file into the site output directory instead
    and link the viewer at that to-be-deployed copy. Without a known site URL
    (e.g. a local ephemeral results/ dir), fall back to the normal
    GitHub-raw-URL link rather than fail outright.
    """

    def build(record_path: Path) -> str | None:
        if site_base_url is None:
            return fallback(record_path)
        dest = output_dir / record_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record_path, dest)
        hosted_url = f"{site_base_url}/{record_path.as_posix()}"
        return external_viewer_url(viewer_base_url, hosted_url)

    return build


def _first_source(results: list[MethodResult], failed: list[BenchmarkRecord]):
    """A software_hash to source a build's scikit-learn commit link from -
    handles the (unlikely but possible) case where every case for one side
    failed outright."""
    source = results[0] if results else (failed[0] if failed else None)
    return source.software_hash if source is not None else None


if __name__ == "__main__":
    results = read_all_results()
    failed_records = read_failed_records()

    if not results and not failed_records:
        raise SystemExit("No benchmark results found under results/ - nothing to compare.")

    hardware_hashes = {r.hardware_hash for r in results} | {
        r.hardware_hash for r in failed_records
    }
    if len(hardware_hashes) != 1:
        raise SystemExit(
            "Expected results from exactly one hardware_hash (this dashboard assumes a "
            f"single self-hosted runner), found {len(hardware_hashes)}: {sorted(hardware_hashes)}."
        )
    (hardware_hash,) = hardware_hashes

    by_build = groupby(results, lambda r: software_build_name(r.software_hash))
    failed_by_build = groupby(failed_records, lambda r: software_build_name(r.software_hash))
    all_builds = sorted(set(by_build) | set(failed_by_build))

    non_dev_builds = [b for b in all_builds if not _is_sklearn_dev_build(b)]
    if non_dev_builds:
        raise SystemExit(f"Expected only sklearn-dev@... builds in results/, found: {non_dev_builds}.")

    base_build = _base_build(all_builds)
    if base_build is None:
        raise SystemExit(f"No sklearn-dev@...:main baseline build found among: {all_builds}.")

    variant_builds = [b for b in all_builds if b != base_build]
    if len(variant_builds) != 1:
        raise SystemExit(
            f"Expected exactly one PR-branch build besides the {base_build!r} baseline, "
            f"found {len(variant_builds)}: {variant_builds}."
        )
    (variant_build,) = variant_builds

    base_results = by_build.get(base_build, [])
    variant_results = by_build.get(variant_build, [])
    base_failed = failed_by_build.get(base_build, [])
    variant_failed = failed_by_build.get(variant_build, [])

    matches = find_matches(base_results, variant_results, result_matches)
    matches_by_method = groupby(matches, lambda match: match.matched_result.method)

    # A failed record means find_matches never sees a pair for that case -
    # the side that *did* succeed would otherwise silently vanish from the
    # table. Look those up by case identity so they still show up (with no
    # speedup, since there's no successful counterpart).
    base_by_case_key = groupby(base_results, lambda r: _case_key(r.case))
    variant_by_case_key = groupby(variant_results, lambda r: _case_key(r.case))
    unmatched_variant_results = [
        r for record in base_failed for r in variant_by_case_key.get(_case_key(record.case), [])
    ]
    unmatched_base_results = [
        r for record in variant_failed for r in base_by_case_key.get(_case_key(record.case), [])
    ]

    output_dir = dashboard_output_dir()
    # Set by .github/workflows/pr-comparison.yml's "Generate comparison
    # dashboard" step to the Cloudflare Pages URL this run's "Deploy to
    # Cloudflare Pages" step will publish to next - known ahead of the deploy
    # itself since the branch name is deterministic from PR number + runner.
    site_base_url = os.environ.get("SKLBENCH_PR_COMPARE_SITE_URL", "").rstrip("/") or None

    table_html = detailed_results_table_html(
        "all",
        matches_by_method,
        baseline_label=_branch_label(base_build),
        variant_label=lambda result: _branch_label(variant_build),
        failed_records=(
            [(record, _branch_label(base_build)) for record in base_failed]
            + [(record, _branch_label(variant_build)) for record in variant_failed]
        ),
        unmatched_base_results=unmatched_base_results,
        unmatched_candidate_results=unmatched_variant_results,
        open=True,
        variant_column_title="Branch name",
        default_variant_filter=_branch_label(variant_build),
        json_url_fn=_hosted_url_fn(
            JSON_VIEWER_BASE_URL, json_viewer_url, site_base_url, output_dir
        ),
        profile_url_fn=_hosted_url_fn(
            FLAMEGRAPH_VIEWER_BASE_URL, profile_viewer_url, site_base_url, output_dir
        ),
    )

    commit_urls = {}
    for build_name, build_results, build_failed in (
        (base_build, base_results, base_failed),
        (variant_build, variant_results, variant_failed),
    ):
        software_hash = _first_source(build_results, build_failed)
        commit_urls[build_name] = (
            _sklearn_commit_url(software_hash) if software_hash is not None else None
        )

    commit_links = []
    for build_name in (base_build, variant_build):
        label = escape(_branch_label(build_name))
        commit_url = commit_urls[build_name]
        if commit_url is None:
            commit_links.append(f"<li>{label}: <span class=\"muted\">commit unknown</span></li>")
        else:
            commit_links.append(f'<li>{label}: <a href="{escape(commit_url)}">view commit</a></li>')

    variant_label_html = escape(_branch_label(variant_build))
    if commit_urls[variant_build] is not None:
        variant_label_html = f'<a href="{escape(commit_urls[variant_build])}">{variant_label_html}</a>'

    about_html = f"""<section class="panel">
  <p>This page compares two scikit-learn builds &mdash; <code>main</code> and
  {variant_label_html} &mdash; benchmarked back-to-back on the same self-hosted
  runner in one CI job (see the commits below for exactly what was compared).
  Each table row is one benchmark case (estimator, dataset, hyperparameters);
  <code>fit speedup</code>/<code>predict speedup</code> is the branch's time
  relative to the <code>main</code> baseline.
  Use the column filters to narrow down by estimator, dataset, or branch name.</p>
</section>"""

    rows = [
        about_html,
        DATE_RANGE_TEMPLATE.render(date_range(results)),
        HARDWARE_TEMPLATE.render(summarize_hardware_env(read_env("hardware", hardware_hash))),
        '<section class="panel"><h2>scikit-learn commits</h2>'
        f'<ul class="compact">{"".join(commit_links)}</ul></section>',
        table_html or '<section class="empty">No comparable benchmark cases.</section>',
    ]

    html = BASE_TEMPLATE.render(
        title=f"PR comparison: {variant_build} vs {base_build}",
        rows=[f'<div class="page-row">{row}</div>' for row in rows],
    )

    output = output_dir / "pr_comparison.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
