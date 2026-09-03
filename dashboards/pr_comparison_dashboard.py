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

Unlike those other dashboards, there's exactly one hardware_hash, and
exactly one base + one variant sklearn-dev build per pixi env being
compared (usually just one env; a `runs:` directive entry - see
COMPARISONS_PR.md - can compare more than one, e.g. sklearn-dev and
sklearn-dev-libomp, in which case each env is matched/labeled
independently and merged into one table). This script assumes that shape
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
# here rather than imported from gen_hgb_dev_speedup_breakdown.py, per this
# codebase's convention of each gen_*.py dashboard owning its own such
# helpers instead of importing another dashboard module's internals.
_SKLEARN_DEV_BUILD_RE = re.compile(rf"^{re.escape(SKLEARN_DEV_PIXI_ENV)}-?.*@")


def _is_sklearn_dev_build(build_name: str) -> bool:
    return bool(_SKLEARN_DEV_BUILD_RE.match(build_name))


def _base_build(builds: list[str]) -> str | None:
    """The `sklearn-dev@<owner>:main` build among `builds` - picked
    dynamically since `<owner>` varies with whichever fork/remote this
    comparison's baseline used (see CONTRIBUTING.md's `env@owner:ref`
    workflow), same heuristic as gen_hgb_dev_speedup_breakdown.py's
    `_base_build`."""
    return next((build for build in builds if build.rsplit(":", 1)[-1] == "main"), None)


def _branch_label(build_name: str) -> str:
    """Strip the `sklearn-dev@owner:` prefix off a build name, leaving just
    the ref/branch (e.g. "sklearn-dev@cakedev0:ridge/optim_cholesky" ->
    "ridge/optim_cholesky") - the owner is noise in a table whose whole point
    is comparing two branches of the same PR."""
    return build_name.rsplit(":", 1)[-1]


def _env_of(build_name: str) -> str:
    """The pixi env a build name was built under (e.g.
    "sklearn-dev-libomp@cakedev0:ridge/optim_cholesky" -> "sklearn-dev-libomp"),
    per `software_build_name`'s `{pixi_env}@{owner:ref}` format. A `runs:`
    directive entry (see COMPARISONS_PR.md) can run the same sklearn_ref vs
    main comparison under more than one such pixi env, in which case
    matching/labels are scoped per env group below rather than assuming
    exactly one base + one variant build overall."""
    return build_name.split("@", 1)[0]


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

    # A `runs:` directive entry can compare the same sklearn_ref against main
    # under more than one pixi env (see _env_of) - exactly one base + one
    # variant build per env group, matched/labeled independently per group,
    # never across envs.
    builds_by_env = groupby(all_builds, _env_of)
    env_groups: list[tuple[str, str, str]] = []  # (env, base_build, variant_build)
    for env, env_builds in sorted(builds_by_env.items()):
        env_base_build = _base_build(env_builds)
        if env_base_build is None:
            raise SystemExit(f"No {env}@...:main baseline build found among: {env_builds}.")
        env_variant_builds = [b for b in env_builds if b != env_base_build]
        if len(env_variant_builds) != 1:
            raise SystemExit(
                f"Expected exactly one PR-branch build besides the {env_base_build!r} baseline "
                f"for env {env!r}, found {len(env_variant_builds)}: {env_variant_builds}."
            )
        env_groups.append((env, env_base_build, env_variant_builds[0]))

    base_branch_labels = {_branch_label(base) for _, base, _ in env_groups}
    if len(base_branch_labels) != 1:
        raise SystemExit(f"Expected the same baseline branch across all envs, found: {sorted(base_branch_labels)}.")
    variant_branch_labels = {_branch_label(variant) for _, _, variant in env_groups}
    if len(variant_branch_labels) != 1:
        raise SystemExit(
            f"Expected the same PR-branch across all envs (one shared sklearn_ref), "
            f"found: {sorted(variant_branch_labels)}."
        )
    (base_branch_label,) = base_branch_labels
    (variant_branch_label,) = variant_branch_labels

    multi_env = len(env_groups) > 1

    def build_label(build_name: str) -> str:
        """Branch label, env-qualified only when more than one env is being
        compared - keeps the common single-env case's labels unchanged."""
        label = _branch_label(build_name)
        return f"{label} [{_env_of(build_name)}]" if multi_env else label

    matches = []
    unmatched_base_results = []
    unmatched_variant_results = []
    failed_records_for_table: list[tuple[BenchmarkRecord, str]] = []
    for env, base_build, variant_build in env_groups:
        base_results = by_build.get(base_build, [])
        variant_results = by_build.get(variant_build, [])
        base_failed = failed_by_build.get(base_build, [])
        variant_failed = failed_by_build.get(variant_build, [])

        matches.extend(find_matches(base_results, variant_results, result_matches))

        # A failed record means find_matches never sees a pair for that case -
        # the side that *did* succeed would otherwise silently vanish from the
        # table. Look those up by case identity (within this env group only)
        # so they still show up (with no speedup, since there's no successful
        # counterpart).
        base_by_case_key = groupby(base_results, lambda r: _case_key(r.case))
        variant_by_case_key = groupby(variant_results, lambda r: _case_key(r.case))
        unmatched_variant_results.extend(
            r for record in base_failed for r in variant_by_case_key.get(_case_key(record.case), [])
        )
        unmatched_base_results.extend(
            r for record in variant_failed for r in base_by_case_key.get(_case_key(record.case), [])
        )

        failed_records_for_table.extend((record, build_label(base_build)) for record in base_failed)
        failed_records_for_table.extend((record, build_label(variant_build)) for record in variant_failed)

    matches_by_method = groupby(matches, lambda match: match.matched_result.method)

    output_dir = dashboard_output_dir()
    # Set by .github/workflows/pr-comparison.yml's "Generate comparison
    # dashboard" step to the Cloudflare Pages URL this run's "Deploy to
    # Cloudflare Pages" step will publish to next - known ahead of the deploy
    # itself since the branch name is deterministic from PR number + runner.
    site_base_url = os.environ.get("SKLBENCH_PR_COMPARE_SITE_URL", "").rstrip("/") or None

    table_html = detailed_results_table_html(
        "all",
        matches_by_method,
        baseline_label=lambda result: build_label(software_build_name(result.software_hash)),
        variant_label=lambda result: build_label(software_build_name(result.software_hash)),
        failed_records=failed_records_for_table,
        unmatched_base_results=unmatched_base_results,
        unmatched_candidate_results=unmatched_variant_results,
        open=True,
        variant_column_title="Branch name",
        default_variant_filter=None if multi_env else build_label(env_groups[0][2]),
        json_url_fn=_hosted_url_fn(
            JSON_VIEWER_BASE_URL, json_viewer_url, site_base_url, output_dir
        ),
        profile_url_fn=_hosted_url_fn(
            FLAMEGRAPH_VIEWER_BASE_URL, profile_viewer_url, site_base_url, output_dir
        ),
    )

    commit_urls = {}
    for env, base_build, variant_build in env_groups:
        for build_name in (base_build, variant_build):
            software_hash = _first_source(by_build.get(build_name, []), failed_by_build.get(build_name, []))
            commit_urls[build_name] = (
                _sklearn_commit_url(software_hash) if software_hash is not None else None
            )

    commit_links = []
    for env, base_build, variant_build in env_groups:
        for build_name in (base_build, variant_build):
            label = escape(build_label(build_name))
            commit_url = commit_urls[build_name]
            if commit_url is None:
                commit_links.append(f"<li>{label}: <span class=\"muted\">commit unknown</span></li>")
            else:
                commit_links.append(f'<li>{label}: <a href="{escape(commit_url)}">view commit</a></li>')

    variant_label_html = escape(variant_branch_label)
    first_variant_commit_url = next((commit_urls[v] for _, _, v in env_groups if commit_urls.get(v)), None)
    if first_variant_commit_url is not None:
        variant_label_html = f'<a href="{escape(first_variant_commit_url)}">{variant_label_html}</a>'

    envs_note = ""
    if multi_env:
        env_names_html = ", ".join(f"<code>{escape(env)}</code>" for env, _, _ in env_groups)
        envs_note = f" Compared under {len(env_groups)} pixi envs: {env_names_html} &mdash; use the branch-name filter to isolate one."

    about_html = f"""<section class="panel">
  <p>This page compares two scikit-learn builds &mdash; <code>main</code> and
  {variant_label_html} &mdash; benchmarked back-to-back on the same self-hosted
  runner in one CI job (see the commits below for exactly what was compared).{envs_note}
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

    title = (
        f"PR comparison: {env_groups[0][2]} vs {env_groups[0][1]}"
        if not multi_env
        else f"PR comparison: {variant_branch_label} vs {base_branch_label}"
    )
    html = BASE_TEMPLATE.render(
        title=title,
        rows=[f'<div class="page-row">{row}</div>' for row in rows],
    )

    output = output_dir / "pr_comparison.html"
    output.write_text(html)
    print(f"Dashboard written to {output}")
