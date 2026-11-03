import subprocess

from sklbench.reporting.provenance import RESULTS_REPOSITORY_URL, record_source


def _git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=someone",
         "-c", "user.email=someone@example.com", *args],
        check=True, capture_output=True,
    )


def _add_record(repo, name, message, author="someone"):
    (repo / "results" / "records" / name).write_text("{}")
    _git(repo, "add", ".")
    _git(repo, "-c", f"user.name={author}", "commit", "-m", message)


def test_record_source_from_git_history(tmp_path):
    repo = tmp_path / "repo"
    (repo / "results" / "records").mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _add_record(repo, "manual.json", "laptop results")

    # A CI results PR, merged on GitHub.
    _git(repo, "switch", "-c", "bench-results/intel-xeon-12345")
    _add_record(repo, "ci.json", "Benchmark results", author="github-actions[bot]")
    _git(repo, "switch", "main")
    _git(repo, "merge", "--no-ff", "bench-results/intel-xeon-12345",
         "-m", "Merge pull request #7 from org/bench-results/intel-xeon-12345")
    merged = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    # A local commit pulled on top of it: the PR merge is now on the second
    # parent side of "Merge branch 'main'".
    _git(repo, "switch", "-c", "local", f"{merged}~1")
    _add_record(repo, "squashed.json", "RES more results (#9)")
    _git(repo, "merge", "--no-ff", merged, "-m", "Merge branch 'main'")

    records = repo / "results" / "records"
    manual = record_source(records / "manual.json", None)
    assert manual.run_url is None and not manual.from_ci
    assert manual.pr_number is None and manual.commit_sha is not None

    ci = record_source(records / "ci.json", None)
    assert ci.run_url == f"{RESULTS_REPOSITORY_URL}/actions/runs/12345"
    assert ci.pr_number == "7"

    assert record_source(records / "squashed.json", None).pr_number == "9"

    # The URL stored in the record wins over the branch name.
    stored = "https://github.com/org/repo/actions/runs/1"
    assert record_source(records / "ci.json", stored).run_url == stored

    (records / "local.json").write_text("{}")
    assert record_source(records / "local.json", None).commit_sha is None
