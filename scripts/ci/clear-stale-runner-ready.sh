#!/usr/bin/env bash
# Clear the monitored job after an unsuccessful provision.
#
# Shared by the `cleanup-aws` job of `.github/workflows/run-benchmarks.yml` and
# of `.github/workflows/pr-comparison.yml`. Only the shell below is shared: each
# caller keeps its own `if:`/`needs:`/`permissions:` and passes `GH_TOKEN`.
#
# The job is looked up through `scripts/ci/monitored-job.sh`, the resolver the
# provision waiter also uses, which holds the name of the readiness job. A run
# whose job list answers with no such job, or several, is reported rather than
# guessed at. Cancelling the run is what releases a job still queued on the
# runner group.
#
# Usage: clear-stale-runner-ready.sh [--cancel-on-unknown]
#
#   --cancel-on-unknown  Treat an unreadable job list as stale too, and cancel
#                        the run (pr-comparison.yml: PR comparisons are
#                        low-stakes). Without it the run is left alone
#                        (run-benchmarks.yml: a spurious cancel would destroy an
#                        owner-dispatched hours-long benchmark run).
#
# Reads GH_TOKEN, GITHUB_REPOSITORY and GITHUB_RUN_ID from the environment.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cancel_on_unknown=false
for arg in "$@"; do
  case "$arg" in
    --cancel-on-unknown)
      cancel_on_unknown=true
      ;;
    *)
      echo "::error::unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

resolve_rc=0
status="$("$SCRIPT_DIR/monitored-job.sh")" || resolve_rc=$?
case "$resolve_rc" in
  0)
    ;;
  2)
    # Absent. The resolver named the job; the branch below cancels the run.
    status="missing"
    ;;
  3)
    # Several jobs answer to the readiness job's name. Cancelling is what
    # releases all of them, and duplicates are a workflow-definition error
    # worth clearing rather than leaving queued.
    echo "::error::more than one job in run ${GITHUB_RUN_ID} answers to the readiness job's name; cancelling the run to clear them"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    exit 0
    ;;
  *)
    # Job list unreadable: the `--cancel-on-unknown` policy decides below.
    status="unknown"
    ;;
esac

case "$status" in
  in_progress|completed)
    echo "the readiness job is $status; nothing to clear"
    ;;
  unknown)
    if [ "$cancel_on_unknown" = "true" ]; then
      echo "::error::provision failed and the readiness job's status is unreadable; cancelling the run to clear it"
      gh api -X POST \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
        >/dev/null || true
    else
      echo "::warning::could not read the readiness job's status; leaving the run for manual cleanup"
    fi
    ;;
  *)
    echo "::error::the readiness job is still $status after an unsuccessful provision; cancelling the run to clear it"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    ;;
esac
