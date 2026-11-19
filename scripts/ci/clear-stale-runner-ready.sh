#!/usr/bin/env bash
# Clear the monitored job after an unsuccessful provision.
#
# Shared by the `cleanup-aws` job of `.github/workflows/run-benchmarks.yml` and
# of `.github/workflows/pr-comparison.yml`. Only the shell below is shared: each
# caller keeps its own `if:`/`needs:`/`permissions:` and passes `GH_TOKEN`.
#
# Both callers invoke this script with no arguments and get the same cleanup
# policy. The job is looked up through `scripts/ci/monitored-job.sh`, the
# resolver the provision waiter also uses, which holds the name of the
# readiness job.
#
# Cancelling a run is what releases a readiness job still queued on the runner
# group, so a run whose job list answers with no such job, or with several, has
# its queued job cleared by cancelling it. A cancel cannot be undone: it ends
# the run and discards whatever work the run had already done. An unreadable
# job list warns and leaves the run alone, and the queued readiness job is
# released the next time the host is started.
#
# Usage: clear-stale-runner-ready.sh
#
# Reads GH_TOKEN, GITHUB_REPOSITORY and GITHUB_RUN_ID from the environment.
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ne 0 ]; then
  echo "::error::clear-stale-runner-ready.sh takes no arguments, got: $*" >&2
  exit 2
fi

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
    # releases all of them, and a duplicate name is a workflow-definition
    # error that this cleanup clears.
    echo "::error::more than one job in run ${GITHUB_RUN_ID} answers to the readiness job's name; cancelling the run to clear them"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    exit 0
    ;;
  *)
    # Any other resolver exit: the job list could not be read, so no status is
    # known and the run is left alone by the `unknown` branch below.
    status="unknown"
    ;;
esac

case "$status" in
  in_progress|completed)
    echo "the readiness job is $status; nothing to clear"
    ;;
  unknown)
    # Unreadable job list: no status is known, so warn and leave the run alone.
    echo "::warning::could not read the readiness job's status (unreadable job list); leaving the run for manual cleanup"
    ;;
  *)
    echo "::error::the readiness job is still $status after an unsuccessful provision; cancelling the run to clear it"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    ;;
esac
