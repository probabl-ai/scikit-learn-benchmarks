#!/usr/bin/env bash
# Clear the `runner-ready` job after an unsuccessful provision by cancelling the
# run, which releases a job still queued on the runner group. Shared by the
# `cleanup-aws` job of run-benchmarks.yml and of pr-comparison.yml; each caller
# keeps its own `if:`/`needs:`/`permissions:` and passes `GH_TOKEN`.
#
# Usage: clear-stale-runner-ready.sh
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
    # No job by that name; the branch below cancels the run.
    status="missing"
    ;;
  *)
    # The job list could not be read, so no status is known and the run is left
    # alone by the `unknown` branch below.
    status="unknown"
    ;;
esac

case "$status" in
  in_progress|completed)
    echo "the readiness job is $status; nothing to clear"
    ;;
  unknown)
    echo "::warning::could not read the readiness job's status; leaving the run for manual cleanup"
    ;;
  *)
    echo "::error::the readiness job is still $status after an unsuccessful provision; cancelling the run to clear it"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    ;;
esac
