#!/usr/bin/env bash
# Clear a stale `runner-ready` job after an unsuccessful provision.
#
# Shared by the `cleanup-aws` job of `.github/workflows/run-benchmarks.yml` and
# of `.github/workflows/pr-comparison.yml`. Only the shell below is shared: each
# caller keeps its own `if:`/`needs:`/`permissions:` and passes `GH_TOKEN`.
#
# The `runner-ready` job name is a hard contract: this lookup matches it
# literally, exactly like `scripts/ci/aws-provision.sh` (the waiter there). If
# the job is renamed or moved behind a reusable-workflow `uses:` call it would
# be reported as `<caller> / <job>` by the jobs API and never match.
#
# Usage: clear-stale-runner-ready.sh [--cancel-on-unknown]
#
#   --cancel-on-unknown  Treat an unreadable status as stale too, and cancel the
#                        run (pr-comparison.yml: PR comparisons are low-stakes).
#                        Without it the run is left alone (run-benchmarks.yml: a
#                        spurious cancel would destroy an owner-dispatched
#                        hours-long benchmark run).
#
# Reads GH_TOKEN, GITHUB_REPOSITORY and GITHUB_RUN_ID from the environment.
set -euo pipefail

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

status="$(gh api \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
  --jq '[.jobs[] | select(.name == "runner-ready")][0].status // "missing"')" \
  || status="unknown"

case "$status" in
  in_progress|completed)
    echo "runner-ready is $status; nothing to clear"
    ;;
  unknown)
    if [ "$cancel_on_unknown" = "true" ]; then
      echo "::error::provision failed and runner-ready status is unreadable; cancelling the run to clear it"
      gh api -X POST \
        "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
        >/dev/null || true
    else
      echo "::warning::could not read runner-ready status; leaving the run for manual cleanup"
    fi
    ;;
  *)
    echo "::error::runner-ready is still $status after an unsuccessful provision; cancelling the run to clear it"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    ;;
esac
