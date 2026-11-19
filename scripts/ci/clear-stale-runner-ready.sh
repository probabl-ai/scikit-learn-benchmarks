#!/usr/bin/env bash
# Clear the monitored job after an unsuccessful provision.
#
# Shared by the `cleanup-aws` job of `.github/workflows/run-benchmarks.yml` and
# of `.github/workflows/pr-comparison.yml`. Only the shell below is shared: each
# caller keeps its own `if:`/`needs:`/`permissions:` and passes `GH_TOKEN`.
#
# Both callers invoke this script with no arguments, and they get one policy:
# there used to be a flag that let PR comparisons cancel a run whose job list was
# unreadable; it is gone on purpose. An unreadable job list is a warning here,
# for every caller - see the `unknown` branch below for why.
#
# The job is looked up through `scripts/ci/monitored-job.sh`, the resolver the
# provision waiter also uses, which holds the name of the readiness job. A run
# whose job list answers with no such job, or several, is reported rather than
# guessed at. Cancelling the run is what releases a job still queued on the
# runner group.
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
    # releases all of them, and duplicates are a workflow-definition error
    # worth clearing rather than leaving queued.
    echo "::error::more than one job in run ${GITHUB_RUN_ID} answers to the readiness job's name; cancelling the run to clear them"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    exit 0
    ;;
  *)
    # Job list unreadable: the resolver could not tell us whether the readiness
    # job is queued, so the decision is made below - and both callers make it
    # the same way.
    status="unknown"
    ;;
esac

case "$status" in
  in_progress|completed)
    echo "the readiness job is $status; nothing to clear"
    ;;
  unknown)
    # Unreadable job list: warn, and leave the run alone. This is ONE policy for
    # both callers, chosen deliberately - it is not a behaviour that a refactor
    # dropped from one workflow by accident.
    #
    # Deciding under ignorance must not destroy a run. A readiness job still
    # queued on the runner group is released the next time the host is started,
    # so if we are wrong to leave it, the cost is one stale queued job that the
    # next run clears anyway. A cancel, by contrast, is unrecoverable: it kills a
    # run whose state we could not even read, throwing away whatever work it had
    # already done. We cannot tell which of those two we are choosing, so we take
    # the one that can be undone by simply starting the host again.
    echo "::warning::could not read the readiness job's status (unreadable job list); leaving the run for manual cleanup"
    ;;
  *)
    echo "::error::the readiness job is still $status after an unsuccessful provision; cancelling the run to clear it"
    gh api -X POST \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/cancel" \
      >/dev/null || true
    ;;
esac
