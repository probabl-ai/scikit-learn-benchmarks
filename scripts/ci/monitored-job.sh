#!/usr/bin/env bash
# Resolve the job that the AWS readiness handshake monitors.
#
# The job name is defined once, here, and nowhere else: the waiter
# (`scripts/ci/aws-provision.sh`) and the cleanup
# (`scripts/ci/clear-stale-runner-ready.sh`) both call this script, which holds
# the name.
#
# Prints the job's `.status` on stdout when exactly one job in the current run
# carries that name. Anything else is a hard error, reported as an `::error::`
# annotation that names the job and the run:
#
#   0  exactly one match; its status is on stdout
#   2  the run has no job by that name (exit 2)
#   3  the run has more than one job by that name (exit 3)
#   4  the run's job list could not be read (exit 4)
#
# The match is an exact `.name` comparison over the job list of
# `$GITHUB_RUN_ID`, so it covers every job the run creates. Two properties of
# that payload keep the rule well defined:
#
#   * a matrix job is named `<job> (<parameters>)`, and a job called through a
#     reusable workflow is named `<caller> / <job>`; neither equals a plain job
#     name, so a handshake given the plain name reports 2;
#   * two job ids in one workflow file may carry the same `name:`, which is
#     exactly what exit 3 catches.
#
# The job list is read with `--paginate`: a run that spreads across several
# pages of `per_page=100` would otherwise look like it had no such job.
#
# Reads GITHUB_REPOSITORY and GITHUB_RUN_ID from the environment. Requires
# `gh` (GH_TOKEN) and `jq`.
set -euo pipefail

# The one job the readiness handshake waits for: the `name:` of the
# `runner-ready` job in `.github/workflows/run-benchmarks.yml`, and of the
# `runner-ready` job in `.github/workflows/pr-comparison.yml`.
readonly AWS_READY_JOB="runner-ready"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is not set}"

if ! payload="$(gh api --paginate \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100")"; then
  echo "::error::could not read the job list of run ${GITHUB_RUN_ID}, so it is unknown whether the run has a job named '${AWS_READY_JOB}'" >&2
  exit 4
fi

# `--paginate` writes one JSON object per page, so the pages are slurped
# together before selecting.
matches="$(jq -s --arg name "$AWS_READY_JOB" \
  '[.[] | (.jobs // [])[] | select(.name == $name)]' <<<"$payload")" || {
  echo "::error::could not parse the job list of run ${GITHUB_RUN_ID}, so it is unknown whether the run has a job named '${AWS_READY_JOB}'" >&2
  exit 4
}

count="$(jq 'length' <<<"$matches")"
case "$count" in
  0)
    echo "::error::no job named '${AWS_READY_JOB}' in run ${GITHUB_RUN_ID}: the run has no job whose name is exactly '${AWS_READY_JOB}', so the readiness handshake has nothing to wait for" >&2
    exit 2
    ;;
  1)
    jq -r '.[0].status // ""' <<<"$matches"
    ;;
  *)
    echo "::error::${count} jobs are named '${AWS_READY_JOB}' in run ${GITHUB_RUN_ID}: the readiness handshake needs exactly one job by that name to know which one to wait for" >&2
    exit 3
    ;;
esac
