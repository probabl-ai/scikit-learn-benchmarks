#!/usr/bin/env bash
# Print the status of this run's `runner-ready` job, the readiness job the AWS
# provisioning wait watches. `scripts/ci/clear-stale-runner-ready.sh` calls it
# to find the job to clear after a failed provision.
#
# Exit codes: 0, the status is on stdout; 2, the run has no job by that name;
# 4, the run's job list could not be read.
#
# Reads GITHUB_REPOSITORY and GITHUB_RUN_ID; needs `gh` (GH_TOKEN) and `jq`.
set -euo pipefail

readonly AWS_READY_JOB="runner-ready"

: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is not set}"

if ! payload="$(gh api \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100")"; then
  echo "::error::could not read the job list of run ${GITHUB_RUN_ID}, so it is unknown whether the run has a job named '${AWS_READY_JOB}'" >&2
  exit 4
fi

matches="$(jq -c --arg name "$AWS_READY_JOB" \
  '[.jobs[] | select(.name == $name)] | .[0]' <<<"$payload")" || {
  echo "::error::could not parse the job list of run ${GITHUB_RUN_ID}, so it is unknown whether the run has a job named '${AWS_READY_JOB}'" >&2
  exit 4
}

if [ "$matches" = "null" ]; then
  echo "::error::no job named '${AWS_READY_JOB}' in run ${GITHUB_RUN_ID}, so the readiness handshake has nothing to wait for" >&2
  exit 2
fi

jq -r '.status // ""' <<<"$matches"
