#!/usr/bin/env bash
# Provision the AWS GPU runner for a benchmark run.
#
# AWS credentials are configured by the caller. The script starts the instance
# and then waits for one job of the calling workflow -- the readiness job, the
# one job that runs on the self-hosted runner -- to be picked up by that
# runner. `scripts/ci/monitored-job.sh` resolves it against this run's job
# list, and a run whose job list does not answer with exactly one job of that
# name is a hard error that says which job was being looked for.
set -euo pipefail

: "${INSTANCE_ID:?INSTANCE_ID is not set}"
: "${REGION:?REGION is not set}"
: "${GH_TOKEN:?GH_TOKEN is not set}"
: "${AWS_RUNNER_NAME:?AWS_RUNNER_NAME is not set}"
: "${AWS_RUNNER_GROUP:?AWS_RUNNER_GROUP is not set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is not set}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly MONITORED_JOB="$SCRIPT_DIR/monitored-job.sh"

readonly READY_DEADLINE="$(( $(date +%s) + 900 ))"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"

# Start the instance if needed.
state="$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text)"
echo "instance $INSTANCE_ID is $state"
case "$state" in
  running|pending)
    ;;
  stopped)
    aws ec2 start-instances \
      --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    ;;
  stopping)
    # Still shutting down; wait, then start it again.
    aws ec2 wait instance-stopped \
      --instance-ids "$INSTANCE_ID" --region "$REGION"
    aws ec2 start-instances \
      --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    ;;
  *)
    echo "::error::instance $INSTANCE_ID is $state; refusing to continue"
    exit 1
    ;;
esac
if [ "$state" != "running" ]; then
  aws ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" --region "$REGION"
  echo "instance $INSTANCE_ID is running"
fi

# The monitored job can stay queued if the AWS runner is busy with another
# job. Detect that through the jobs API; the workflow token cannot read
# runner status.
runner_is_busy() {
  local run_id jobs active_runs
  active_runs="$(gh run list \
    --status in_progress \
    --limit 100 \
    --json databaseId \
    --jq '.[].databaseId')" || {
    echo "::warning::could not list active runs; treating the runner as not busy"
    return 1
  }
  while IFS= read -r run_id; do
    [ -n "$run_id" ] || continue
    if ! jobs="$(gh api \
      "repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}/jobs?per_page=100")"; then
      echo "::warning::could not list jobs for run $run_id"
      continue
    fi
    if jq -e --arg runner "$AWS_RUNNER_NAME" --arg group "$AWS_RUNNER_GROUP" '
      [.jobs[]
       | select(.status == "in_progress")
       | select(.runner_name == $runner or .runner_group_name == $group)]
      | length > 0
    ' <<<"$jobs" >/dev/null; then
      return 0
    fi
  done <<< "$active_runs"
  return 1
}

# Read the monitored job's status into `status`, and the resolver's exit code
# into `resolve_rc`. Its diagnostic is held in `resolve_error` so that the one
# tolerated absence below stays a retry rather than an annotation.
resolve_status() {
  local err_file
  err_file="$(mktemp)"
  resolve_rc=0
  status="$("$MONITORED_JOB" 2>"$err_file")" || resolve_rc=$?
  resolve_error="$(cat "$err_file")"
  rm -f "$err_file"
}

# A run's job list is read a moment after the run is created, before the jobs
# API has necessarily caught up, so a single absence is confirmed once more
# before it is treated as a wrong job name.
missing_confirmed=false
while [ "$(date +%s)" -lt "$READY_DEADLINE" ]; do
  resolve_status
  case "$resolve_rc" in
    0)
      case "$status" in
        in_progress|completed)
          echo "the readiness job is $status; the self-hosted runner picked it up"
          exit 0
          ;;
        queued|waiting|requested|pending)
          if runner_is_busy; then
            echo "the readiness job is $status, but the AWS runner is busy; leaving it queued behind the active job"
            exit 0
          fi
          sleep 10
          ;;
        *)
          echo "the readiness job is $status; the self-hosted runner picked it up"
          exit 0
          ;;
      esac
      ;;
    2)
      if [ "$missing_confirmed" = "true" ]; then
        printf '%s\n' "$resolve_error" >&2
        exit 1
      fi
      echo "the readiness job is not in the job list yet; checking once more"
      missing_confirmed=true
      sleep 10
      ;;
    3)
      printf '%s\n' "$resolve_error" >&2
      exit 1
      ;;
    *)
      echo "::warning::could not read the readiness job's status; retrying"
      sleep 10
      ;;
  esac
done

# Re-check after the last sleep; status may have changed.
resolve_status
case "$resolve_rc" in
  0)
    if [ "$status" = "in_progress" ] || [ "$status" = "completed" ]; then
      echo "the readiness job is $status; the self-hosted runner picked it up"
      exit 0
    fi
    ;;
  2|3)
    # Absent or ambiguous: the resolver already named the job and the reason.
    printf '%s\n' "$resolve_error" >&2
    exit 1
    ;;
  *)
    # Unreadable at the deadline.
    status="unknown"
    ;;
esac
if runner_is_busy; then
  echo "::notice::the readiness job is $status at the deadline, but the AWS runner is busy; leaving it queued"
  exit 0
fi

echo "::error::the self-hosted runner did not pick up the readiness job before the deadline"
# `cleanup-aws` in the caller clears a still-queued job, looking it up by that
# same name through `scripts/ci/monitored-job.sh`.
exit 1
