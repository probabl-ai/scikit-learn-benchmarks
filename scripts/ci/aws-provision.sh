#!/usr/bin/env bash
# Provision the AWS GPU runner for a benchmark run.
#
# AWS credentials are configured by the caller; this script starts the
# instance and waits for the `runner-ready` job in the caller workflow.
set -euo pipefail

: "${INSTANCE_ID:?INSTANCE_ID is not set}"
: "${REGION:?REGION is not set}"
: "${GH_TOKEN:?GH_TOKEN is not set}"
: "${AWS_RUNNER_NAME:?AWS_RUNNER_NAME is not set}"
: "${AWS_RUNNER_GROUP:?AWS_RUNNER_GROUP is not set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is not set}"

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

# `runner-ready` can stay queued if the AWS runner is busy with another
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

while [ "$(date +%s)" -lt "$READY_DEADLINE" ]; do
  status="$(gh api \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
    --jq '[.jobs[] | select(.name == "runner-ready")][0].status // "missing"')" \
    || status="unknown"
  case "$status" in
    in_progress|completed)
      echo "runner-ready is $status; the self-hosted runner picked it up"
      exit 0
      ;;
    queued|missing|waiting|requested|pending)
      if runner_is_busy; then
        echo "runner-ready is $status, but the AWS runner is busy; leaving it queued behind the active job"
        exit 0
      fi
      sleep 10
      ;;
    unknown)
      echo "::warning::could not read runner-ready status; retrying"
      sleep 10
      ;;
    *)
      echo "runner-ready is $status; the self-hosted runner picked it up"
      exit 0
      ;;
  esac
done

# Re-check after the last sleep; status may have changed.
status="$(gh api \
  "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
  --jq '[.jobs[] | select(.name == "runner-ready")][0].status // "missing"')" \
  || status="unknown"
if [ "$status" = "in_progress" ] || [ "$status" = "completed" ]; then
  echo "runner-ready is $status; the self-hosted runner picked it up"
  exit 0
fi
if runner_is_busy; then
  echo "::notice::runner-ready is $status at the deadline, but the AWS runner is busy; leaving it queued"
  exit 0
fi

echo "::error::the self-hosted runner did not pick up runner-ready before the deadline"
# `cleanup-aws` in the caller clears a still-queued runner-ready job.
exit 1
