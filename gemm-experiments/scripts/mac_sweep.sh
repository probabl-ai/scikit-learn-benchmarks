#!/bin/bash
# Run a benchmark command under the machine lock on mac-m2, after waiting
# (up to 10 min) for background daemons (Spotlight mds, FileProvider...) to
# drop under $QUIET %CPU (default 150: a Spotlight daemon of another user
# keeps one E core busy for hours), and sample the top CPU users every 10 s.
#   scripts/mac_sweep.sh NAME [sweep.py args...]
#   CMD="python scripts/mac_numpy_bench.py" scripts/mac_sweep.sh NAME [args...]
# writes results/mac-m2/NAME.{csv,meta.json,log,top}. Set PREFIX_CMD (e.g.
# "taskpolicy -b") to run gemmbench under a QoS wrapper.
set -u
cd "$(dirname "$0")/.."
name=$1; shift
out=results/mac-m2/$name
mkdir -p results/mac-m2
export CMD=${CMD:-python scripts/sweep.py}
{
  echo "# start $(date) $(uptime)"
  benchlock --tag "gemm mac: $name" all -- bash -c '
    busy() { top -l 2 -s 2 -stats cpu | awk "/^%CPU/{n++; next} n==2 {s+=\$1} END {print int(s)}"; }
    for i in $(seq 60); do
      b=$(busy); echo "# background %CPU: $b"; [ "$b" -lt ${QUIET:-150} ] && break; sleep 8
    done
    top -l 2 -s 2 -o cpu -n 6 -stats pid,command,cpu | tail -7 | sed "s/^/# /"
    echo "# locked $(date) $(uptime)"
    top -l 0 -s 10 -o cpu -n 4 -stats command,cpu > "$0.top" & tp=$!
    # shell-hook, not `pixi run`: its task shell would expand e.g. $NEWLAPACK.
    eval "$(pixi shell-hook)"
    $CMD --out "$0.csv" ${PREFIX_CMD:+--prefix-cmd "$PREFIX_CMD"} "$@"
    rc=$?; kill $tp; exit $rc
  ' "$out" "$@"
  echo "# exit $? $(date) $(uptime)"
  pmset -g therm 2>&1 | sed 's/^/# /'
} > "$out.log" 2>&1
