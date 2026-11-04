#!/bin/bash
# Set up the positive control in a fresh directory on this machine. usage: setup_control.sh DIR
set -e
here=$(cd "$(dirname "$0")" && pwd); dir=$1
mkdir -p "$dir/tsan" "$dir/cy"
for pair in 331=3.3.1 fixA=prange-fix-end-barrier fixB2=prange-no-start-barrier; do
  n=${pair%%=*}; rev=${pair#*=}
  mkdir -p "$dir/cy/$n"
  git -C ~/cython-prange archive --format=tar -o "$dir/cy/$n.tar" "$rev"
  tar -xf "$dir/cy/$n.tar" -C "$dir/cy/$n"
done
cp "$here/pixi.toml" "$here/probe.sh" "$here/../../probes/race.pyx" "$dir/tsan/"
