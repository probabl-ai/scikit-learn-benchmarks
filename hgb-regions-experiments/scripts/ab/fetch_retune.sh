#!/bin/bash
# Fetch the jsonl of Exp 22-27 from each machine to results/MACHINE/expN/.
# usage: fetch_retune.sh (from hgb-regions-experiments)
for pair in intel-xeon:xeon-s0 intel-gamer-gpu:gamer intel-laptop:laptop mac-m2:mac-m2; do
  host=${pair%:*} dir=${pair#*:}
  ssh -o ConnectTimeout=8 "$host" 'cd ~/work/hgb-regions && tar c exp2[2-7]*.jsonl' | tar x -C "$(mktemp -d /tmp/claude-1000/fetch.XXXX)" --to-command='
    n=$TAR_FILENAME; e=${n%%-*}; mkdir -p results/'"$dir"'/$e; cat > results/'"$dir"'/$e/$n'
done
