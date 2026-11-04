#!/bin/bash
# Exp 24-25 on xeon-s1: build the split-K branch, check it, scan thread
# counts of the shapes PR #4 speeds up or slows down, then split-K vs develop.
set -uo pipefail
cd "$(dirname "$0")/.."
git -C openblas-fork fetch -q origin splitk
test -d openblas-sk || git -C openblas-fork worktree add -q --detach ../openblas-sk origin/splitk
test -f builds/sk/lib/libopenblas.so.0 || scripts/build_openblas.sh openblas-sk builds/sk
{
  scripts/check_patches.sh builds/sk
  # split path at odd thread counts and with beta
  for t in 3 43 86; do for dt in d s; do for beta in 0 0.5; do
    OPENBLAS_NUM_THREADS=$t build/gemmbench --lib builds/sk/lib/libopenblas.so.0 --label sk \
      --dtype $dt --beta $beta --check 1 100x100x200000:NT 37x45x100000:TN 300x7x60000:TT \
      syrk:100x200000:N syrk:37x200000:T 2>/dev/null | sed "s|^|sk T=$t |"
  done; done; done
} > results/xeon-s1/check_splitk.txt 2>&1
grep -c OK results/xeon-s1/check_splitk.txt; grep -v OK results/xeon-s1/check_splitk.txt | head
# Exp 25: where does PR #4 (sr-fix) gain or lose, at which thread counts?
S=(256x256x256 512x512x512 1024x1024x1024 2000x2000x200 200x200x50000:TN 256x100000x50:TN
   1000x1000x1000 syrk:990x463810:N syrk:500x100000:N)
pixi run python scripts/sweep.py --out results/xeon-s1/switch_ratio_scan.csv \
  --threads 8,12,16,22,24,32,43,48,64,86 \
  --variant "sr-base=builds/sr-base/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --variant "sr-fix=builds/sr-fix/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --variant "sk=builds/sk/lib/libopenblas.so.0,OPENBLAS_NUM_THREADS={T}" \
  --dtype d --rounds 2 --min-time 0.2 --pin compact --note "exp25 switch ratio thread scan" \
  --shapes "${S[@]}"
scripts/run_splitk.sh xeon-s1 8,16,24,43,64,86
