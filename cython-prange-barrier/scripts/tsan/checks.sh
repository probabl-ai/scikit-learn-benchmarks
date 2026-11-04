#!/bin/bash
# Read-only, on the host with the TSan builds: are the extensions instrumented, did the values change?
W=$(cd "$(dirname "$0")" && pwd); cd "$W"
for b in m331 mA mB fB; do
  echo "== instrumented extensions in $b"
  for m in cluster/_k_means_lloyd metrics/_pairwise_distances_reduction/_argkmin manifold/_barnes_hut_tsne _loss/_loss ensemble/_hist_gradient_boosting/histogram; do
    so=$(ls venv-$b/lib/python3.13/site-packages/sklearn/$m*.so)
    echo "$m tsan_symbols=$(nm -D "$so" | grep -c __tsan_)"
  done
done
for b in mA mB fB; do
  echo "== values, m331 vs $b"
  grep -E '^[a-z_]+: ' logs/out-m331.txt > logs/values-m331.txt
  grep -E '^[a-z_]+: ' logs/out-$b.txt > logs/values-$b.txt
  wc -l < logs/values-$b.txt
  diff logs/values-m331.txt logs/values-$b.txt && echo identical
done
