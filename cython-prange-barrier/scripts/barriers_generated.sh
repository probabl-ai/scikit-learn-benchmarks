#!/bin/bash
# Run under benchlock. Barrier calls per prange function, in the C generated from probes/warn.pyx, per clause variant.
# usage: barriers_generated.sh WARN_DIR (output dir of warn_matrix.sh)
cd "$1"
count() { objdump -dr "$1" | awk '/^[0-9a-f]+ <.*>:$/ {f=$2; gsub(/[<>:]/,"",f); sub(/^__pyx_pf_4warn_[0-9]*/,"",f)} /GOMP_barrier|__kmpc_barrier/ {c[f]++} END {for(f in c) print c[f], f}' | sort -k2 | tr '\n' ';'; echo; }
for cc in "gcc_linux-64=15 x86_64-conda-linux-gnu-gcc" "clang=21 clang"; do
  spec=${cc% *}; bin=${cc#* }
  for src in warn warn_nofirst warn_private; do
    pixi exec -c conda-forge -s python=3.13 -s llvm-openmp -s $spec -- bash -c "\$0 -O2 -w -fopenmp -fPIC -I\$(python -c 'import sysconfig;print(sysconfig.get_paths()[\"include\"])') -c $src.c -o $src.${spec%%[= ]*}.o" "$bin" 2>&1 | tail -2
    printf '%-14s %-14s ' "${spec%%[= ]*}" "$src"; count "$src.${spec%%[= ]*}.o"
  done
done
