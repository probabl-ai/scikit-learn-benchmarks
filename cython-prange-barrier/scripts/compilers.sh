#!/bin/bash
# Run under benchlock. Compile probes/clauses.c with many compilers and count barriers per function.
# usage: compilers.sh OUTDIR
HERE=$(cd "$(dirname "$0")/.." && pwd); out=$1; mkdir -p "$out"; cd "$out"
count() { # object file: "func barriers" for each function
  objdump -dr "$1" | awk '/^[0-9a-f]+ <.*>:$/ {f=$2; gsub(/[<>:]/,"",f); order[++n]=f; c[f]=0}
    /GOMP_barrier|__kmpc_barrier/ {c[f]++} END {for(i=1;i<=n;i++) printf "%s=%d ", order[i], c[order[i]]; print ""}'
}
run() { # label, pixi specs, compiler
  local label=$1 spec=$2 cc=$3
  local args=(); for s in $spec; do args+=(-s "$s"); done
  local cmd="$cc -O2 -fopenmp -c $HERE/probes/clauses.c -o $label.o && $cc --version | head -1"
  if [ -z "$spec" ]; then v=$(bash -c "$cmd" 2>&1 | tail -1); else v=$(pixi exec -c conda-forge "${args[@]}" -- bash -c "$cmd" 2>&1 | tail -1); fi
  if [ -f "$label.o" ]; then printf '%-10s %-55s %s\n' "$label" "$v" "$(count $label.o)"; else printf '%-10s FAILED: %s\n' "$label" "$v"; fi
}
run sys-gcc13 "" gcc
for v in 7 8 9 11 13 14 15; do run gcc$v "gcc_linux-64=$v" x86_64-conda-linux-gnu-gcc; done
for v in 11 14 18 20 21; do run clang$v "clang=$v" clang; done
