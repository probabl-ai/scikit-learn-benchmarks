#!/bin/bash
# Calls to GOMP_barrier in each compiled extension, per build. Prints only modules where builds differ.
# usage: barrier_sites.sh BUILD...   on a host with ~/work/cython-prange/skl-BUILD/build
cd ~/work/cython-prange
tmp=$(mktemp -d)
for b in "$@"; do
  (cd "skl-$b/build" && find sklearn -name '*.so' | sort) | while read -r so; do
    n=$(objdump -d "skl-$b/build/$so" | grep -c 'call.*GOMP_barrier')
    echo "${so%%.cpython*} $n"
  done > "$tmp/$b"
done
first=$1; shift
echo "module $first $*"
while read -r mod n; do
  row="$mod $n"; same=1
  for b in "$@"; do
    m=$(grep -m1 "^$mod " "$tmp/$b" | cut -d' ' -f2)
    row="$row ${m:-NA}"
    [ "$m" = "$n" ] || same=0
  done
  [ "$same" = 1 ] || echo "$row"
done < "$tmp/$first"
echo "total per build:"
for b in "$first" "$@"; do
  echo "$b $(awk '{s+=$2} END {print s}' "$tmp/$b")"
done
