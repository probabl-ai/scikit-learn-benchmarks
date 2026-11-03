#!/bin/bash
echo "C3 disable flags: $(cat /sys/devices/system/cpu/cpu*/cpuidle/state3/disable | sort | uniq -c | tr -s " ")"
~/work/gamer-run/msd/counters_c3off.sh
cd ~/work/gamer-run/wake
echo "== wake2 libgomp passive, C3 off"
OMP_NUM_THREADS=23 ./wake2 1000 32 4 0 160 500 2000
echo "== c3off done $(date -Is)"
