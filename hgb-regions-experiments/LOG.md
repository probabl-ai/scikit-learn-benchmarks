# Experiment log

All runs on `intel-gamer-gpu` (Core Ultra 9 285: 8 P cores = CPUs 0-7, 16 E
cores = CPUs 8-23, 64 GB) unless stated, under `benchlock`, times UTC. The
host env makes OpenMP passive (`OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=1`,
`KMP_BLOCKTIME=0`). Idle states (`intel_idle`, `menu` governor): POLL, C1
(exit 1 us), C2 (exit 127 us, target residency 381 us), C3 (exit 1048 us,
target 3144 us).

The `year_prediction_msd` case is `scripts/gamer-msd/case.json`:
HistGradientBoostingRegressor, 464K x 90, `max_iter=100`,
`max_leaf_nodes=127`, `min_samples_leaf=5`, `early_stopping=False`.
`patched_runner.py` forces the HGB team size with `HGB_N_THREADS` and prints
the OpenMP pools with `SHOW_POOLS=1`.

## Exp 1: thread sweep, PR vs base (2026-11-03 15:50-16:40)

`scripts/gamer-msd/run_msd.sh` (sklearn-src at the PR or at its base,
rebuilt between). Results in `results/gamer/msd/{gomp_main,gomp_pr}.log`.
`libomp_main.log` is **libgomp too**: calling the libomp env's python
directly loads `sklearn-dev/lib/libgomp.so.1` through the shared build's
RPATH. Only `pixi run -e sklearn-dev-libomp` gets libomp.

Fit time (s), passive, `OMP_NUM_THREADS=24`, team size forced:

| team | base | PR |
| --- | --- | --- |
| 16 | 4.3-4.8 | 4.2-5.0 |
| 20 | 6.8-8.4 | 14.4-17.2 |
| 22 | 7.1-9.0 | 18.5-20.1 |
| 23 | 9.0-15.2 | 20.8-21.4 (hist 17.4 s) |
| 24 (base default) | 11.6-13.5 | |
| 23, spinning (`GOMP_SPINCOUNT=300000`) | 1.8 | 1.7-2.8 |

The PR's default team is 23 (ceil of 90 features over 24 threads). The cliff
between 18 and 20 threads matches its passive `find_node_split` guard
(serial when `n_split_candidates < 5 * n_threads`, 90 / 5 = 18). Spinning
makes both ~10x faster than passive.

## Exp 2: PR wait-policy override, both runtimes (16:25-16:37)

`scripts/gamer-msd/run_msd2.sh`, through `pixi run -e <env>`
(`results/gamer/msd/msd2.log`). PR, 23 threads, median fit:

| variant | libgomp | libomp |
| --- | --- | --- |
| default (serial find_split, serial split_indices < 115K samples) | 17.2 s (hist 13.6) | 16.5 s |
| `SKLEARN_OMP_WAIT_POLICY=active` (both guards off) | 5.1 s (hist 2.6) | 4.9 s |
| `HGB_N_THREADS=18` (find_split parallel again) | 5.4 s | 5.7 s |

Base at 24 threads in the dashboard run: 12.5 s (libgomp), 6.0 s (libomp).
The serial phases make the *parallel* histogram phase slower.

## Exp 3: OS counters of one fit (17:27)

`scripts/gamer-msd/counters.sh`: `/usr/bin/time` plus C3 entries summed over
CPUs (`results/gamer/session_outputs.md`). PR, 3 fits per process:

| | C3 entries | voluntary ctx switches | fits |
| --- | --- | --- | --- |
| default | 3.6-3.7 M | 7.4 M | ~16.5 s |
| policy active | 0.55-0.63 M | 12.9 M | ~5.2 s |

The fast variant switches context *more* often but enters C3 6x less.

## Exp 4: wake-up microbenchmarks (17:00-17:30)

`src/wake.c` (compute-only items) and `src/wake2.c` (each item gathers
random cache lines from its own persistent buffer, so cache loss shows up as
busy time). 23 threads, 4 items of 32 KB per thread, a serial busy gap
between regions (`results/gamer/session_outputs.md`):

| gap | region | wake (max over threads) | C3 per iteration |
| --- | --- | --- | --- |
| 0 us | 24 us | 12 us | 3 |
| 160 us | 233 us | 214 us | 35 |
| 2000 us | 353 us | | 45 |
| spinning, any gap | 6-7 us | | 0 |

Busy time grows ~3x after long gaps; no thread migrations.

## Exp 5: C3 disabled by hand (17:30-17:40)

The user ran `cpupower idle-set -d 3`. `scripts/gamer-msd/counters_c3off.sh`
and `wake2` again:

- PR default: 18.8-20.8 s (C2 entries 6.0-6.1 M), PR policy active: 24.7-31.7 s
  (C2 7.7-8.4 M, sys time 31 s). Both slower than with C3.
- wake2 passive, gap 0: region 380-393 us, wake ~340 us (12 us with C3
  on). The `menu` governor then picks C2 even for short idles. Spinning
  unchanged.

Confounded test: it made every wake expensive.

## Exp 6: only C1 vs all idle states (20:07-20:11)

`scripts/gamer-msd/cstates_ab.sh`: `cpupower idle-set -D 2` (only POLL and
C1) vs `-E`, interleaved twice in one benchlock call, restores C3-disabled
after. `results/gamer/msd/cstates_ab.log`.

| PR, 23 threads | only C1 | all states |
| --- | --- | --- |
| default (serial find_split) | 4.1-4.2 s | 16.3-18.4 s (C2+C3 entries 4.7 M per 3 fits) |
| policy active (parallel find_split) | 3.3-3.4 s | 4.7-5.4 s (0.7 M) |

wake2, wake of the whole team after a gap: 12 us at 0-160 us gaps and
20 us at 500 us with only C1, vs 16-29 us, 187-204 us and 158 us with all
states. Deep idle states explain the regression; the PR's serial
find_split still loses with C1 only (4.2 vs 3.3 s). C3 re-enabled with
`cpupower idle-set -E` afterwards.

## Exp 7: changing the team size between regions (20:24-20:26)

`src/team.c`: regions alternating 23 and 4 (or 18) threads, 5 us of work
per thread, 2000 regions. libomp loaded with `LD_PRELOAD` (the binary's
RPATH pins libgomp). `results/gamer/session_outputs.md`.

| | 23 constant | 23/4 alternating: 23-thread region | distinct thread ids |
| --- | --- | --- | --- |
| libgomp passive | 23-37 us | 287-290 us | 23 vs 19004 |
| libgomp spin | 7 us | 290 us | 23 vs 19004 |
| libomp passive | 20 us | 19 us | 23 vs 23 |
| libomp spin | 6 us | 13 us | 23 vs 23 |

libgomp makes the threads beyond a smaller team exit, and growing the team
again `pthread_create`s them (~13 us each). 23/18 alternating: 19 new
threads become 5 (5018 ids), 54-70 us. libomp keeps them.

## Exp 8: two regions vs one region with a barrier (20:44-20:50)

`src/barrier.c`, `scripts/microbench/run_barrier.sh`, 23 threads, 4 items
per thread and phase, 2 runs (`results/gamer/microbench/barrier_run{1,2}.txt`).
Overhead per iteration above the pure compute time, median:

| | spin | passive, back-to-back | passive, 30-300 us serial gap between phases |
| --- | --- | --- | --- |
| 2 regions | 2-4 us | 26-63 us | 330-730 us (one run: 38 us) |
| 1 region + barrier | 2-3 us | 25-29 us | 25-29 us (the gap moves inside) |
| 1 region, no barrier | 1-2 us | 13-22 us | 13-22 us |

Passive numbers are bimodal between runs. One reproducible oddity: libgomp
passive, 10 us items, 30 us outer gap, the single region without barrier
costs 257-311 us.

## Exp 9: scikit-learn variants (2026-11-03 20:15-21:00, local thinkpad)

Branch `cakedev0/scikit-learn@hgb/fewer-regions`:

- 873fa541: `SKLEARN_HGB_GUARD` (`pr`/`abs`/`none`) and `fuse`: split_indices
  2 regions -> 1 (barrier between partition and copy-back, each thread
  computes its offsets), children histograms 3 -> 1 (gather, barrier,
  brute + subtraction per feature), both children `find_node_split` 2 -> 1.
  7 -> 3 regions per split.
- f8698ca7: `fuse2` merges histograms and split finding of both children
  (`histogram.pxd` so `splitting.pyx` can call the per-feature histogram
  code): 2 regions per split. `fuse3` also gathers the smallest child's
  gradients in split_indices' copy-back: 2 regions, 1 barrier per split.

`scripts/ab/check_identical.py` (8 threads, passive, forced team size):
every variant gives bit-identical trees and predictions to the PR on 5
cases (missing values, categorical, monotonic and interaction constraints,
`max_features < 1`, multiclass). HGB test suite with `fuse`/`fuse3` forced
on (thresholds of 1, so every threaded path runs): 365 passed, 105 skipped.

## Exp 10: interleaved A/B on 3 machines (started 20:55)

`scripts/ab/run_all.sh` + `bench.py`: the 15 `hgb_scalability.py` cases
(9 synthetic, 6 real) at the max thread count, 3 fits per process,
variants shuffled per case. Base = d1278c27 (the PR's merge base).

| host | slot | threads | profiles (variants) |
| --- | --- | --- | --- |
| intel-gamer-gpu | all | 24 | libgomp, libomp (`LD_PRELOAD`) x all 7 |
| intel-laptop (Core Ultra X7 358H, 16 cores) | all | 16 | libgomp x all 7 |
| intel-xeon | s0 (one socket) | 86 | default = active wait detected (main, pr, pr+fuse, pr+fuse3); `OMP_WAIT_POLICY=passive` x all 7 |

Variants: `main`, `pr`, `abs`, `pr+fuse`, `pr+fuse3`, `abs+fuse3`,
`none+fuse3`. `abs` thresholds untuned: 20K samples for split_indices, 4000
bins (~16 features) for find_node_split.

First launch at 20:55 on f8698ca7, stopped at 21:05: the fused methods
added ~15 us of Python overhead per split (feature-slot arrays allocated per
call, `np.asarray` copies, long argument lists), which doubled the time of
the 1-thread cases (XS 0.109 vs 0.054 s on gamer). Partial results kept as
`results/<host>/ab/*_v1.jsonl`.

9ade78bf: no fused paths when the team has 1 thread (nothing to merge),
slot arrays allocated once per Splitter, memoryviews returned as in the
original methods. Identity check: all 55 variant fits identical. The HGB
tests mostly fit on 1 thread, so they no longer exercise the fused code;
the identity check at 8 threads does.

Relaunched at 21:08 on 9ade78bf. Results in `results/<host>/ab/`, tables
from `scripts/ab/analyze.py` (one process of 3 fits per case and variant;
the cases that get 1 thread run identical code in every variant, so their
spread is noise: up to ~15% on gamer, where a 1-thread process lands on a P
or an E core).

### gamer-gomp: median fit (s), and in parentheses time / pr

| case | main | pr | abs | pr+fuse | pr+fuse3 | abs+fuse3 | none+fuse3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XS | 1.4 (25.51) | 0.0549 | 0.0546 (0.99) | 0.055 (1.00) | 0.0549 (1.00) | 0.0541 (0.99) | 0.0545 (0.99) |
| S-thin | 0.995 (12.83) | 0.0775 | 0.0784 (1.01) | 0.0786 (1.01) | 0.0886 (1.14) | 0.0879 (1.13) | 0.0789 (1.02) |
| S | 0.645 (2.23) | 0.29 | 0.292 (1.01) | 0.288 (1.00) | 0.289 (1.00) | 0.288 (0.99) | 0.289 (1.00) |
| M | 1.19 (3.23) | 0.368 | 0.367 (1.00) | 0.368 (1.00) | 0.37 (1.01) | 0.366 (0.99) | 0.367 (1.00) |
| M-very-thin | 2.94 (19.93) | 0.147 | 0.145 (0.98) | 0.145 (0.98) | 0.145 (0.99) | 0.145 (0.99) | 0.153 (1.04) |
| L-thin | 9.91 (12.61) | 0.786 | 0.78 (0.99) | 0.753 (0.96) | 0.696 (0.89) | 0.735 (0.93) | 0.743 (0.95) |
| L | 7.56 (8.15) | 0.927 | 0.91 (0.98) | 0.934 (1.01) | 0.919 (0.99) | 0.836 (0.90) | 0.882 (0.95) |
| L-stumps | 1.22 (1.10) | 1.11 | 1.15 (1.04) | 1.1 (0.99) | 1.1 (0.99) | 1.13 (1.02) | 1.15 (1.04) |
| L-leaf255 | 7.44 (8.44) | 0.881 | 0.816 (0.93) | 0.772 (0.88) | 0.678 (0.77) | 0.669 (0.76) | 0.743 (0.84) |
| ames_housing | 1.18 (6.67) | 0.178 | 0.177 (0.99) | 0.18 (1.01) | 0.181 (1.02) | 0.179 (1.01) | 0.177 (1.00) |
| kddcup09_churn | 6 (3.11) | 1.93 | 1.85 (0.96) | 1.97 (1.02) | 1.96 (1.01) | 1.86 (0.96) | 1.86 (0.96) |
| amazon_employee_access | 3.81 (11.77) | 0.324 | 0.323 (1.00) | 0.324 (1.00) | 0.324 (1.00) | 0.322 (0.99) | 0.324 (1.00) |
| covtype | 168 (19.22) | 8.75 | 8.61 (0.98) | 7.41 (0.85) | 6.4 (0.73) | 6.72 (0.77) | 6.66 (0.76) |
| susy | 24.1 (1.48) | 16.3 | 12.1 (0.75) | 13.7 (0.84) | 13.3 (0.82) | 13.2 (0.81) | 13.7 (0.84) |
| year_prediction_msd | 12.8 (0.61) | 21 | 10.5 (0.50) | 13 (0.62) | 7.82 (0.37) | 8.46 (0.40) | 7.92 (0.38) |
| geomean / pr | 5.65 | 1 | 0.93 | 0.94 | 0.89 | 0.89 | 0.90 |

### laptop-gomp: median fit (s), and in parentheses time / pr

| case | main | pr | abs | pr+fuse | pr+fuse3 | abs+fuse3 | none+fuse3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XS | 1.57 (23.49) | 0.0668 | 0.0673 (1.01) | 0.0674 (1.01) | 0.068 (1.02) | 0.0679 (1.02) | 0.0676 (1.01) |
| S-thin | 1.85 (19.30) | 0.0957 | 0.0959 (1.00) | 0.0967 (1.01) | 0.0956 (1.00) | 0.0956 (1.00) | 0.0977 (1.02) |
| S | 2.2 (5.61) | 0.392 | 0.395 (1.01) | 0.393 (1.01) | 0.376 (0.96) | 0.397 (1.01) | 0.393 (1.00) |
| M | 2.25 (4.91) | 0.459 | 0.455 (0.99) | 0.456 (0.99) | 0.457 (1.00) | 0.457 (1.00) | 0.456 (0.99) |
| M-very-thin | 1.97 (11.02) | 0.179 | 0.177 (0.99) | 0.178 (1.00) | 0.177 (0.99) | 0.179 (1.00) | 0.179 (1.00) |
| L-thin | 2.77 (2.70) | 1.02 | 1.06 (1.03) | 1.02 (1.00) | 0.906 (0.88) | 0.867 (0.85) | 0.922 (0.90) |
| L | 3.03 (2.38) | 1.27 | 1.24 (0.97) | 1.18 (0.92) | 1.23 (0.97) | 1.15 (0.90) | 1.13 (0.89) |
| L-stumps | 2.03 (1.07) | 1.9 | 1.78 (0.93) | 1.89 (0.99) | 1.89 (0.99) | 1.77 (0.93) | 1.78 (0.94) |
| L-leaf255 | 4.48 (3.70) | 1.21 | 1 (0.83) | 1.17 (0.97) | 0.93 (0.77) | 0.917 (0.76) | 1.09 (0.90) |
| ames_housing | 1.79 (7.35) | 0.243 | 0.242 (1.00) | 0.242 (1.00) | 0.242 (1.00) | 0.244 (1.00) | 0.244 (1.00) |
| kddcup09_churn | 3.71 (1.63) | 2.27 | 2.36 (1.04) | 2.13 (0.94) | 2.03 (0.89) | 2.01 (0.89) | 2.01 (0.89) |
| amazon_employee_access | 3.24 (7.98) | 0.406 | 0.402 (0.99) | 0.402 (0.99) | 0.401 (0.99) | 0.401 (0.99) | 0.402 (0.99) |
| covtype | 64.6 (5.71) | 11.3 | 12 (1.06) | 9.15 (0.81) | 8.06 (0.71) | 7.65 (0.68) | 9.29 (0.82) |
| susy | 15.1 (1.09) | 13.8 | 13.6 (0.98) | 13.7 (0.99) | 13.4 (0.97) | 13 (0.94) | 11.9 (0.86) |
| year_prediction_msd | 13.5 (1.07) | 12.6 | 11.8 (0.94) | 8.55 (0.68) | 4.59 (0.36) | 4.3 (0.34) | 7.55 (0.60) |
| geomean / pr | 4.20 | 1 | 0.98 | 0.95 | 0.88 | 0.86 | 0.91 |

### xeon-s0-default: median fit (s), and in parentheses time / pr

| case | main | pr | pr+fuse | pr+fuse3 |
| --- | --- | --- | --- | --- |
| XS | 0.649 (7.22) | 0.0899 | 0.0901 (1.00) | 0.0891 (0.99) |
| S-thin | 0.767 (3.30) | 0.232 | 0.218 (0.94) | 0.158 (0.68) |
| S | 0.927 (1.44) | 0.642 | 0.542 (0.85) | 0.423 (0.66) |
| M | 0.933 (1.28) | 0.726 | 0.514 (0.71) | 0.445 (0.61) |
| M-very-thin | 0.786 (2.54) | 0.31 | 0.317 (1.02) | 0.269 (0.87) |
| L-thin | 1.22 (1.16) | 1.05 | 1.01 (0.96) | 1.03 (0.98) |
| L | 1.16 (1.00) | 1.17 | 0.985 (0.84) | 0.906 (0.78) |
| L-stumps | 1.09 (1.11) | 0.977 | 0.901 (0.92) | 0.95 (0.97) |
| L-leaf255 | 1.49 (1.01) | 1.47 | 1.12 (0.76) | 0.918 (0.62) |
| ames_housing | 0.811 (1.59) | 0.51 | 0.384 (0.75) | 0.3 (0.59) |
| kddcup09_churn | 1.16 (1.01) | 1.15 | 1.01 (0.88) | 0.882 (0.77) |
| amazon_employee_access | 1.09 (2.29) | 0.476 | 0.506 (1.06) | 0.395 (0.83) |
| covtype | 17.2 (1.31) | 13.1 | 11.9 (0.91) | 9.39 (0.72) |
| susy | 7.1 (1.11) | 6.4 | 7.06 (1.10) | 6.53 (1.02) |
| year_prediction_msd | 3.48 (0.86) | 4.07 | 3.46 (0.85) | 2.39 (0.59) |
| geomean / pr | 1.54 | 1 | 0.90 | 0.76 |

Notes:
- xeon-s0-default (active wait detected): fuse3 helps even when spinning,
  0.76 geomean, 0.59-0.78 on 10 of 15 cases. The PR is slower than main on
  msd there (4.07 vs 3.48 s); fuse3 gets 2.39 s.
- laptop: abs+fuse3 makes msd 3x faster than both main and the PR (4.3 vs
  13.5 and 12.6 s). With fuse3, split finding lives in the histogram region,
  so only the split_indices guard remains, and it still matters: none+fuse3
  (always threaded) gets 7.6 s.
- gamer libgomp: abs alone halves msd (21 -> 10.5 s) by keeping find_split
  parallel; fuse3 gets 7.8 s, still 2x the C1-only time of Exp 6.

Pending: gamer-omp, xeon-s0-passive.

## Side: segfault reproduction (2026-11-03 ~15:00)

A `sklearn-dev-libomp` HGB run segfaulted on gamer once. `scripts/segv/repro.sh`
reran the case 50x at 8 threads and 15x at 2/4/16/24 threads (110 runs): no
crash, one 300 s timeout (`results/gamer/segv/fail_T24_11.err`). The core's
stack (`results/gamer/segv/bt.txt`, from `coredumpctl info`) is in the
CPython parser (`_PyPegen_expect_token`) during an import. An earlier
sklearnex RF crash on the same host was also inside CPython
(`_PyOpcode_num_pushed`). Likely hardware instability (CPU at 92-99 C under
load); a memtest was suggested. `repro.sh` called the env python directly,
so it ran libgomp, not libomp.

## Exp 11: hardware C-state residency with turbostat (2026-11-03 ~22:50)

`scripts/gamer-msd/turbo.sh`: `turbostat -q -S` around one msd fit (one
process, data loading included in the interval), `pr` vs `abs+fuse3`
(9ade78bf), interleaved twice, all idle states enabled. Averages over the
24 CPUs:

| | fit | Busy% | Bzy_MHz | requested C1 / C2 / C3 (% time) | hardware CPU%c1 / c6 / c7 | Pkg%pc* | PkgWatt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pr | 19.7-19.9 s | 11.8 | 3410-3476 | 20 / 12.5 / 56 | 63 / 6.4-6.9 / 17-18 | 0 | 36-37 |
| abs+fuse3 | 8.1-8.2 s | 18.5 | 4101-4166 | 19 / 13 / 50 | 56 / 8.4-8.6 / 16 | 0 | 52-53 |

- The cores do reach core C6 and C7 (~24% of the time), so the states are
  real hardware states. But most of the time the OS spent requesting C3
  (MWAIT 0x60) ended in hardware C1 or C6: the hardware demotes requests.
  MWAIT 0x60 asks for core C7 and allows package C10; the package never left
  C0 (one core is always busy), so no package C-state.
- The busy frequency is ~20% lower with the PR's serial phases (3.4 vs
  4.1 GHz): woken cores run slower for a while, which matches the longer
  busy times after long gaps in `wake2`.
- Energy over the interval: ~770 J (pr) vs ~505 J (abs+fuse3).
