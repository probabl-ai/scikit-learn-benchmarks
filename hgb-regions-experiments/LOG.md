# Experiment log

All runs on `intel-gamer-gpu` (Core Ultra 9 285: 8 P cores = CPUs 0-7, 16 E
cores = CPUs 8-23, 64 GB) unless stated, under `benchlock`, times UTC. OpenMP is
passive there with no variable set: the runtimes report `GOMP_SPINCOUNT=1` and
`KMP_BLOCKTIME=0` by themselves (Exp 12). Idle states (`intel_idle`, `menu` governor): POLL, C1
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

### gamer-omp: median fit (s), and in parentheses time / pr

| case | main | pr | abs | pr+fuse | pr+fuse3 | abs+fuse3 | none+fuse3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XS | 0.598 (11.42) | 0.0524 | 0.0596 (1.14) | 0.0576 (1.10) | 0.0592 (1.13) | 0.0576 (1.10) | 0.0597 (1.14) |
| S-thin | 0.707 (9.42) | 0.075 | 0.0752 (1.00) | 0.0846 (1.13) | 0.0762 (1.02) | 0.0742 (0.99) | 0.075 (1.00) |
| S | 0.686 (2.30) | 0.298 | 0.295 (0.99) | 0.29 (0.97) | 0.294 (0.99) | 0.301 (1.01) | 0.29 (0.97) |
| M | 0.956 (2.47) | 0.387 | 0.387 (1.00) | 0.386 (1.00) | 0.391 (1.01) | 0.387 (1.00) | 0.396 (1.02) |
| M-very-thin | 1.01 (6.90) | 0.147 | 0.145 (0.99) | 0.146 (0.99) | 0.148 (1.00) | 0.145 (0.99) | 0.148 (1.00) |
| L-thin | 3.7 (4.46) | 0.828 | 0.822 (0.99) | 0.807 (0.97) | 0.728 (0.88) | 0.72 (0.87) | 0.732 (0.88) |
| L | 2.13 (2.08) | 1.02 | 0.986 (0.96) | 0.977 (0.95) | 0.894 (0.87) | 0.833 (0.81) | 0.881 (0.86) |
| L-stumps | 1.28 (0.99) | 1.29 | 1.19 (0.92) | 1.1 (0.85) | 1.24 (0.96) | 1.1 (0.85) | 1.33 (1.03) |
| L-leaf255 | 1.76 (1.91) | 0.92 | 0.909 (0.99) | 0.841 (0.91) | 0.751 (0.82) | 0.689 (0.75) | 0.731 (0.79) |
| ames_housing | 0.704 (3.95) | 0.178 | 0.178 (1.00) | 0.176 (0.99) | 0.177 (0.99) | 0.177 (0.99) | 0.177 (0.99) |
| kddcup09_churn | 2.32 (1.25) | 1.86 | 1.89 (1.01) | 1.9 (1.02) | 1.96 (1.05) | 1.88 (1.01) | 1.77 (0.95) |
| amazon_employee_access | 2 (6.21) | 0.322 | 0.325 (1.01) | 0.322 (1.00) | 0.326 (1.01) | 0.324 (1.01) | 0.322 (1.00) |
| covtype | 39.6 (4.14) | 9.57 | 9.43 (0.98) | 8.28 (0.87) | 7.07 (0.74) | 6.82 (0.71) | 7.99 (0.84) |
| susy | 19.6 (1.18) | 16.6 | 12.2 (0.73) | 15.3 (0.92) | 15.3 (0.93) | 12.3 (0.74) | 13.8 (0.83) |
| year_prediction_msd | 5.69 (0.32) | 17.9 | 5.54 (0.31) | 9.66 (0.54) | 7.3 (0.41) | 7.22 (0.40) | 7.06 (0.40) |
| geomean / pr | 2.73 | 1 | 0.90 | 0.94 | 0.90 | 0.86 | 0.89 |

### xeon-s0-passive: median fit (s), and in parentheses time / pr

| case | main | pr | abs | pr+fuse | pr+fuse3 | abs+fuse3 | none+fuse3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XS | 26.1 (289.06) | 0.0904 | 0.0888 (0.98) | 0.0886 (0.98) | 0.0887 (0.98) | 0.0896 (0.99) | 0.0881 (0.98) |
| S-thin | 30.6 (221.23) | 0.138 | 0.145 (1.05) | 0.149 (1.08) | 0.149 (1.08) | 0.141 (1.02) | 0.145 (1.05) |
| S | 27.1 (46.96) | 0.577 | 0.517 (0.90) | 0.555 (0.96) | 0.574 (1.00) | 0.559 (0.97) | 0.521 (0.90) |
| M | 32.5 (43.10) | 0.755 | 0.735 (0.97) | 0.73 (0.97) | 0.742 (0.98) | 0.819 (1.09) | 0.76 (1.01) |
| M-very-thin | 31.1 (113.36) | 0.274 | 0.251 (0.92) | 0.269 (0.98) | 0.255 (0.93) | 0.27 (0.99) | 0.271 (0.99) |
| L-thin | 33.6 (6.14) | 5.47 | 5.46 (1.00) | 4.87 (0.89) | 3.62 (0.66) | 3.68 (0.67) | 4.14 (0.76) |
| L | 34.1 (5.22) | 6.52 | 8.35 (1.28) | 5.49 (0.84) | 3.8 (0.58) | 3.69 (0.57) | 5.29 (0.81) |
| L-stumps | 2.02 (1.14) | 1.77 | 1.92 (1.09) | 1.64 (0.93) | 1.66 (0.94) | 1.77 (1.00) | 1.73 (0.98) |
| L-leaf255 | 56.2 (11.79) | 4.76 | 4.69 (0.98) | 5.17 (1.08) | 4 (0.84) | 4.16 (0.87) | 5.62 (1.18) |
| ames_housing | 27.5 (85.65) | 0.321 | 0.33 (1.03) | 0.311 (0.97) | 0.346 (1.08) | 0.336 (1.05) | 0.351 (1.09) |
| kddcup09_churn | 24.2 (3.31) | 7.33 | 7.38 (1.01) | 5.42 (0.74) | 4.31 (0.59) | 4.01 (0.55) | 5.14 (0.70) |
| amazon_employee_access | 49.4 (87.64) | 0.564 | 0.539 (0.96) | 0.537 (0.95) | 0.57 (1.01) | 0.583 (1.03) | 0.545 (0.97) |
| covtype |  | 91.6 | 91.7 (1.00) | 86.3 (0.94) | 58.7 (0.64) | 58.9 (0.64) | 78.5 (0.86) |
| susy | 105 (2.51) | 41.8 | 64.4 (1.54) | 25 (0.60) | 21.3 (0.51) | 24.2 (0.58) | 24.7 (0.59) |
| year_prediction_msd | 141 (2.57) | 54.9 | 98.4 (1.79) | 25.9 (0.47) | 17.6 (0.32) | 24.5 (0.45) | 43.3 (0.79) |
| geomean / pr | 19.33 | 1 | 1.08 | 0.87 | 0.77 | 0.80 | 0.90 |

failed: [('xeon-s0-passive', 'covtype', 'main', 'timeout')]

Notes:
- gamer libomp: on msd, abs alone (5.5 s, as fast as main) beats every
  fused variant (7.1-7.3 s); with libgomp it was the other way round. My
  guess: libomp's fork is cheap (hot team), so merging saves less, while the
  barriers and imbalance inside the merged regions cost more with
  `KMP_BLOCKTIME=0`. Not measured.
- xeon-s0-passive: main with 86 passive threads timed out on covtype (one
  fit 976 s). pr+fuse3 is the best variant (0.77 geomean, msd 54.9 -> 17.6 s,
  susy 0.51, covtype 0.64). abs hurts at 86 threads (msd 1.79, susy 1.54):
  a 20K-sample split_indices threshold threads far more nodes than the PR's
  86 x 5000 = 430K.

Geomean of time / pr over the 15 cases:

| profile | abs | pr+fuse | pr+fuse3 | abs+fuse3 | none+fuse3 |
| --- | --- | --- | --- | --- | --- |
| xeon-s0, active | | 0.90 | 0.76 | | |
| xeon-s0, passive | 1.08 | 0.87 | 0.77 | 0.80 | 0.90 |
| laptop, libgomp | 0.98 | 0.95 | 0.88 | 0.86 | 0.91 |
| gamer, libgomp | 0.93 | 0.94 | 0.89 | 0.89 | 0.90 |
| gamer, libomp | 0.90 | 0.94 | 0.90 | 0.86 | 0.89 |

year_prediction_msd, fit time (s):

| profile | main | pr | best variant |
| --- | --- | --- | --- |
| xeon-s0, active | 3.48 | 4.07 | 2.39 (pr+fuse3) |
| xeon-s0, passive | 141 | 54.9 | 17.6 (pr+fuse3) |
| laptop, libgomp | 13.5 | 12.6 | 4.3 (abs+fuse3), 4.6 (pr+fuse3) |
| gamer, libgomp | 12.8 | 21.0 | 7.8 (pr+fuse3) |
| gamer, libomp | 5.69 | 17.9 | 5.5 (abs), 7.3 (pr+fuse3) |

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

## Exp 12: phase times of msd, and where passive comes from (2026-11-04 06:50-07:05)

`bench.py --phases` runs `scripts/ab/phase_runner.py`, which sums the grower's
phase times over the fits. `results/gamer/exp12/`, 3 processes of 1 fit, median (s):

| libomp | fit | hist | find | apply |
| --- | --- | --- | --- | --- |
| main | 5.54 | 2.64 | 1.22 | 1.28 |
| abs | 5.89 | 2.99 | 1.48 | 0.93 |
| abs+fuse | 8.20 | 4.07 | 2.20 | 1.44 |
| abs+fuse3 | 7.74 | 5.05 (with find) | 0.06 | 2.13 |
| pr | 16.87 | 13.23 | 2.04 | 1.06 |

So merging makes both the parallel work and split_indices slower with libomp,
although the busy work is the same.

Correction of earlier notes: no OpenMP variable is set on gamer or the laptop.
With an empty environment, `OMP_DISPLAY_ENV` reports `GOMP_SPINCOUNT=1` on both
(libomp's `KMP_SETTINGS` reports `KMP_BLOCKTIME=0ms` on gamer), and
`GOMP_SPINCOUNT=300000` on the xeon, with the same pixi env. It looks like the
runtimes' default on hybrid CPUs. I didn't read their source.

## Exp 13: the barrier at the start of every prange (07:00-08:05)

`objdump` of the extensions: `GOMP_barrier` is the first call of every `_omp_fn`
generated for a `prange`. Cython emits, for the loop variable and every other
non-Python private of the loop:

    #pragma omp for nowait firstprivate(i) lastprivate(i) schedule(static)

GCC 15.3 adds a barrier at the start of the loop for the combination of the two
clauses, and not for either one alone (small C test). `Nodes.py` in Cython 3.3.0:
"The target iteration variable is firstprivate to shut up compiler warnings caused
by lastprivate", and in the GIL code "Any firstprivate creates a barrier at least
on GCC". Consequences:

- The counts of this log were wrong: the PR has 7 regions and 7 barriers per split
  (the threads must all be awake before any of them works), fuse3 has 2 regions
  and 3 barriers, not 1. Exp 8's `barrier.c` has none of these start barriers.
- A `prange` has no barrier at its end (`nowait`). The fused code of Exp 9 only
  worked thanks to the start barrier of the next `prange`, which is a side effect
  of a compiler. A clean version needs an explicit barrier.

`cakedev0/scikit-learn@hgb/fewer-regions-nb` (10b5afac14, build `nb`) replaces the
`prange` of the HGB fit by `parallel` blocks with manual static chunks, plus one
explicit barrier in the fused split_indices. `check_identical.py` passes within
that build. One xeon socket, 86 threads, 3 processes of 1 fit, ratio to `pr`
(`results/xeon-s0/exp13/`):

| case | passive pr (s) | passive nb:pr | passive pr+fuse3 | passive nb:pr+fuse3 | active pr (s) | active nb:pr | active pr+fuse3 | active nb:pr+fuse3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| year_prediction_msd | 48.07 | 0.76 | 0.38 | 0.27 | 4.28 | 0.88 | 0.64 | 0.61 |
| susy | 41.25 | 0.71 | 0.50 | 0.39 | 6.70 | 1.00 | 0.97 | 0.96 |
| L | 7.01 | 0.91 | 0.55 | 0.45 | 1.28 | 0.89 | 0.73 | 0.70 |
| L-leaf255 | 6.31 | 0.91 | 0.63 | 0.53 | 1.51 | 0.87 | 0.64 | 0.60 |
| kddcup09_churn | 7.36 | 0.83 | 0.54 | 0.50 | 1.24 | 0.85 | 0.79 | 0.74 |

## Exp 14: msd on E cores only (07:05-07:20)

`results/gamer/exp14/`. `phase_runner.py` also records idle state entries of the
whole machine (shallow = POLL + C1, deep = C2 + C3). 3 processes of 1 fit:

| | abs | abs+fuse3 | pr+fuse3 | nb:abs | nb:abs+fuse3 | nb:pr+fuse3 |
| --- | --- | --- | --- | --- | --- | --- |
| 24 threads, libomp | 5.17 | 8.29 | 7.33 | 8.76 | 5.45 | 4.83 |
| 24 threads, libgomp | 11.01 | 8.49 | 8.02 | 9.61 | 7.01 | 6.04 |
| 16 E cores, libomp | 4.23 | 3.19 | 3.40 | 3.45 | 2.93 | 3.14 |
| 16 E cores, libgomp | 3.95 | 3.15 | 3.22 | 3.56 | 3.01 | 3.08 |

On E cores merging wins with both runtimes, and 16 E cores beat the 24 cores by 2x
or more. Spinning (`KMP_BLOCKTIME=200ms`) is about 2-3x faster than passive, with
bimodal fits (`abs` 1.86-3.29 s), so no conclusion from its order.

## Exp 15: dynamic schedule of the merged feature loop (07:20-07:35)

`cakedev0/scikit-learn@hgb/fewer-regions-dyn` (27c1e978dc, build `dyn`):
`schedule='dynamic', chunksize=1` on the fused feature loop.
`results/gamer/exp15/`, 3 processes of 1 fit:

| | abs | abs+fuse3 | dyn:abs+fuse3 | pr+fuse3 | dyn:pr+fuse3 | nb:pr+fuse3 |
| --- | --- | --- | --- | --- | --- | --- |
| 24 threads, libomp | 6.05 | 8.60 | 5.82 | 7.15 | 6.01 | 5.45 |
| 24 threads, libgomp | 10.92 | 8.51 | 5.78 | 8.33 | 7.11 | 6.02 |
| 8 P cores, libomp | 4.44 | 5.59 | 4.75 | 6.36 | 5.11 | 7.13 |

So the loss of the merged variants also shows on identical P cores: the P/E
imbalance is not the whole cause. nb is the slowest variant there.

## Exp 16: split_indices threshold sweep with fuse3 (07:20-08:05)

`scripts/ab/run_exp16.sh`, `analyze_sweep.py`, `results/{gamer,xeon-s0}/exp16/`.
Variants `abs+fuse3@X` (threaded when n_samples >= X), `none+fuse3`, `pr+fuse3`.
Median of 2 fits, seconds.


### exp16-gamer-gomp.jsonl 

| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | 0.89 | 0.85 | 0.89 | 0.94 | 0.99 | 0.88 | 0.95 | 0.87 |
| L-leaf255 | 0.77 | 0.69 | 0.73 | 0.74 | 0.74 | 0.75 | 0.76 | 0.71 |
| kddcup09_churn | 1.95 | 1.93 | 1.96 | 2.00 | 1.99 | 2.06 | 2.05 | 1.90 |
| covtype | 6.75 | 6.31 | 6.87 | 7.09 | 7.59 | 8.52 | 8.20 | 6.74 |
| susy | 13.24 | 12.79 | 12.35 | 13.66 | 14.38 | 15.83 | 18.77 | 12.91 |
| year_prediction_msd | 8.97 | 7.69 | 9.13 | 8.28 | 8.39 | 8.20 | 8.19 | 8.20 |

### exp16-gamer-omp.jsonl 

| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | 0.94 | 0.91 | 0.95 | 0.90 | 0.97 | 1.05 | 0.99 | 0.99 |
| L-leaf255 | 0.86 | 0.73 | 0.75 | 0.80 | 0.81 | 0.84 | 0.76 | 0.78 |
| kddcup09_churn | 1.80 | 1.85 | 2.05 | 2.14 | 2.05 | 2.03 | 2.03 | 1.88 |
| covtype | 7.95 | 6.93 | 7.23 | 7.27 | 7.95 | 8.29 | 8.71 | 7.06 |
| susy | 13.10 | 13.74 | 13.12 | 13.61 | 15.22 | 16.30 | 20.47 | 13.80 |
| year_prediction_msd | 6.49 | 7.28 | 7.96 | 7.30 | 8.24 | 7.85 | 7.68 | 7.56 |

### exp16-xeon32.jsonl 

| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | 4.86 | 3.85 | 3.54 | 3.18 | 3.17 | 3.21 | 3.21 | 3.67 |
| L-leaf255 | 4.66 | 3.68 | 3.57 | 3.59 | 3.55 | 3.35 | 3.64 | 3.82 |
| kddcup09_churn | 4.94 | 4.35 | 3.90 | 3.91 | 3.77 | 3.71 | 3.89 | 4.08 |
| susy | 25.78 | 25.65 | 24.86 | 21.71 | 18.83 | 19.22 | 23.14 | 21.87 |
| year_prediction_msd | 35.87 | 25.38 | 21.08 | 18.72 | 17.40 | 17.11 | 17.32 | 17.85 |

### exp16-xeon86.jsonl 

| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | 5.14 | 3.97 | 3.74 | 3.36 | 3.34 | 3.41 | 3.39 | 3.74 |
| L-leaf255 | 5.58 | 3.96 | 3.84 | 3.93 | 3.74 | 3.96 | 3.99 | 4.02 |
| kddcup09_churn | 5.03 | 4.49 | 3.84 | 3.76 | 3.71 | 3.73 | 3.68 | 4.08 |
| susy | 24.16 | 24.36 | 23.92 | 20.54 | 18.07 | 18.95 | 22.68 | 21.23 |
| year_prediction_msd | 40.03 | 26.32 | 22.42 | 18.12 | 18.92 | 19.76 | 18.57 | 18.79 |

The PR's rule is 5000 samples per thread: ~115K on gamer (23-24 threads), 160K and
430K on the xeon (32 and 86 threads). Best thresholds: flat up to ~100K on gamer,
~300K on the xeon at both 32 and 86 threads. So the best value here depends on
the machine more than on the thread count. The PR's rule is within ~15% of the
best on most cases (xeon 32 threads: L 3.67 vs 3.17 s, susy 21.87 vs 18.83 s).
Always threaded is up to 2.2x slower on the xeon, always serial up to 1.5x slower
on susy on gamer. The laptop part of the sweep waited behind another agent's lock.

## Exp 17: only C1 allowed (07:50-08:05)

`scripts/ab/run_exp17.sh`: same variants with all idle states, then holding a 2 us
latency limit on `/dev/cpu_dma_latency` (through sudo, released when the holder
exits, no holder left after the run). Deep entries were 0.00 M with the limit.
`results/gamer/exp17/`, 3 processes of 1 fit, median (s):

| | abs | abs+fuse3 | pr+fuse3 | dyn:abs+fuse3 | nb:abs+fuse3 |
| --- | --- | --- | --- | --- | --- |
| 8 P cores, libomp, all states | 4.30 | 5.15 | 5.66 | 4.69 | 7.00 |
| 8 P cores, libomp, only C1 | 3.97 | 3.18 | 3.09 | 3.65 | 3.11 |
| 8 P cores, libgomp, all states | 4.32 | 4.06 | 5.01 | 4.16 | 5.18 |
| 8 P cores, libgomp, only C1 | 3.56 | 3.18 | 3.11 | 3.61 | 3.03 |
| 24 threads, libomp, all states | 6.00 | 8.17 | 7.88 | 6.37 | 5.55 |
| 24 threads, libomp, only C1 | 3.45 | 2.16 | 2.32 | 2.53 | 1.94 |
| 24 threads, libgomp, all states | 11.95 | 8.67 | 8.17 | 5.84 | 7.22 |
| 24 threads, libgomp, only C1 | 3.61 | 2.09 | 2.32 | 2.52 | 1.92 |

With only C1, merging wins everywhere, the two runtimes give the same times, and
the dynamic schedule is slower than the static one.

Laptop part of Exp 16 (16 threads, ran later that morning, `balanced` profile). Same
picture as gamer: flat up to ~100K, always serial is worse.

### exp16-laptop-gomp.jsonl 

| case | always threaded | 10K | 30K | 100K | 300K | 1M | always serial | PR rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | 1.19 | 1.12 | 1.13 | 1.24 | 1.20 | 1.24 | 1.43 | 1.30 |
| L-leaf255 | 1.12 | 0.96 | 0.95 | 0.92 | 0.98 | 0.94 | 1.03 | 0.95 |
| kddcup09_churn | 2.14 | 2.04 | 2.13 | 2.12 | 2.14 | 2.09 | 2.29 | 2.08 |
| covtype | 8.64 | 8.03 | 8.46 | 8.66 | 9.18 | 10.18 | 10.13 | 8.06 |
| susy | 12.83 | 12.56 | 12.93 | 13.94 | 15.71 | 18.14 | 20.98 | 12.81 |
| year_prediction_msd | 7.12 | 4.55 | 4.55 | 4.77 | 5.10 | 5.40 | 5.36 | 4.75 |

## Exp 18: noise level, and the clean branch (08:07-10:50)

`cakedev0/scikit-learn@hgb/merge-regions` (6ee4279445, build `clean`, variant
`clean:pr`): fuse3 only, no env var, one path at every thread count, explicit
barrier in split_indices, no gather when the children are bound to be leaves.
`scripts/ab/run_exp18.sh`, `analyze_noise.py`, `results/*/exp18/`. 5 processes of 1
fit per case and variant, all 15 cases, variants shuffled per case. Ratio to `pr`,
median (min-max of the 5 processes), `pr spread` = (max - min) / median of `pr`,
geomean interval = 5-95% bootstrap over the processes.

`identity.py`: tree bytes and predictions, 6 parameter sets, forced 1 and 8 threads:
`clean` == `pr` == `main` for all 12 on gamer, laptop and xeon.

`exp18-gamer-gomp-run2-mislabeled-omp.jsonl` was meant to be libomp, but a literal
`~` in `LD_PRELOAD` made ld.so ignore it, so it's a second libgomp run. The earlier
libomp profiles have an absolute path. `bench.py` now refuses a missing file. The
libomp profile was redone with `run_exp18_omp.sh`.

### exp18-gamer-gomp

| case | pr | pr+fuse3 | abs+fuse3 | clean:pr | pr spread |
| --- | --- | --- | --- | --- | --- |
| XS | 0.0558 | 0.98 (0.97-0.99) | 0.99 (0.98-1.01) | 0.92 (0.91-0.92) | 3% |
| S-thin | 0.121 | 1.00 (0.99-1.01) | 1.00 (0.98-1.01) | 0.96 (0.93-0.96) | 1% |
| S | 0.326 | 1.01 (1.00-1.02) | 1.01 (1.00-1.01) | 0.95 (0.95-0.96) | 1% |
| M | 0.416 | 1.00 (1.00-1.01) | 1.00 (1.00-1.01) | 0.97 (0.97-0.98) | 2% |
| M-very-thin | 0.188 | 1.00 (0.99-1.01) | 1.00 (0.98-1.00) | 0.98 (0.97-0.98) | 1% |
| L-thin | 0.848 | 0.91 (0.88-0.94) | 0.88 (0.87-0.95) | 0.87 (0.86-0.88) | 4% |
| L | 0.968 | 0.91 (0.87-0.93) | 0.93 (0.90-0.99) | 0.91 (0.85-0.95) | 9% |
| L-stumps | 1.13 | 0.99 (0.97-1.04) | 1.02 (1.00-1.09) | 0.99 (0.98-1.03) | 5% |
| L-leaf255 | 0.873 | 0.82 (0.78-0.83) | 0.82 (0.76-0.86) | 0.83 (0.81-0.85) | 3% |
| ames_housing | 0.221 | 0.99 (0.99-1.00) | 0.99 (0.98-1.00) | 0.96 (0.95-0.97) | 2% |
| kddcup09_churn | 1.94 | 0.97 (0.96-1.00) | 1.00 (0.94-1.03) | 0.98 (0.96-0.98) | 14% |
| amazon_employee_access | 0.365 | 1.00 (1.00-1.01) | 1.00 (0.99-1.01) | 0.99 (0.98-0.99) | 1% |
| covtype | 8.41 | 0.76 (0.75-0.80) | 0.76 (0.76-0.79) | 0.76 (0.75-0.80) | 5% |
| susy | 17.1 | 0.77 (0.74-0.81) | 0.70 (0.68-0.76) | 0.80 (0.80-0.84) | 16% |
| year_prediction_msd | 20.3 | 0.39 (0.38-0.39) | 0.40 (0.39-0.44) | 0.40 (0.39-0.41) | 8% |
| geomean / pr | 1 | 0.88 (0.87-0.89) | 0.88 (0.87-0.89) | 0.87 (0.86-0.88) | |

### exp18-laptop-gomp

| case | pr | pr+fuse3 | abs+fuse3 | clean:pr | pr spread |
| --- | --- | --- | --- | --- | --- |
| XS | 0.0675 | 1.00 (1.00-1.02) | 1.00 (0.99-1.01) | 0.94 (0.93-0.94) | 1% |
| S-thin | 0.146 | 0.99 (0.97-1.04) | 1.03 (1.01-1.04) | 0.96 (0.95-1.00) | 9% |
| S | 0.436 | 0.99 (0.98-1.00) | 0.99 (0.98-1.00) | 0.95 (0.95-0.96) | 1% |
| M | 0.513 | 0.99 (0.99-1.00) | 1.00 (0.99-1.00) | 0.96 (0.96-0.97) | 1% |
| M-very-thin | 0.225 | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 0.97 (0.97-0.98) | 1% |
| L-thin | 1.06 | 1.05 (0.88-1.09) | 1.03 (0.92-1.07) | 1.00 (0.87-1.08) | 22% |
| L | 1.35 | 0.89 (0.84-0.91) | 0.87 (0.85-0.91) | 0.86 (0.84-0.90) | 21% |
| L-stumps | 1.86 | 1.04 (0.98-1.07) | 1.08 (0.99-1.16) | 1.00 (0.94-1.02) | 13% |
| L-leaf255 | 1.27 | 0.78 (0.74-0.80) | 0.73 (0.68-0.79) | 0.74 (0.72-0.80) | 9% |
| ames_housing | 0.288 | 0.99 (0.99-1.01) | 1.00 (0.99-1.00) | 0.97 (0.96-0.97) | 1% |
| kddcup09_churn | 2.27 | 0.92 (0.90-0.93) | 0.90 (0.89-0.93) | 0.91 (0.90-0.92) | 3% |
| amazon_employee_access | 0.446 | 1.00 (0.99-1.01) | 1.00 (1.00-1.01) | 0.99 (0.98-0.99) | 1% |
| covtype | 11 | 0.74 (0.71-0.77) | 0.71 (0.70-0.76) | 0.71 (0.69-0.72) | 9% |
| susy | 13.5 | 0.97 (0.90-1.03) | 0.95 (0.86-0.96) | 0.94 (0.87-0.97) | 2% |
| year_prediction_msd | 12.1 | 0.40 (0.38-0.41) | 0.36 (0.35-0.38) | 0.38 (0.37-0.41) | 3% |
| geomean / pr | 1 | 0.90 (0.88-0.91) | 0.88 (0.87-0.90) | 0.87 (0.85-0.88) | |

### exp18-xeon-active

| case | pr | pr+fuse3 | clean:pr | pr spread |
| --- | --- | --- | --- | --- |
| XS | 0.0901 | 1.00 (0.99-1.02) | 0.92 (0.92-0.93) | 4% |
| S-thin | 0.229 | 0.80 (0.76-0.85) | 0.75 (0.75-0.78) | 5% |
| S | 0.7 | 0.67 (0.65-0.68) | 0.65 (0.65-0.66) | 4% |
| M | 0.757 | 0.61 (0.61-0.66) | 0.61 (0.59-0.62) | 6% |
| M-very-thin | 0.319 | 0.91 (0.89-0.93) | 0.84 (0.80-0.90) | 7% |
| L-thin | 1.21 | 0.88 (0.83-1.00) | 0.97 (0.87-1.06) | 20% |
| L | 1.26 | 0.77 (0.71-0.80) | 0.75 (0.72-0.75) | 3% |
| L-stumps | 0.971 | 0.97 (0.96-1.00) | 0.98 (0.94-1.00) | 3% |
| L-leaf255 | 1.53 | 0.61 (0.60-0.63) | 0.59 (0.57-0.61) | 2% |
| ames_housing | 0.526 | 0.65 (0.64-0.67) | 0.63 (0.62-0.65) | 5% |
| kddcup09_churn | 1.17 | 0.81 (0.79-0.87) | 0.79 (0.79-0.80) | 11% |
| amazon_employee_access | 0.485 | 0.85 (0.80-0.86) | 0.81 (0.78-0.83) | 7% |
| covtype | 13.4 | 0.76 (0.67-0.81) | 0.70 (0.65-0.76) | 8% |
| susy | 6.66 | 0.98 (0.96-0.99) | 0.96 (0.94-0.96) | 2% |
| year_prediction_msd | 4.33 | 0.66 (0.56-0.71) | 0.64 (0.54-0.65) | 5% |
| geomean / pr | 1 | 0.78 (0.77-0.79) | 0.76 (0.75-0.77) | |

### exp18-xeon-passive

| case | pr | pr+fuse3 | abs+fuse3 | clean:pr | pr spread |
| --- | --- | --- | --- | --- | --- |
| XS | 0.089 | 1.01 (1.00-1.05) | 1.03 (1.01-1.04) | 0.94 (0.93-0.98) | 3% |
| S-thin | 0.205 | 0.99 (0.94-1.01) | 0.99 (0.97-1.03) | 0.94 (0.90-0.95) | 5% |
| S | 0.593 | 0.98 (0.98-1.01) | 1.00 (0.97-1.03) | 0.97 (0.96-0.98) | 4% |
| M | 0.813 | 1.01 (0.99-1.02) | 1.01 (1.00-1.02) | 0.98 (0.95-1.00) | 5% |
| M-very-thin | 0.32 | 1.00 (0.99-1.02) | 1.01 (1.00-1.04) | 0.99 (0.97-1.03) | 6% |
| L-thin | 6.06 | 0.59 (0.58-0.60) | 0.58 (0.56-0.60) | 0.63 (0.60-0.64) | 19% |
| L | 8.1 | 0.46 (0.45-0.48) | 0.47 (0.46-0.50) | 0.47 (0.46-0.48) | 15% |
| L-stumps | 1.95 | 0.91 (0.89-0.95) | 0.94 (0.93-1.01) | 0.88 (0.87-0.97) | 10% |
| L-leaf255 | 5.68 | 0.68 (0.64-0.74) | 0.69 (0.64-0.72) | 0.68 (0.67-0.71) | 43% |
| ames_housing | 0.386 | 0.99 (0.97-1.04) | 0.98 (0.97-0.99) | 0.97 (0.94-0.98) | 5% |
| kddcup09_churn | 7.19 | 0.56 (0.54-0.58) | 0.56 (0.56-0.58) | 0.56 (0.55-0.57) | 8% |
| amazon_employee_access | 0.606 | 1.00 (0.99-1.01) | 1.00 (0.99-1.01) | 0.97 (0.97-0.98) | 1% |
| covtype | 97.5 | 0.58 (0.55-0.62) | 0.59 (0.55-0.61) | 0.61 (0.60-0.63) | 9% |
| susy | 42.2 | 0.50 (0.48-0.51) | 0.57 (0.56-0.60) | 0.54 (0.53-0.57) | 6% |
| year_prediction_msd | 48.6 | 0.40 (0.38-0.43) | 0.57 (0.47-0.58) | 0.38 (0.37-0.44) | 11% |
| geomean / pr | 1 | 0.74 (0.73-0.75) | 0.77 (0.75-0.78) | 0.73 (0.72-0.75) | |

Noise: the per-case spread of `pr` is 1-5% on most small cases, up to 16-22% on
some multi-threaded cases (43% once, L-leaf255 on xeon passive). Geomeans are good to
about +-0.01-0.02. So 0.9x differences on one case and one process, like in Exp 10,
aren't reliable, but the geomean differences to `pr` are. `pr+fuse3` and `abs+fuse3`
can't be told apart on most profiles, except msd and susy on xeon passive, where `pr`'s
rule is better (msd 0.40 vs 0.57). `clean` is as good as the experimental fuse3 on
multi-threaded cases, and 1-8% faster than `pr` on the cases that get 1 thread.

Libomp rerun (`run_exp18_omp.sh`, `threadpool_info` shows libomp loaded):

### exp18-gamer-omp2

| case | pr | pr+fuse3 | abs+fuse3 | clean:pr | pr spread |
| --- | --- | --- | --- | --- | --- |
| XS | 0.0583 | 1.01 (1.01-1.04) | 1.03 (1.00-1.04) | 0.96 (0.95-0.99) | 2% |
| S-thin | 0.127 | 1.01 (1.00-1.01) | 1.00 (0.99-1.02) | 0.97 (0.96-0.99) | 3% |
| S | 0.331 | 1.00 (0.99-1.00) | 0.99 (0.99-1.01) | 0.97 (0.96-0.97) | 1% |
| M | 0.437 | 0.98 (0.98-0.99) | 1.00 (0.99-1.00) | 0.97 (0.93-0.98) | 2% |
| M-very-thin | 0.199 | 1.01 (1.00-1.03) | 1.02 (1.01-1.04) | 1.00 (0.99-1.02) | 2% |
| L-thin | 0.872 | 0.90 (0.85-0.94) | 0.90 (0.89-0.91) | 0.88 (0.87-0.93) | 11% |
| L | 1.09 | 0.85 (0.83-0.91) | 0.84 (0.81-0.88) | 0.85 (0.80-0.89) | 4% |
| L-stumps | 1.14 | 1.02 (1.00-1.17) | 1.06 (1.01-1.11) | 1.03 (1.00-1.09) | 21% |
| L-leaf255 | 1.02 | 0.75 (0.71-0.87) | 0.73 (0.69-0.78) | 0.73 (0.70-0.75) | 9% |
| ames_housing | 0.227 | 1.00 (1.00-1.01) | 1.00 (1.00-1.00) | 0.98 (0.97-0.98) | 2% |
| kddcup09_churn | 1.97 | 1.01 (1.00-1.16) | 1.02 (0.98-1.03) | 0.98 (0.94-1.00) | 6% |
| amazon_employee_access | 0.383 | 0.97 (0.97-1.00) | 1.00 (0.98-1.00) | 0.98 (0.95-0.99) | 4% |
| covtype | 9.56 | 0.73 (0.72-0.74) | 0.74 (0.73-0.75) | 0.72 (0.70-0.73) | 7% |
| susy | 17.4 | 0.78 (0.76-0.83) | 0.75 (0.72-0.81) | 0.82 (0.74-0.87) | 17% |
| year_prediction_msd | 16.8 | 0.44 (0.41-0.49) | 0.48 (0.43-0.48) | 0.42 (0.40-0.58) | 16% |
| geomean / pr | 1 | 0.88 (0.87-0.90) | 0.89 (0.87-0.90) | 0.87 (0.85-0.89) | |

## Exp 19: portability of hgb/merge-regions (2026-11-04 ~11:00-12:15, local time)

`scripts/portability/`, `results/portability/`. Conda-forge clang 23.1.4 and libomp,
Python 3.13, branch at 6ee427944, HGB tests on 4 threads (macOS: same):

| build | result |
| --- | --- |
| Linux, GCC 15.3 (Exp 18) | 395 passed, 105 skipped |
| Linux, clang + libomp | 395 passed, 105 skipped |
| macOS 27 arm64, conda clang + libomp | 395 passed, 105 skipped |
| macOS 27 arm64, Apple clang 16, no OpenMP (0 OpenMP symbols, 1 effective thread) | 395 passed, 105 skipped |
| MSVC | not tested. `__pragma(omp barrier)` in a macro gives `__kmpc_barrier` with clang `-fms-extensions` (`ms.c`) |

On the mac, the Command Line Tools' macOS 27 SDK can't be used with this Xcode 16
linker, so the builds use Xcode's macOS 15 SDK. That's the machine's setup, not the
branch.

With clang, a `prange` has 2 barriers instead of GCC's 1: `find_node_split` 2,
`compute_histograms_and_find_splits` 2, `split_indices` 5 (2 + explicit + 2).

TSan with Archer (`libarcher` from conda's llvm-openmp, so that OpenMP
synchronization is known to TSan), C extensions built with `-fsanitize=thread`, 5 fits
on 4 threads with active wait forced so that the threaded `split_indices` runs
from 2000 samples (`tsan_fit.py`: numerical, categorical + `max_features`, monotonic,
interaction constraints, classifier with `max_depth`):

| build | TSan reports |
| --- | --- |
| hgb/merge-regions | 0 |
| hgb/fewer-regions-nb (manual chunks, no start barrier, explicit barrier) | 0 |
| same, explicit barrier deleted, counts initialized to 0 | 14 (`split_indices`, then a crash) |
| same, explicit barrier deleted, counts not initialized | crash |

The last two show that the tool sees this kind of race, and that the explicit barrier
is what makes manual chunks correct. On `hgb/merge-regions`, clang's own `prange`
barriers would also hide a missing explicit barrier, so this run doesn't prove that the
explicit one is needed there. It's what the OpenMP spec requires with `nowait`. I only
ran TSan on these 5 fits, not on the test suite.

## Note: Cython version of all the builds (2026-11-04 12:10)

`pyproject.toml` pins `cython>=3.1.2,<3.2.6`, but `--no-build-isolation` ignored it. All
the builds of this log used the Cython of the env: 3.3.0 on gamer and the xeon,
3.3.0a2.dev0 on the laptop, 3.3.1 for Exp 19's clang and macOS builds. So `main`, the
PR and every variant here were built with a version that a release of scikit-learn
wouldn't use. From the Cython session (`cython-prange-barrier/FINDINGS.md`): `nowait` on
every `prange` comes from GH-6562 in 3.3. Before that, a `prange` followed by code in a
`parallel` block also has a barrier at its end. The start barrier is in all these
versions, and it covers more variables since 3.2 (GH-7178). Barrier counts and timings
of Exp 8-19 may differ on the pinned versions. Not measured yet.


## Exp 20: A/B with Cython 3.2.5, the version the pin allows (2026-11-04)

`scripts/ab/run_exp20.sh`, `build_cython325.sh`. 4 builds interleaved on the 15 cases, 5
processes of 1 fit: `c325pr` (#34935 at 313be4c8) and `c325clean` (`hgb/merge-regions` at
cdc4301324) on 3.2.5, against the existing `pr` (`hgb/fewer-regions` with its switches set
to match #34935) and `clean` (6ee427944, differs by one comment) on 3.3.0 (3.3.0a2.dev0 on
the laptop). `main` and `c325main` on msd only, 3 processes of 1 fit. Results in
`results/*/exp20/`, tables with `analyze_noise.py --ref c325pr:pr` and `analyze_exp20.py`.

The C headers say 3.2.5. Trees and predictions: `identical 12 of 12` between `c325main`,
`c325pr` and `c325clean` on the 3 machines. HGB tests on `c325clean`, gamer: 395 passed,
105 skipped. No failed run in 1530.

Geomean over the 15 cases of the PR vs #34935, same Cython:

| profile | 3.2.5 (5-95% bootstrap) | 3.3 | PR 3.3/3.2.5 | #34935 3.3/3.2.5 |
| --- | --- | --- | --- | --- |
| xeon socket, active | 0.74 (0.73-0.75) | 0.75 | 0.96 | 0.94 |
| xeon socket, passive | 0.77 (0.76-0.80) | 0.74 | 0.91 | 0.95 |
| laptop | 0.86 (0.84-0.88) | 0.85 | 0.97 | 0.98 |
| gamer libgomp | 0.89 (0.87-0.90) | 0.86 | 0.96 | 0.98 |
| gamer libomp | 0.88 (0.87-0.91) | 0.87 | 0.98 | 0.99 |

msd, median fit in seconds:

| profile | main 3.2.5 | main 3.3 | #34935 3.2.5 | #34935 3.3 | PR 3.2.5 | PR 3.3 |
| --- | --- | --- | --- | --- | --- | --- |
| xeon socket, active | 3.89 | 3.42 | 4.82 | 4.33 | 2.81 | 2.70 |
| xeon socket, passive | 178 | 145 | 63.8 | 52.9 | 23.9 | 19.6 |
| laptop | 15.0 | 12.9 | 13.5 | 12.2 | 4.82 | 4.59 |
| gamer libgomp | 12.8 | 11.9 | 24.3 | 20.6 | 9.94 | 8.12 |
| gamer libomp | 6.48 | 5.44 | 17.7 | 17.1 | 8.81 | 7.92 |

- The conclusion holds on 3.2.5: the gain is within 0.03 of 3.3's. It fixes msd against
  main everywhere but gamer libomp (8.81 vs 6.48 s). The worst median over all cases and
  profiles is `kddcup09_churn` on gamer libgomp at 1.05x, with processes from 0.93x to
  1.21x, then `L-stumps` at 1.03x on gamer libomp and xeon active.
- The 3.3 column reproduces Exp 18 (0.76, 0.73, 0.87, 0.87, 0.87) within 0.02 in a
  separate run.
- Same source, 3.3 is faster than 3.2.5: 2-9% on the PR, 1-6% on #34935, 7-19% on main's
  msd. Same compiler (GCC 15.3) and options (release, -O3) on gamer's `main` and
  `c325main`. I don't know why: `splitting.so` and `histogram.so` of `main` have the same
  `GOMP_barrier` sites on both versions (3 and 4), so the `nowait` isn't an obvious cause.
- Executed barriers, counted on the main thread with `count_barriers.c`, a `LD_PRELOAD`
  wrapper around `GOMP_barrier` and `GOMP_parallel`, in a 4-thread active-wait fit on
  gamer (`run_count_barriers.sh`). `split_indices` runs 4 per region on 3.2.5 (start of the
  first `prange`, end of it, the explicit one, start of the second) and 3 on 3.3. The
  histogram region runs 1 on both. So a split of `hgb/merge-regions` has 2 regions and 5
  barriers on 3.2.5, 4 on 3.3. Exp 13's 3 was right for fuse3, which has no explicit
  barrier, but I copied it to the PR. It was wrong there.
- A first count showed no `split_indices` region at all. The fit had 200K x 10 = 2M cells,
  which #34935 sends to 1 thread with a passive runtime. Active wait fixed the test.
- No clear cost for the extra barrier on 3.2.5: on msd, the 3.3 gain of the PR (4-18%)
  isn't consistently bigger than main's (7-19%), which has none to remove.

## Exp 21: mac-m2 and aws-l4-1, main vs #34935 vs merge-regions (2026-11-04)

`scripts/ab/port/` (private pixi env, `build_port.sh`, `run_exp21.sh`), `analyze_exp21.py`.
All on Cython 3.2.5, 3 builds interleaved on the 15 cases, 5 processes of 1 fit. Results in
`results/{mac-m2,aws-l4-1}/exp21/`. No failed run in 1125.

mac-m2: conda-forge clang and llvm-openmp, 8 threads, under `caffeinate`, since one import
took 584 s without it (I think the Mac slept). The largest gap between wall and fit time in
these runs is 14 s (`L-stumps`, data generation), so no such stall here. aws-l4-1: EPYC 7R13
VM, 2 cores with hyperthreads, GCC 16.2, libgomp, 2 threads is the default there.

The Mac is passive by default: libomp reports `KMP_BLOCKTIME=0` and
`OMP_WAIT_POLICY=PASSIVE` with no variable set, and `_openmp_uses_active_wait` says False.
aws-l4-1 detects active wait by default.

Identity: 12 of 12 on both. HGB tests on `c325clean`: 395 passed, 105 skipped on both.

| profile | #34935/main | #42/main | #42/#34935 (5-95%) | msd main | #34935 | #42 |
| --- | --- | --- | --- | --- | --- | --- |
| mac-m2, default (passive) | 0.34 | 0.30 | 0.87 (0.86-0.88) | 13.0 s | 11.6 s | 7.73 s |
| mac-m2, active forced | 0.95 | 0.88 | 0.92 (0.87-0.96) | 9.76 s | 9.69 s | 9.59 s |
| aws-l4-1, 2 threads (default) | 0.99 | 0.94 | 0.95 (0.93-0.97) | 15.3 s | 15.3 s | 15.3 s |
| aws-l4-1, 4 threads | 0.98 | 0.91 | 0.93 (0.83-0.96) | 11.7 s | 11.7 s | 11.4 s |
| aws-l4-1, 2 threads, passive | 0.62 | 0.55 | 0.89 (0.88-0.91) | 18.4 s | 17.8 s | 16.1 s |

- #42 is faster than #34935 on all 5, and #34935 has no msd regression against main here.
- The biggest gain over main is on the Mac by default: 3.3x with #42, up to 18x on `XS`, where
  #34935 gets the same gain. main has the passive cost, and its active-wait numbers are close.
- Worst median of #42 vs #34935: `L-stumps` 1.07x and `L-leaf255` 1.06x on the Mac with
  active wait forced, with overlapping process ranges. `L-stumps` is 0.99-1.03x on the other
  9 profiles of Exp 20-21, so I don't read a pattern.
- Worst median of #42 vs main: `L-thin` 1.16x (1.01-1.33x) on the Mac with active wait
  forced. #34935 is 1.21x on this case, and main's spread is 34%.
- The Mac with active wait forced and aws-l4-1 at 4 threads are noisy, with single fits up to
  5x (`kddcup09_churn`, aws-l4-1, #34935). So their geomean intervals are wide.

## Note: answer from the Cython session on 3.3's gain (2026-11-04)

It reproduced the gain on main with 3.2.4 vs 3.3.1, about 7-17% on msd, spread over
histograms, split finding and applying splits. It doesn't know the cause either. In the
parallel functions of `histogram.c` and `splitting.c`, it found `nowait`, a different
`num_threads` expression and a different setup of Python calls. It didn't diff utility code
or disassembly. Details in `cython-prange-barrier/LOG.md` Exp 8 on the branch
`cython-prange-barrier`.

## Decisions (2026-11-04)

- Manual chunks (`nb`): dropped, on Arthur's call.
- `hgb/merge-regions` goes up as a follow-up to #34935, after Arthur's review.
- Next: retune the heuristics in #34935, before the follow-up.

## Exp 22: fit time against the thread count (2026-11-05)

`run_exp22.sh`, `analyze_exp22.py`, `knobs.patch` and `build_knob.sh` (#34935 at 313be4c8 plus env
knobs: `T` forces the thread count, `SM` the samples per thread of `split_indices`, `FS` the factor of
`find_node_split`). All of Exp 22-27 use Cython 3.2.5, leave #42 out and compare against #34935.
19 cases: the 15 of `hgb_scalability` plus `N-thin` 200K x 10, `N` 40K x 50, `N-wide` 10K x 300 and
`O` 300K x 100. 3 processes of 1 fit. Results in `results/*/exp22/`. The NUMA node and whole-machine
Xeon results are in `xeon-s0/` too.

| profile | main / #34935 | best T per case / #34935 |
| --- | --- | --- |
| xeon, 1 NUMA node, 22 threads, active | 1.30 | 0.92 |
| xeon, 1 socket, 86 threads, active | 1.42 | 0.79 |
| xeon, whole machine, 172 threads, active | 2.23 | 0.81 |
| xeon, 1 NUMA node, 22 threads, passive | 12.48 | 0.64 |
| gamer, 24 threads, libgomp | 5.24 | 0.80 |
| gamer, 24 threads, libomp | 2.60 | 0.86 |
| gamer, active forced | 1.15 | 0.97 |
| laptop, 16 threads, libgomp | 3.92 | 0.89 |
| laptop, active forced | 1.16 | 0.93 |
| mac-m2, 8 threads | 2.65 | 0.97 |
| mac-m2, active forced | 1.06 | 0.94 |

- With active wait, #34935 gives too many threads to small and mid problems. On one socket, `S`, `M`,
  `N` and `ames_housing` are at 0.57-0.61x with T=8 instead of T=40-50. The big ones need more:
  `L-stumps` 0.96x at T=64, `kddcup09_churn` and `susy` 0.98x at T=32. `msd` is best at T=16 (0.68x).
- On the whole machine, T=8 or T=16 is best on most cases, and T=172 costs up to 19x (`XS`). T=64 is
  the best only on `L-stumps`.
- In passive mode, #34935's top band is the problem: `msd` gets T=23 on gamer and is at 0.17x with
  T=16 (libgomp), 0.27x with libomp. `O` is at 0.40x with T=8. T=24 can cost 25x on small cases.
- The Xeon in passive mode wants even fewer threads: `msd` 0.17x at T=2, `covtype` 0.27x at T=1. I
  didn't tune for this one: passive isn't its default.
- Always threading `find_node_split` (`FS=0`) only helps at #34935's high T: 0.47x on `msd` on gamer.

## Exp 23: passive guards, with the top band capped at 8 threads (2026-11-05)

`run_exp23.sh`, `analyze_exp23.py`. Base: #34935 with `PCAP=8`. 10 cases that get several threads, 3
processes. `results/{gamer,laptop,mac-m2}/exp23/`. Geomean against the base:

| guard | gamer libgomp | gamer libomp | laptop | mac-m2 |
| --- | --- | --- | --- | --- |
| `SM=500` | 1.00 | 1.03 | 1.09 | 1.03 |
| `SM=2000` | 0.99 | 1.01 | 1.03 | 1.01 |
| `SM=20000` | 1.06 | 1.02 | 1.01 | 1.00 |
| `SM=100000` | 1.06 | 1.09 | 1.02 | 1.02 |
| `FS=0` (always threaded) | 1.02 | 1.05 | 1.02 | 1.04 |
| `FS=1` | 1.02 | 1.03 | 1.01 | 1.03 |
| `FS=2` | 1.02 | 1.03 | 1.00 | 1.03 |
| `FS=100000` (never) | 1.11 | 1.08 | 1.11 | 1.02 |
| #34935, no cap | 1.30 | 1.22 | 1.03 | 0.99 |

- Once the top band is capped, no guard setting beats 5000 samples per thread and factor 5 by more
  than 1%. So the guards stay as they are in passive mode.
- The cap removes the msd regression on gamer: #34935 is 5.08x the capped base with libgomp, 3.14x with
  libomp. `L-stumps` pays for it, at 0.72-0.76x for #34935, so the top band needs a size limit (Exp 26).

## Exp 24: active guards at a provisional stair (2026-11-05)

`run_exp24.sh`. Base: stair 200K:4/5M:8/20M:16. 18 cases, 3 processes, laptop and mac-m2 with active
wait forced. It did not run on the Xeon or gamer. `results/{laptop,mac-m2}/exp24/`.

| guard | laptop | mac-m2 |
| --- | --- | --- |
| `SM=100` | 1.02 | 1.03 |
| `SM=2000` | 1.02 | 1.02 |
| `SM=5000` | 1.03 | 1.01 |
| `SM=20000` | 1.12 | 1.05 |
| `FS=1` | 0.99 | 1.01 |
| `FS=2` | 1.04 | 1.01 |
| `FS=5` | 1.13 | 1.06 |
| #34935 | 1.14 | 1.09 |

No setting clearly beats 500 samples per thread and always threaded, so these stay too. The laptop
is noisy here (intervals up to 1.93).

## Exp 25: candidate stairs on 15 held-out shapes (2026-11-05)

`run_exp25.sh`, `analyze_exp25.py`. 15 shapes from 100K to 200M values, 5 processes, interleaved.
`results/*/exp25/`. Candidates: A32 is 200K:4/5M:8/32, A16 is 200K:4/5M:8/20M:16/32, A64 is
200K:4/5M:8/20M:16/100M:32/64 and P8 is 2M:1/20M:4/8. Geomean against #34935:

| profile | main | A32 | A16 | A64 | P8 |
| --- | --- | --- | --- | --- | --- |
| xeon, 1 NUMA node | 1.14 | 0.94 | 0.92 | 0.92 | |
| xeon, 1 socket | 1.16 | 0.86 | 0.85 | 0.82 | |
| xeon, 1 socket, `OMP_PROC_BIND=close` | | 0.97 | 0.98 | 0.95 | |
| gamer, active forced | 1.09 | 0.98 | 1.00 | | |
| laptop, active forced | 1.10 | | 0.97 | | |
| mac-m2, active forced | 0.99 | | 0.94 | | |
| gamer, libgomp | 4.33 | | | | 0.85 |
| gamer, libomp | 2.07 | | | | 0.85 |
| laptop, libgomp | 2.89 | | | | 0.95 |
| mac-m2 | 1.98 | | | | 1.00 |

- Capping too low hurts at the top end. `h-200M` is 1.34x with a cap of 32 on one socket (T=25) and
  0.99x with 64 (T=50). On the laptop, `P8` has `h-200M` at 1.30x and `h-80M` at 1.23x.
- So the top bands need a second look (Exp 26). The Xeon `s0-close` run was stopped at 181 of 375 rows
  to free the machine for Exp 26a and 27, so that row only covers part of the processes.

## Exp 26: where the passive cap goes from 8 to 16 (2026-11-05)

`run_exp26.sh`, `analyze_exp26.py`. 11 shapes from 30M to 120M values, 5 processes, cap 16 against
cap 8. `results/{gamer,laptop}/exp26/`. T is often 15 or 13 and not 16, from the features chunking.

| shape | laptop | gamer libgomp | gamer libomp |
| --- | --- | --- | --- |
| b-30M | 1.01 | 0.96 | 1.03 |
| b-36M | 1.69 | 0.88 | 0.89 |
| b-45M | 0.93 | 0.83 | 1.02 |
| b-50M | 1.49 | 0.78 | 0.77 |
| b-60M | 0.93 | 1.29 | 1.05 |
| b-60M-t | 1.09 | 0.94 | 0.84 |
| b-60M-w | 0.80 | 0.85 | 0.88 |
| b-60M-l127 | 1.71 | 0.88 | 0.88 |
| b-75M | 0.77 | 0.88 | 0.94 |
| b-100M | 0.83 | 1.18 | 1.28 |
| b-120M | 0.78 | 1.02 | 1.07 |

This doesn't give a sharp limit, and the two machines disagree. Gamer mostly prefers 16 from 36M to 75M
(`b-60M` is the exception), and 8 at 100M. The laptop can lose 1.5-1.7x with 16 up to 60M, and prefers
16 from 75M. I chose cap 8 up to 75M and 16 above, because the losses with 16 are the largest (1.7x,
against 1.3x with 8) and Exp 22 agrees at or below 30M (16 vs 8: `covtype` 1.16-2.11x, `O` 1.43-1.74x).
`L-stumps` (100M) wants 16: 0.78-0.83x in Exp 22. Anywhere from about 50M to 100M does about as well,
so 75M isn't a measured optimum.

## Exp 26a and 26b: active caps above 16 (2026-11-05)

`run_exp26a.sh`, `run_exp26b.sh`, `analyze_exp26a.py`, `analyze_exp26b.py`. Cap 32 and 64, then 20 and
24, against cap 16, 5 processes, on socket 1 (86 threads) and one NUMA node (22 threads).
`results/xeon-s0/exp26a/`, `results/xeon-s0/exp26b/`. One fit of 26a on the socket failed because I
killed it with a `pkill` meant for the Exp 25 queue. On the socket:

| case | values | cap 32 / 16 | cap 64 / 16 |
| --- | --- | --- | --- |
| `year_prediction_msd` | | 1.31 | 1.44 |
| h-25M | 25M | 1.23 | 1.23 |
| b-36M | 36M | 1.14 | 1.20 |
| h-40M-l127 | 40M | 1.33 | 1.38 |
| b-50M | 50M | 1.02 | 1.05 |
| b-60M | 60M | 0.75 | 0.93 |
| b-75M | 75M | 0.75 | 1.00 |
| b-100M | 100M | 0.72 | 0.59 |
| b-120M | 120M | 0.72 | 0.60 |
| h-200M | 200M | 0.69 | 0.52 |
| `L-stumps` | 100M | 0.78 | 0.62 |

- So the active stair ends with 50M:16/75M:32/64. On a NUMA node, the cap of 22 threads binds first:
  T=20-22 is 0.80-0.90x of cap 16 on most big cases, and `b-36M` is the worst at 1.15x.
- Between 20M and 50M, 26b tries a cap of 20 or 24. Geomean on the socket 1.02 and 1.04 (worst `b-36M`
  1.27x, `h-40M-l127` 1.36x), on a NUMA node 0.95 and 0.97 (cap 16's worst is `t-30M` at 0.87x, where
  the features chunking gives T=11 against 20).
- I kept 16: the effects are small and opposite, and the stair keeps doubling.

## The retuned rule (2026-11-05)

Branch `hgb/retune-threads`, local only. Commit 9d7e7756f4, on top of the merge of main (67a9bc9cdd).
`retune.patch` is the same rule on 313be4c8, which is what Exp 27 runs. It lowers `max_n_threads` before
the rest of the heuristic, by `n_samples * n_features` and wait mode. The guards don't change:

| values up to | active | passive |
| --- | --- | --- |
| 200K | 4 | 1 |
| 2M | 8 | 1 |
| 5M | 8 | 4 |
| 20M | 16 | 4 |
| 50M | 16 | 8 |
| 75M | 32 | 8 |
| more | 64 | 16 |

Up to 20K values it stays at 1 thread, in both modes. The features chunking can lower T further: 20
features with a cap of 16 gives T=10 or 11.

Tests: `test_get_heuristic_optimal_n_threads_monotonic_in_n_samples` and
`test_get_heuristic_optimal_n_threads_fewer_with_passive_wait` are new, and
`test_feature_thread_count_preserves_static_chunk_size` runs in both modes.
`check_retune_tests.py` and `check_retune_tests2.py` break the rule to see them fail. My first
monotonic test used powers of ten and missed a non-monotonic stair, so it now uses `2**k`. With the fix,
clean: 39 pass, tables swapped: 14 fail, non-monotonic stair: 2 fail.

On 9d7e7756f4: HGB and OpenMP helper tests 406 passed, 105 skipped, HGB tests with active wait forced 387
passed, 105 skipped (on this thinkpad). Trees and predictions of `retune.patch` are identical to #34935's on 12 of
12 parameter sets on the 4 machines.

## Exp 27: the retuned rule against #34935 and main (2026-11-05)

`run_exp27.sh`, `analyze_exp27.py`, `build_retune.sh`, `retune.patch`. Real code, with no knobs: #34935 at
313be4c8, the same plus the rule, and main at its merge base, all on Cython 3.2.5, interleaved. 15 cases
of `hgb_scalability`, which also served for tuning, and 15 `t-*` shapes from 150K to 300M values, which
did not. `t-22M`, `t-30M` and `t-48M` also ran in Exp 26b, but the rule didn't change after that. 5
processes of 1 fit (3 for the Xeon passive and whole-machine rows). `results/*/exp27/`.

Geomean against #34935, with the 5-95% bootstrap over the processes, and msd in seconds:

| profile | retune, 15 cases | retune, `t-*` | main, 15 cases | main, `t-*` | msd main | #34935 | retune |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xeon, 1 NUMA node, 22 threads | 0.98 (0.97-0.99) | 0.97 (0.97-0.98) | 1.35 | 1.06 | 2.82 | 2.53 | 2.73 |
| xeon, 1 socket, 86 threads | 0.84 (0.83-0.85) | 0.85 (0.84-0.86) | 1.53 | 1.06 | 3.88 | 4.88 | 3.28 |
| xeon, whole machine, 172 threads | 0.79 (0.74-0.80) | 0.78 (0.77-0.79) | 2.58 | 1.54 | 10.97 | 7.70 | 6.08 |
| xeon, 1 socket, `OMP_PROC_BIND=close` | 0.96 (0.95-0.97) | 1.03 (1.03-1.04) | | | | 3.16 | 4.26 |
| xeon, 1 socket, passive | 1.00 (0.97-1.03) | 0.98 (0.94-1.00) | | | | 66.2 | 59.7 |
| xeon, 1 NUMA node, passive | 0.98 (0.96-1.00) | 0.96 (0.93-0.97) | | | | 80.3 | 60.9 |
| gamer, 24 threads, libgomp | 0.89 (0.88-0.89) | 0.79 (0.77-0.80) | 5.06 | 3.29 | 14.06 | 24.25 | 4.80 |
| gamer, 24 threads, libomp | 0.92 (0.90-0.94) | 0.82 (0.80-0.85) | 2.79 | 1.63 | 6.56 | 17.79 | 5.55 |
| gamer, active forced | 1.03 (1.01-1.04) | 1.01 (1.00-1.03) | 1.24 | 1.06 | 2.03 | 2.00 | 2.21 |
| laptop, 16 threads, libgomp | 0.95 (0.92-1.01) | 0.96 (0.88-1.07) | 4.38 | 2.23 | 15.47 | 13.83 | 8.66 |
| laptop, active forced | 0.96 (0.87-1.17) | 0.97 (0.82-1.22) | 1.31 | 1.06 | 4.61 | 4.71 | 4.52 |
| mac-m2, 8 threads | 1.00 (0.99-1.00) | 1.00 (0.99-1.01) | 2.96 | 1.65 | 12.94 | 11.60 | 11.56 |
| mac-m2, active forced | 0.87 (0.81-0.93) | 0.97 (0.94-1.00) | 1.11 | 1.01 | 10.34 | 10.21 | 10.33 |

Worst medians on the default profiles:

| profile | vs #34935 | vs main |
| --- | --- | --- |
| xeon, 1 NUMA node | `t-30M` 1.17, `t-48M` 1.17, `L-thin` 1.10 | `t-48M` 1.22, `t-30M` 1.18 |
| xeon, 1 socket | `t-30M` 1.09, `t-22M` 1.08 | `t-22M` 1.29, `t-48M` 1.12, `kddcup09_churn` 1.09 |
| xeon, whole machine | `t-60M-l7` 1.09, `L-stumps` 1.07 | `t-150M` 1.24 |
| gamer, libgomp | `t-90M` 1.06 | none above 1 (`L-stumps` 0.95) |
| gamer, libomp | `L-stumps` 1.08 | none above 1 (`t-70M` 0.96) |
| laptop, libgomp | `t-70M` 1.20, `t-60M-l7` 1.17, `covtype` 1.14 | `t-60M-l7` 1.30 |
| mac-m2 | none above 1.01 | `kddcup09_churn` 1.03 |

- msd is faster than main on all 7 default profiles, without #42. On gamer with libomp, where #34935 is
  2.7x slower than main, the rule gets 5.55 s against 6.56 s.
- The biggest gains over #34935 come from fewer threads on small problems with active wait (`t-150K` is
  0.48x on one socket, 0.28x on the whole machine) and from the 20-75M band in passive mode (`t-22M` is
  0.21x on gamer with libgomp, 0.54x on the laptop).
- The Mac at 8 threads gets the same T as with #34935 on all 30 cases.
- `OMP_PROC_BIND=close` on one socket is the one real loss: `t-48M` 1.70x, `t-30M` 1.42x, msd 1.35x.
  #34935 puts 60 threads on several NUMA nodes there, and the rule's 15 stay on one. It's not a default
  setup, and without binding the same cases are 0.90x, 1.09x and 0.67x.
- `covtype` on the laptop has the same T=4 on both sides and goes from 0.82 to 1.18, so 1.14 is noise.
- The Mac with active wait forced gets 0.87 partly from `ames_housing` at 0.35x. #34935 is slow in this run
  (0.53 s vs 0.25 s in Exp 22), so I wouldn't read that case as a gain of 3x.

About the runs: of 4980 fits, one failed, `L-leaf255` on the rule, on gamer with active wait forced. It was a
SIGSEGV with CPython frames only, like the random ones known on this machine, and the rule changes no
compiled code. On the laptop with active wait forced, part of processes 4 and 5 is disturbed on all three builds
(one `susy` at 154 s instead of 14 s), I don't know by what. Medians hold, but the intervals of that row are
wide. The Mac stayed on AC, with one 2 s maintenance sleep during
`exp27-mac`. The laptop is on the `balanced` power profile.

## Merge of main into #34935 (2026-11-05)

Arthur asked to merge main into the PR while Exp 27 ran. 67a9bc9cdd is pushed on `hgb/active_wait`. One
conflict, at the end of `test_gradient_boosting.py`, keeping both sides. HGB and OpenMP helper tests: 388
passed, 105 skipped, and with active wait forced 369 passed, 105 skipped (on this thinkpad). The retune is rebased on it
and not pushed. Exp 27 ran before the merge, on 313be4c8.
