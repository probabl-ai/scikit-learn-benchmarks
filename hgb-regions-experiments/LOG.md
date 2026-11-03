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

Results: pending.

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
