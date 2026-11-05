# Findings so far

## The year_prediction_msd regression on intel-gamer-gpu

- With a passive OpenMP runtime, what a parallel region costs depends on
  how deep the sleeping workers went. From C1, waking 22 threads takes
  ~12 us. From C2 or C3 it takes ~200 us, and the woken threads also run
  their work 2-3x slower (busy time in `wake2`). See LOG Exp 4 and 6.
- The PR's passive guards make `find_node_split` (90 features < 5 x 23
  threads) and `split_indices` (< 115K samples) serial. Those serial phases
  are long enough for the `menu` governor to put the workers in C2/C3, so
  the *next* parallel region (histograms) pays a deep wake. Histogram time
  goes from 2.6 to 13.6 s (Exp 2).
- With only C1 allowed, the PR default fits in 4.2 s instead of 16.6 s
  (Exp 6). Even then, the serial find_split loses to the parallel one
  (3.3 s).
- Disabling only C3 made everything slower: the governor then picks C2 even
  for short idles (Exp 5). Don't use that as a test.
- This depends on the hardware and the idle governor. A server with
  shallow idle states, or an active-wait runtime, won't show it.

## OpenMP runtime behavior

- libgomp destroys the threads beyond a smaller team and creates them again
  when the team grows: alternating 23/4 threads costs ~290 us per regrow,
  even when spinning. libomp keeps its threads (Exp 7). This is the likely
  reason a dynamic thread count went badly before: on libgomp, a "partial
  team" must keep the full team and give work to fewer threads.
- In passive mode, a barrier inside a region costs ~10 us, about the same
  as a back-to-back region. A region after a serial gap of 30 us or more
  often costs 400-600 us. Merging regions pays off when it removes the
  serial code between them (Exp 8). When spinning, all of this is 1-4 us.

## Merging parallel regions (LOG Exp 9-10)

- fuse3 (2 regions and 3 barriers per split, instead of 7 regions and 7 barriers;
  `hgb/merge-regions` adds an explicit barrier, see Exp 20) is faster than the PR on every profile: geomean 0.76-0.90 over the 15
  hgb_scalability cases, up to 3x on msd. It also fixes the msd regression
  everywhere except gamer libomp (7.3 s vs main's 5.7 s).
- It helps even with active wait on one Xeon socket (0.76 geomean), where
  the microbenchmarks predicted only 1-2 us saved per region. Fewer Python
  round trips between regions probably matter there too.
- The 3-region version (fuse) gets about half of fuse3's gain.
- Energy follows time: an msd fit on gamer uses ~505 J with abs+fuse3 vs
  ~770 J with the PR (turbostat, Exp 11).

## The barrier at the start of prange (LOG Exp 13)

- On GCC, every Cython `prange` starts with a barrier, a side effect of marking
  its variables both firstprivate and lastprivate. All the threads must be awake
  before any of them starts. From Cython 3.3, there is no barrier at the end of a
  `prange`. Before that, there is one when code follows it in the `parallel` block.
- Manual chunks in a `parallel` block avoid it: `nb:pr+fuse3` takes 0.27-0.53x of
  `pr` on one Xeon socket in passive mode, against 0.38-0.63x for `pr+fuse3`, and
  0.60-0.96x against 0.64-0.97x with active wait.
- Removing it isn't always a gain: on gamer P cores with deep idle states, `nb` is
  the slowest variant (7.0 s vs 5.15 s for `abs+fuse3`, libomp).
- The fix in Cython itself is the topic of `cython_prange_barrier_prompt.md`.

## Why merging loses on msd with libomp (LOG Exp 12, 14, 15, 17)

- It comes from deep idle states: with only C1 allowed, `abs+fuse3` beats `abs` on
  every profile (24 threads 2.16 vs 3.45 s, 8 P cores 3.18 vs 3.97 s) and both
  runtimes give the same times.
- With all idle states, merging replaces many short waits by fewer long ones, and
  deep entries go up (24 threads, libomp: 0.33 M for `abs`, 0.69 M for `abs+fuse3`),
  although the number of waits goes down. libomp then loses its advantage over
  libgomp: `abs` 6.0 vs 11.95 s, `abs+fuse3` 8.17 vs 8.67 s.
- It depends on the cores. On 16 E cores merging wins with few deep entries. On 8 P
  cores it loses with libomp (5.15 vs 4.30 s), and not with libgomp (4.06 vs
  4.32 s). I don't know why P and E cores differ, or why libomp and libgomp differ
  on P cores. Possibly a P core loses more in a deep state, since it has its own L2
  while E cores share theirs by cluster, but I didn't test that.
- A dynamic schedule on the merged feature loop reduces the loss with deep states
  (24 threads, Exp 15: 5.82 vs 8.60 s with libomp, 5.78 vs 8.51 s with libgomp), but
  it is slower with only C1 (2.53 vs 2.16 s). So it isn't a free fix.
- Passive is what both runtimes report by default on the hybrid CPU here (only
  libgomp checked on the laptop), with no OpenMP variable set.

## The split_indices guard (LOG Exp 16)

- Once split finding lives in the histogram region, the only serial guard left is
  split_indices'. Always threaded is up to 2.2x slower on one Xeon socket, always
  serial up to 1.5x slower on susy on gamer.
- I found no simple rule that clearly beats the PR's 5000 samples per thread. It is
  within ~15% of the best threshold on most cases. A threshold of 10K-30K loses at
  32 and 86 threads on the xeon (msd 21-26 s vs ~18 s). The best threshold looks
  like a property of the machine (~100K or less on gamer, ~300K on the xeon at both
  32 and 86 threads), not of the thread count.

## Noise and the clean branch (LOG Exp 18)

- Per-case spread of `pr` between processes: 1-5% on most small cases, up to ~20% on
  some multi-threaded ones. Geomean over 15 cases: about +-0.01-0.02.
- `hgb/merge-regions`, the clean fuse3, gives 0.76x of the PR on one Xeon socket with
  active wait, 0.73x in passive mode, 0.87x on the laptop and 0.87x on gamer with
  both runtimes.
  Its trees and predictions are identical to the PR's, and to main's.

## Cython version (LOG Exp 20)

Exp 8-19 used Cython 3.3.x, not the `<3.2.6` that `pyproject.toml` pins. On 3.2.5:

- `hgb/merge-regions` gives 0.74x of #34935 on one Xeon socket with active wait, 0.77x
  in passive mode, 0.86x on the laptop, 0.89x on gamer with libgomp and 0.88x with
  libomp. That's within 0.03 of 3.3. Trees are identical to main's and the HGB tests pass.
- msd against main: 2.81 vs 3.89 s, 23.9 vs 178 s, 4.82 vs 15.0 s, 9.94 vs 12.8 s and,
  still slower, 8.81 vs 6.48 s on gamer with libomp.
- A split runs 5 barriers on 3.2.5, 4 on 3.3, with GCC. The explicit one is redundant on
  3.2.5, with no clear cost.
- The same source is faster on 3.3: 2-9% on this branch, 1-6% on #34935, 7-19% on main's
  msd. I don't know why, it isn't explained by the barrier sites of `main`.

The other findings on this page were not repeated on 3.2.5.

## mac-m2 and aws-l4-1 (LOG Exp 21)

- `hgb/merge-regions` is 0.87x of #34935 on mac-m2 by default, 0.92x with active wait
  forced. On aws-l4-1, it's 0.95x at 2 threads (the default), 0.93x at 4 threads and 0.89x
  at 2 threads in passive mode. Trees are identical and the HGB tests pass.
- mac-m2 is passive by default, like the hybrid Intel machines, so #34935's gain over main
  is largest there (0.34x geomean, up to 18x on `XS`).
- #34935 has no msd regression on these two. msd is 7.7 s vs main's 13.0 s on the Mac and
  15.3 s for all three on aws-l4-1 at 2 threads.

## Retuning the thread rule of #34935 (LOG Exp 22-27)

- #34935 gives too many threads: to small and mid problems with active wait, and in its top band in
  passive mode. A cap by `n_samples * n_features` and wait mode fixes most of it, and the guards can
  stay (Exp 23-24: once the cap is there, no guard setting wins by more than 1%).
- The rule, on `hgb/retune-threads` (local, 9d7e7756f4), in values:
  active 200K:4, 5M:8, 50M:16, 75M:32, then 64. Passive 2M:1, 20M:4, 75M:8, then 16.
- Geomean against #34935, on the 15 `hgb_scalability` cases then on 15 shapes not used for tuning: 0.98
  and 0.97 on one Xeon NUMA node, 0.84 and 0.85 on one socket, 0.79 and 0.78 on the whole machine, 0.89
  and 0.79 on gamer with libgomp, 0.92 and 0.82 with libomp, 0.95 and 0.96 on the laptop, 1.00 and 1.00 on
  mac-m2.
- It fixes the msd regression without #42: main, #34935, rule are 14.06, 24.25, 4.80 s on gamer with
  libgomp, 6.56, 17.79, 5.55 s with libomp, 15.47, 13.83, 8.66 s on the laptop, 3.88, 4.88, 3.28 s on one
  socket, 10.97, 7.70, 6.08 s on the whole machine.
- Worst medians against #34935 on default profiles: 1.20x (`t-70M`, laptop), 1.17x (`t-30M` and `t-48M`,
  NUMA node). Against main: 1.30x (`t-60M-l7`, laptop), 1.29x (`t-22M`, socket), 1.24x (`t-150M`, whole
  machine).
- The clear loss is `OMP_PROC_BIND=close` on one socket: geomean 0.96 and 1.03, `t-48M` 1.70x. With the
  threads bound together, the fewer threads of the rule stay on one NUMA node.

What this doesn't cover or settle:

- Apart from the 6 datasets of `hgb_scalability`, which served for tuning, all shapes are synthetic.
- libomp on the Xeon and Windows are untested. The laptop ran on the `balanced` power profile.
- The passive limit of 75M isn't a measured optimum: gamer and the laptop disagree between 36M and 100M,
  and anywhere from about 50M to 100M does about as well (Exp 26).
- The active guards were only swept on the laptop and the Mac, not on the Xeon or gamer (Exp 24).
- The number of leaves isn't in the rule, although work per region depends on it.
- The benchmarks ran on 313be4c8 plus the rule, and the branch is rebased on the merge of main. Only the
  tests ran on the rebased commit.

## Decisions

- Manual chunks (`nb`) are dropped.
- `hgb/merge-regions` goes up as a follow-up to #34935, after review. It wasn't part of the retune, and
  its numbers on this page are against #34935 without the rule.
- The active cap stays at 16 between 20M and 50M (Exp 26b: cap 20 is 0.95 on a NUMA node, 1.02 on a socket).

## Open

- How the rule reaches #34935.
- Whether the number of leaves should enter the rule.
- Why P and E cores differ with deep idle states, and why libomp and libgomp differ
  on P cores.
- Whether a dynamic schedule on the merged feature loop is worth a follow-up.
- Fixing the prange barrier in Cython.
