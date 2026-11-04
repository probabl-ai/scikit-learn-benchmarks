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

- fuse3 (2 regions and 3 barriers per split, instead of 7 regions and 7 barriers)
  is faster than the PR on every profile: geomean 0.76-0.90 over the 15
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
  before any of them starts. There is no barrier at the end of a `prange`.
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

## Cython version (LOG, last note)

All the results here come from builds with Cython 3.3.x, not the `<3.2.6` that
`pyproject.toml` pins. "No barrier at the end of a `prange`" only holds from 3.3.
The timings on the pinned versions are not measured yet.

## Open

- Repeat the A/B of `hgb/merge-regions` built with Cython 3.2.5.

- Why P and E cores differ with deep idle states, and why libomp and libgomp differ
  on P cores.
- Whether to use manual chunks (`nb`) or a dynamic schedule in a follow-up.
- Fixing the prange barrier in Cython.
