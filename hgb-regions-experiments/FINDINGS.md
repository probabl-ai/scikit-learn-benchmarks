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

- fuse3 (2 regions and 1 barrier per split instead of 7 regions) is faster
  than the PR on every profile: geomean 0.76-0.90 over the 15
  hgb_scalability cases, up to 3x on msd, and never clearly slower on a
  multi-threaded case. It also fixes the msd regression everywhere except
  gamer libomp (7.3 s vs main's 5.7 s).
- It helps even with active wait on one Xeon socket (0.76 geomean), where
  the microbenchmarks predicted only 1-2 us saved per region. Fewer Python
  round trips between regions probably matter there too.
- The 3-region version (fuse) gets about half of fuse3's gain.
- Once split finding lives in the histogram region, the only serial guard
  left is split_indices'. It still matters (none+fuse3 is the worst fused
  variant on msd), and the PR's per-thread rule is the safer one: the
  absolute 20K-sample rule helps a bit on 16-24 threads but hurts at 86.
- Energy follows time: an msd fit on gamer uses ~505 J with abs+fuse3 vs
  ~770 J with the PR (turbostat, Exp 11).

## Open

- One process of 3 fits per case and variant: repeat the multi-threaded
  cases to get the noise level.
- Why fused variants lose to abs alone on msd with libomp.
- A split_indices guard that works from 16 to 86 threads.
- Turn fuse3 into a reviewable change (no env vars) on top of #34935, or as
  a follow-up PR.
