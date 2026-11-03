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

## Pending

- Exp 10 A/B of the fused variants and absolute guards on gamer, laptop and
  xeon (one socket, active and passive).
- Tune the `abs` thresholds if they turn out to matter.
