# HGB parallel regions and CPU idle states

Tracking branch (not meant to be merged into `main`) for two questions about
[scikit-learn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935)
("PRF Mitigation heuristics for HGB scalability", head 313be4c8):

1. Why is the PR ~3x slower than its base on `year_prediction_msd` on
   `intel-gamer-gpu`, while it speeds up most other cases?
2. Can HGB pay less for its OpenMP parallel regions on passive-wait runtimes
   (fewer regions, cheaper serial guards) without losing the PR's gains?

The scikit-learn side lives on
[`cakedev0/scikit-learn@hgb/fewer-regions`](https://github.com/cakedev0/scikit-learn/tree/hgb/fewer-regions),
a branch on top of the PR. Its variants are selected at fit time with
environment variables, so all of them run from one build:

| variable | values |
| --- | --- |
| `SKLEARN_HGB_GUARD` | `pr` (default, the PR's per-thread work guards), `abs` (absolute work guards, thresholds in `SKLEARN_HGB_ABS_SAMPLES`/`SKLEARN_HGB_ABS_BINS`), `none` (always threaded, like main) |
| `SKLEARN_HGB_EXP` | `fuse` (3 regions per split), `fuse2` (2 regions, 2 barriers), `fuse3` (2 regions, 1 barrier) |

## Layout

- `LOG.md`: chronological experiment log, one entry per run with its
  purpose, command and outcome. Times are UTC.
- `FINDINGS.md`: the conclusions so far.
- `src/`: C OpenMP microbenchmarks (`wake.c`, `wake2.c`: wake-up cost after
  an idle gap; `team.c`: changing the team size; `barrier.c`: two regions vs
  one region with a barrier).
- `scripts/microbench/`: their run scripts, as run on gamer from
  `~/work/gamer-run/{wake,team,barrier}`.
- `scripts/gamer-msd/`: the `year_prediction_msd` case and the scripts that
  ran it on gamer from `~/work/gamer-run/msd` (thread sweeps, OS counters,
  C-state A/B).
- `scripts/ab/`: the interleaved A/B of the scikit-learn variants. On each
  host the files go to `~/work/hgb-regions`.
- `scripts/segv/`: the segfault reproduction attempt (side topic).
- `results/<machine>/`: raw outputs. `results/gamer/session_outputs.md` holds
  outputs that were only printed to the terminal.

## Running the A/B

On a host, with this folder's `scripts/ab/*` copied to `~/work/hgb-regions`:

    bash setup.sh   # clones the branch and the PR's base commit, makes 2 venvs
    benchlock --tag "hgb fewer-regions A/B" all -- \
        bash run_all.sh 24 'gamer-gomp||' 'gamer-omp||LD_PRELOAD=.../libomp.so'

`run_all.sh` rebuilds both venvs (`build.sh`) and then runs `bench.py` once
per `PROFILE|VARIANTS|KEY=VAL ...` spec. `bench.py` takes the
`configs/hgb_scalability.py` cases at one thread count and runs each
(case, variant) as one sklbench runner subprocess, with the variants shuffled
per case. Results land in `results_<profile>.jsonl`.

The venvs are built on the `sklearn-dev` pixi env but drop its editable hooks
(`sitecustomize.py`), so they never touch the shared `sklearn-src` checkout.
