# Barrier at the start of Cython's `prange`

Tracking branch (not meant to be merged into `main`) for two questions:

1. Why does every `prange` start with a barrier on GCC, and can Cython drop it safely?
2. What does that gain on scikit-learn with passive OpenMP runtimes?

Nothing here has been posted upstream.

## Layout

- `LOG.md`: chronological log of the runs. Times are UTC.
- `probes/`: small C and Cython files, each one answering one question.
- `scripts/`: how the probes were compiled and run. All of them run under `benchlock`.
- `results/`: raw outputs.
