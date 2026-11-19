## What does this PR do?



<!--
## Comparing a scikit-learn branch against main (optional)

To benchmark an upstream scikit-learn branch against `main` and get a
before/after dashboard on this PR, add a block like the one below OUTSIDE
this comment, and rename the fence tag from `DISABLED-sklbench-compare` to
`sklbench-compare`. The tag is disabled here because the workflows find the
block anywhere in the description, even inside a comment.

```DISABLED-sklbench-compare
sklearn_ref: cakedev0:ridge/optim_cholesky
runs: configs/hgb_scalability.py, intel-xeon#sklearn-dev-libomp#configs/pipeline.py
```

- `sklearn_ref`: `<fork owner>:<branch or ref>`.
- `runs`: comma-separated `[runner#][env#]configs/<name>.py` entries.
  runner: `intel-laptop`, `intel-xeon`, `aws-nvidia-l4` or `all` (every
  machine, default `all`);
  env: `sklearn-dev`, `sklearn-dev-libomp`, `sklearn-dev-mkl`,
  `sklearn-dev-freethreading`, `sklearn-dev-nvidia` or `auto` (default
  `auto`, resolved per machine: `sklearn-dev` on each Intel machine,
  `sklearn-dev-nvidia` on `aws-nvidia-l4`).
- Put `[skip main]` in a commit message to reuse the last published `main`
  results instead of re-benchmarking `main`.
- The branch must be pushed to this repo, not a fork.

Full reference: COMPARISONS_PR.md
-->
