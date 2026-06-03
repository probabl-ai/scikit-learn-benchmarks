# scikit-learn-benchmarks
Benchmarks scikit-learn-compatible machine learning libraries

Set-up:


```bash
git submodule update --init --recursive
```

Then you should be able to run `pixi run -e sklearn python -m sklbench --config configs/sklearn.json`.
And if you're on an intel machine, the entire `./test.sh`


TODO:
- record (and compare) meaningful attributes of models (e.g.: number of leaves, number of centroids, ...)
- custom synthetic data for trees
- max_bins, solver?
- better naming for files => some parts should be human-readable
- configure a test CI
- once first real benchmarks are running, configure a CI
- is hard interrupt handled?
- display:
    - per solver
    - click a point to see details?
    - ignore some params (e.g. max_bins)
