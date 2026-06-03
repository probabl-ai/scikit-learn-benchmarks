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
- configure a test CI
- once first real benchmarks are running, configure a CI
- display: solver?
