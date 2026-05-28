# scikit-learn-benchmarks
Benchmarks scikit-learn-compatible machine learning libraries

Set-up:
- Clone https://github.com/cakedev0/scikit-learn_bench/
- Checkout branch `probabl-fork`
- `ln -s scikit-learn_bench/sklbench sklbench`
- Then you should be able to run `pixi run -e sklearn python -m sklbench --config configs/sklearn.json`
  And if you're on an intel machine, the entire `./test.sh`


TODO:
- [paused] first CI set-up, from Olivier's laptop
- record (and compare) meaningful attributes of models (e.g.: number of leaves, number of centroids, ...)
- custom synthetic data for trees
