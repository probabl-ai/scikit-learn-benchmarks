# scikit-learn-benchmarks

User-facing benchmarks comparing **scikit-learn** and **scikit-learn-intelex** (sklearnex)
for common estimators, with the goal of showing achievable speed-up gains.

> **Benchmarks are re-run automatically every Monday** – results and plots in this
> repository are always up-to-date.

---

## 📊 Dashboard

Speed-up plots below compare **scikit-learn-intelex vs scikit-learn** (higher = faster with intelex).

### RandomForestClassifier

| Fit & Inference times | Speed-up |
|---|---|
| ![RandomForest times](plots/randomforestclassifier_times.png) | ![RandomForest speedup](plots/randomforestclassifier_speedup.png) |

### LogisticRegression

| Fit & Inference times | Speed-up |
|---|---|
| ![LogisticRegression times](plots/logisticregression_times.png) | ![LogisticRegression speedup](plots/logisticregression_speedup.png) |

### Ridge (RidgeRegression)

| Fit & Inference times | Speed-up |
|---|---|
| ![Ridge times](plots/ridge_times.png) | ![Ridge speedup](plots/ridge_speedup.png) |

---

## 📋 Benchmarked estimators & libraries

| Estimator | Libraries | Task |
|---|---|---|
| `RandomForestClassifier` | `sklearn`, `sklearnex` | Binary classification |
| `LogisticRegression` | `sklearn`, `sklearnex` | Binary classification |
| `Ridge` | `sklearn`, `sklearnex` | Regression |

Dataset sizes: **10,000** and **100,000** samples, 20 features.

---

## 🚀 Running benchmarks locally

### 1. Install dependencies

```bash
pip install -r envs/requirements.txt
```

### 2. Run benchmarks

```bash
python -m sklbench \
  --config configs/benchmarks.json \
  --result-file results/result.json
```

### 3. Generate visualisation plots

```bash
python visualisation/generate_plots.py \
  --result-files results/result.json \
  --output-dir plots
```

Open the generated `plots/` directory to view the PNG files.

---

## 🗂 Repository structure

```
scikit-learn-benchmarks/
├── .github/
│   └── workflows/
│       └── benchmarks.yml      # CI workflow (runs every Monday)
├── configs/
│   └── benchmarks.json         # Benchmark configuration
├── envs/
│   └── requirements.txt        # Python dependencies
├── results/
│   └── result.json             # Latest persisted benchmark results
├── plots/                      # Latest auto-generated plots (PNG)
└── visualisation/
    └── generate_plots.py       # Script to generate plots from results
```

---

## 🔗 References

- [scikit-learn](https://scikit-learn.org/)
- [scikit-learn-intelex](https://intel.github.io/scikit-learn-intelex/)
- [scikit-learn_bench](https://github.com/cakedev0/scikit-learn_bench) – benchmark runner used by this repo
