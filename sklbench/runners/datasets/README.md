# Data Processing and Storage in Benchmarks

Data handling steps:

1. Load the data:
   - if not cached, download or generate the dataset and store it in the raw
     and/or regular cache;
   - if cached, load it from the cached files.
2. Split the data into subsets if requested.
3. Convert it to the requested form (dtype, format, memory order, etc.).

Data sources:

- synthetic data from sklearn
- OpenML datasets
- custom loaders for named datasets

## Named Real-World Datasets

These names can be used as `data:dataset` values. `n_features` is the number
of features of the `x` returned by the loader, after loader-specific steps
such as target extraction, dropped columns, category encoding or one-hot
encoding.

| Name | n_samples | n_features | Task(s) | Features |
| --- | ---: | ---: | --- | --- |
| `airline_depdelay` | ~115,000,000 | 13 | Classification | Numerical, categorical |
| `a9a` | 48,842 | 123 | Classification | Numerical / binary indicators |
| `codrnanorm` | 488,565 | 8 | Classification | Numerical |
| `covtype` | 581,012 | 54 | Classification | Numerical, binary indicators |
| `creditcard` | 284,807 | 30 | Classification | Numerical |
| `digits` | 1,797 | 64 | Classification | Numerical pixels |
| `fraud` | 284,807 | 30 | Classification | Numerical |
| `gisette` | 7,000 | 5,000 | Classification | Numerical |
| `hepmass` | 10,500,000 | 27 | Classification | Numerical |
| `higgs` | 11,000,000 | 28 | Classification | Numerical |
| `susy` | 5,000,000 | 18 | Classification | Numerical |
| `ijcnn` | 191,681 | 22 | Classification | Numerical |
| `klaverjas` | 981,541 | 35 | Classification | Numerical, categorical/string |
| `cifar` | 60,000 | 3,072 | Classification | Numerical pixels |
| `connect` | 67,557 | 126 | Classification | Numerical / binary indicators |
| `covertype` | 581,012 | 54 | Classification | Numerical, categorical |
| `skin_segmentation` | 245,057 | 3 | Classification | Numerical |
| `mnist` | 70,000 | 784 | Classification | Numerical pixels |
| `fashion_mnist` | 70,000 | 784 | Classification | Numerical pixels |
| `svhn` | 99,289 | 3,072 | Classification | Numerical pixels |
| `sensit` | 98,528 | 100 | Classification | Numerical |
| `letters` | 20,000 | 16 | Classification | Numerical |
| `szilard_1m` | 1,100,000 | 9 | Classification | Numerical, categorical |
| `szilard_10m` | 10,100,000 | 9 | Classification | Numerical, categorical |
| `abalone` | 4,177 | 8 | Regression | Numerical, encoded categorical |
| `california_housing` | 20,640 | 8 | Regression | Numerical |
| `fried` | 40,768 | 10 | Regression | Numerical |
| `medical_charges_nominal` | 163,065 | 11 | Regression | Numerical, categorical |
| `twodplanes` | 177,147 | 10 | Regression | Numerical |
| `year_prediction_msd` | 515,345 | 90 | Regression | Numerical |
| `yolanda` | 400,000 | 100 | Regression | Numerical |
| `road_network` | 434,874 | 3 | Regression | Numerical |
| `sift` | 1,010,000 | 128 | Index / nearest-neighbor search | Numerical |
| `gist` | 1,001,000 | 960 | Index / nearest-neighbor search | Numerical |

## Data Caching

There are two cache levels, each with its own directory: the raw cache for
files downloaded from external sources, and the regular cache for files that
load fast in benchmarks.

Each dataset has a few files in the regular cache: data component files (`x`,
`y`, `weights`, etc.) and a JSON file with dataset properties (number of
classes, clusters, default split arguments). For example:

```
data_cache/
...
├── mnist.json
├── mnist_x.parq
├── mnist_y.npz
...
```

Cached file formats:

| Format | File extension | Associated Python types | Comment |
| --- | --- | --- | --- |
| [Parquet](https://parquet.apache.org) | `.parq` | pandas.DataFrame |  |
| NumPy uncompressed binary dense data | `.npz` | numpy.ndarray, pandas.Series | Data is stored under the `arr_0` name |
