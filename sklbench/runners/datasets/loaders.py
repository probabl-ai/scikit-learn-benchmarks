# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import os

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_digits,
)

from .downloaders import download_and_read_csv, load_openml, retrieve


def load_openml_data(openml_id: int, raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(openml_id, raw_data_cache)
    data_desc = dict()
    unique_labels = pd.Series(y).value_counts()
    if len(unique_labels) < 32 and (unique_labels > 4).all():
        data_desc["n_classes"] = len(unique_labels)
    return {"x": x, "y": y}, data_desc


"""
Classification datasets
"""


def load_airline_depdelay(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Airline dataset
    http://kt.ijs.si/elena_ikonomovska/data.html

    Classification task. n_classes = 2.
    """
    url = "http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"

    ordered_columns = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    categorical_int_columns = ["Year", "Month", "DayofMonth", "DayofWeek"]
    continuous_int_columns = [
        "CRSDepTime",
        "CRSArrTime",
        "FlightNum",
        "ActualElapsedTime",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]
    column_dtypes = {
        col: np.int16 for col in categorical_int_columns + continuous_int_columns
    }

    df = download_and_read_csv(
        url, raw_data_cache, names=ordered_columns, dtype=column_dtypes
    )

    for col in df.select_dtypes(["object"]).columns:
        df[col] = df[col].astype("category")

    df["ArrDelay"] = (df["ArrDelay"] > 0).astype(int)

    y = df["ArrDelay"].to_numpy(dtype=np.float32)
    x = df.drop(columns=["ArrDelay"])

    data_description = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_description


def load_hepmass(raw_data_cache: str) -> tuple[dict, dict]:
    """
    HEPMASS dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/HEPMASS.

    Classification task. n_classes = 2.
    """
    url_train = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz"
    )
    url_test = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_test.csv.gz"
    )

    dtype = np.float32
    train_data = download_and_read_csv(
        url_train, raw_data_cache, delimiter=",", compression="gzip", dtype=dtype
    )
    test_data = download_and_read_csv(
        url_test, raw_data_cache, delimiter=",", compression="gzip", dtype=dtype
    )

    data = pd.concat([train_data, test_data])
    label = data.columns[0]
    y = data[label]
    x = data.drop(columns=[label, "mass"])

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": train_data.shape[0],
            "test_size": test_data.shape[0],
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


def load_higgs_susy_subsample(dataset: str, raw_data_cache: str) -> tuple[dict, dict]:
    if dataset == "susy":
        """
        SUSY dataset from UCI machine learning repository
        https://archive.ics.uci.edu/ml/datasets/SUSY

        Classification task. n_classes = 2.
        """
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"
        )
        train_size, test_size = 4500000, 500000
    elif dataset == "higgs":
        """
        Higgs dataset from UCI machine learning repository
        https://archive.ics.uci.edu/ml/datasets/HIGGS

        Classification task. n_classes = 2.
        """
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        )
        train_size, test_size = 10000000, 1000000
    else:
        raise ValueError(
            f'Unknown dataset name {dataset} for "load_higgs_susy_subsample" function'
        )

    data = download_and_read_csv(
        url, raw_data_cache, delimiter=",", header=None, compression="gzip"
    )
    assert data.shape[0] == train_size + test_size, "Wrong number of samples was loaded"
    x, y = data[data.columns[1:]], data[data.columns[0]]

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": train_size,
            "test_size": test_size,
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


def load_higgs(raw_data_cache: str) -> tuple[dict, dict]:
    return load_higgs_susy_subsample("higgs", raw_data_cache)


def load_susy(raw_data_cache: str) -> tuple[dict, dict]:
    return load_higgs_susy_subsample("susy", raw_data_cache)


def load_letters(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Letter Recognition dataset from UCI machine learning repository
    http://archive.ics.uci.edu/ml/datasets/Letter+Recognition

    Classification task. n_classes = 26.
    """
    url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/"
        "letter-recognition/letter-recognition.data"
    )
    data = download_and_read_csv(url, raw_data_cache, header=None, dtype=None)
    x, y = data.iloc[:, 1:], data.iloc[:, 0].astype("category").cat.codes.values

    data_desc = {
        "n_classes": 26,
        "default_split": {"test_size": 0.2, "random_state": 0},
    }
    return {"x": x, "y": y}, data_desc


def load_sklearn_digits(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_digits(return_X_y=True)
    data_desc = {
        "n_classes": 10,
        "default_split": {"train_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_covtype(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Cover type dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/covertype

    y contains 7 unique class labels from 1 to 7 inclusive.
    Classification task. n_classes = 7.
    """
    x, y = fetch_covtype(return_X_y=True, data_home=raw_data_cache)
    y = y.astype(int) - 1

    data_desc = {
        "n_classes": 7,
        "default_split": {"test_size": 0.2, "random_state": 77},
    }
    return {"x": x, "y": y}, data_desc


def load_gisette(raw_data_cache: str) -> tuple[dict, dict]:
    """
    GISETTE is a handwritten digit recognition problem.
    The problem is to separate the highly confusable digits '4' and '9'.
    This dataset is one of five datasets of the NIPS 2003 feature selection challenge.

    Classification task. n_classes = 2.
    """

    def convert_x(x, n_samples, n_features):
        x_out = x.iloc[:n_samples].values
        x_out = pd.DataFrame(
            np.array(
                [
                    np.fromstring(elem[0], dtype=int, count=n_features, sep=" ")
                    for elem in x_out
                ]
            )
        )
        return x_out.values

    def convert_y(y, n_samples):
        y_out = y.iloc[:n_samples].values.astype(int)
        y_out = pd.DataFrame((y_out > 0).astype(int))
        return y_out.values.reshape(-1)

    url_prefix = "http://archive.ics.uci.edu/ml/machine-learning-databases"
    data_urls = {
        "x_train": f"{url_prefix}/gisette/GISETTE/gisette_train.data",
        "x_test": f"{url_prefix}/gisette/GISETTE/gisette_valid.data",
        "y_train": f"{url_prefix}/gisette/GISETTE/gisette_train.labels",
        "y_test": f"{url_prefix}/gisette/gisette_valid.labels",
    }
    data = {}
    for subset_name, subset_url in data_urls.items():
        data[subset_name] = download_and_read_csv(subset_url, raw_data_cache, header=None)

    n_columns, train_size, test_size = 5000, 6000, 1000

    x_train = convert_x(data["x_train"], train_size, n_columns)
    x_test = convert_x(data["x_test"], test_size, n_columns)
    y_train = convert_y(data["y_train"], train_size)
    y_test = convert_y(data["y_test"], test_size)

    x = np.vstack([x_train, x_test])
    y = np.hstack([y_train, y_test])

    data_desc = {
        "n_classes": 2,
        "default_split": {
            "train_size": y_train.shape[0],
            "test_size": y_test.shape[0],
            "shuffle": False,
        },
    }
    return {"x": x, "y": y}, data_desc


def load_a9a(raw_data_cache: str) -> tuple[dict, dict]:
    def transform_x_y(x, y):
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1430, raw_data_cache, transform_x_y)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 11},
    }
    return {"x": x, "y": y}, data_desc


def load_codrnanorm(raw_data_cache: str) -> tuple[dict, dict]:
    def transform_x_y(x, y):
        x = pd.DataFrame(x)
        y = y.astype("int")
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1241, raw_data_cache, transform_x_y_func=transform_x_y)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_creditcard(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(1597, raw_data_cache)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.1, "random_state": 777},
    }
    return {"x": x, "y": y}, data_desc


def load_fraud(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(42175, raw_data_cache)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 77},
    }
    return {"x": x, "y": y}, data_desc


def load_ijcnn(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Author: Danil Prokhorov.
    libSVM,AAD group
    Cite: Danil Prokhorov. IJCNN 2001 neural network competition.
    Slide presentation in IJCNN'01,
    Ford Research Laboratory, 2001. http://www.geocities.com/ijcnn/nnc_ijcnn01.pdf.

    Classification task. n_classes = 2.
    """

    def transform_x_y(x, y):
        y[y == -1] = 0
        return x, y

    x, y = load_openml(1575, raw_data_cache, transform_x_y)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_klaverjas(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Abstract:
    Klaverjas is an example of the Jack-Nine card games,
    which are characterized as trick-taking games where the the Jack
    and nine of the trump suit are the highest-ranking trumps, and
    the tens and aces of other suits are the most valuable cards
    of these suits. It is played by four players in two teams.

    Task Information:
    Classification task. n_classes = 2.
    """
    x, y = load_openml(41228, raw_data_cache)
    data_desc = {
        "n_classes": 2,
        "default_split": {"train_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_skin_segmentation(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Abstract:
    The Skin Segmentation dataset is constructed over B, G, R color space.
    Skin and Nonskin dataset is generated using skin textures from
    face images of diversity of age, gender, and race people.
    Author: Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.

    Classification task. n_classes = 2.
    """

    def transform_x_y(x, y):
        y = y.astype(int)
        y[y == 2] = 0
        return x, y

    x, y = load_openml(1502, raw_data_cache, transform_x_y)
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_cifar(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Source:
    University of Toronto
    Collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
    https://www.cs.toronto.edu/~kriz/cifar.html

    Classification task. n_classes = 10.
    """
    x, y = load_openml(40927, raw_data_cache)
    data_desc = {
        "n_classes": 10,
        "default_split": {"test_size": 1 / 6, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_connect(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Source:
    UC Irvine Machine Learning Repository
    http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.htm

    Classification task. n_classes = 3.
    """
    x, y = load_openml(1591, raw_data_cache)
    y = (y + 1).astype("int")
    data_desc = {
        "n_classes": 3,
        "default_split": {"test_size": 0.1, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_covertype(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Abstract: This is the original version of the famous
    covertype dataset in ARFF format.
    Author: Jock A. Blackard, Dr. Denis J. Dean, Dr. Charles W. Anderson
    Source: [original](https://archive.ics.uci.edu/ml/datasets/covertype)

    Classification task. n_classes = 7.
    """
    x, y = load_openml(1596, raw_data_cache)
    data_desc = {
        "n_classes": 7,
        "default_split": {"test_size": 0.4, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_mnist_template(
    openml_id: int,
    raw_data_cache: str,
) -> tuple[dict, dict]:
    def transform_x_y(x, y):
        return x.astype("uint8"), y.astype("uint8")

    x, y = load_openml(openml_id, raw_data_cache, transform_x_y)
    data_desc = {
        "n_classes": 10,
        "default_split": {"test_size": 10000, "shuffle": False},
    }
    return {"x": x, "y": y}, data_desc


def load_mnist(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Abstract:
    The MNIST database of handwritten digits with 784 features.
    It can be split in a training set of the first 60,000 examples,
    and a test set of 10,000 examples
    Source:
    Yann LeCun, Corinna Cortes, Christopher J.C. Burges
    http://yann.lecun.com/exdb/mnist/

    Classification task. n_classes = 10.
    """
    return load_mnist_template(554, raw_data_cache)


def load_fashion_mnist(raw_data_cache: str) -> tuple[dict, dict]:
    return load_mnist_template(40996, raw_data_cache)


def load_svhn(raw_data_cache: str) -> tuple[dict, dict]:
    return load_mnist_template(41081, raw_data_cache)


def load_sensit(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Abstract: Vehicle classification in distributed sensor networks.
    Author: M. Duarte, Y. H. Hu
    Source: [original](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets)

    Classification task. n_classes = 3.
    """
    x, y = load_openml(1593, raw_data_cache)
    data_desc = {
        "n_classes": 3,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_szilard_template(train_url: str, raw_data_cache: str) -> tuple[dict, dict]:
    """
    https://github.com/szilard/GBM-perf

    Returns the raw (not one-hot encoded) features, with string columns cast to
    pandas `category` dtype, so estimators with native categorical support can
    use them as-is. Use `preprocessing_kwargs: {"category_encoding": "onehot"}`
    to get a numeric-only encoding instead.
    """
    d_train = download_and_read_csv(train_url, raw_data_cache)

    test_url = "https://s3.amazonaws.com/benchm-ml--main/test.csv"
    d_test = download_and_read_csv(test_url, raw_data_cache)

    label_col = "dep_delayed_15min"
    y_train = (d_train[label_col] == "Y").astype(int).values
    y_test = (d_test[label_col] == "Y").astype(int).values
    y = np.concatenate([y_train, y_test])

    x_train = d_train.drop(columns=[label_col])
    x_test = d_test.drop(columns=[label_col])
    x = pd.concat([x_train, x_test], axis=0, ignore_index=True)
    for col in x.select_dtypes(["object"]).columns:
        x[col] = x[col].astype("category")

    n_train = len(d_train)
    n_test = len(d_test)
    data_desc = {
        "default_split": {"train_size": n_train, "test_size": n_test, "shuffle": False}
    }

    return {"x": x, "y": y}, data_desc


def load_szilard_1m(raw_data_cache: str) -> tuple[dict, dict]:
    return load_szilard_template(
        "https://s3.amazonaws.com/benchm-ml--main/train-1m.csv", raw_data_cache
    )


def load_szilard_10m(raw_data_cache: str) -> tuple[dict, dict]:
    return load_szilard_template(
        "https://s3.amazonaws.com/benchm-ml--main/train-10m.csv", raw_data_cache
    )


"""
Regression datasets
"""


def load_abalone(raw_data_cache: str) -> tuple[dict, dict]:
    """
    https://archive.ics.uci.edu/ml/machine-learning-databases/abalone

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    data = download_and_read_csv(url, raw_data_cache, header=None)
    data[0] = data[0].astype("category").cat.codes
    x, y = data.iloc[:, :-1], data.iloc[:, -1].values

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 0}}
    return {"x": x, "y": y}, data_desc


def load_california_housing(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = fetch_california_housing(
        return_X_y=True, as_frame=False, data_home=raw_data_cache
    )
    data_desc = {"default_split": {"test_size": 0.1, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_fried(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(564, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_medical_charges_nominal(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(42559, raw_data_cache)

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_ames_housing(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Ames Housing dataset (openml id 42165).

    1460 samples, 79 features, a mix of numeric and (mostly `object`-dtype)
    categorical columns, with real missing values in both. Small enough to be
    used in the "test" tier of configs.

    Regression task: predict `SalePrice`.
    """
    x, y = load_openml(42165, raw_data_cache, as_frame=True)

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_twodplanes(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(1197, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.4, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_year_prediction_msd(raw_data_cache: str) -> tuple[dict, dict]:
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
        "YearPredictionMSD.txt.zip"
    )
    data = download_and_read_csv(url, raw_data_cache, header=None)
    x, y = data.iloc[:, 1:], data.iloc[:, 0]
    data_desc = {"default_split": {"test_size": 0.1, "shuffle": False}}
    return {"x": x, "y": y}, data_desc


def load_yolanda(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(42705, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_road_network(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
    n_samples, dtype = 20000, np.float32
    data = download_and_read_csv(url, raw_data_cache, dtype=dtype)
    x, y = data.values[:, 1:], data.values[:, 0]
    data_desc = {
        "default_split": {
            "train_size": n_samples,
            "test_size": n_samples,
            "shuffle": False,
        }
    }
    return {"x": x, "y": y}, data_desc


"""
Index/neighbors search datasets
"""


def load_ann_dataset_template(url, raw_data_cache):
    import h5py

    local_path = os.path.join(raw_data_cache, os.path.basename(url))
    retrieve(url, local_path)
    with h5py.File(local_path, "r") as f:
        x_train = np.asarray(f["train"])
        x_test = np.asarray(f["test"])
    x = np.concatenate([x_train, x_test], axis=0)
    data_desc = {
        "default_split": {
            "train_size": x_train.shape[0],
            "test_size": x_test.shape[0],
            "shuffle": False,
        }
    }
    del x_train, x_test
    # TODO: remove placeholding zeroed y
    y = np.zeros((x.shape[0],))
    return {"x": x, "y": y}, data_desc


def load_sift(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    return load_ann_dataset_template(url, raw_data_cache)


def load_gist(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    return load_ann_dataset_template(url, raw_data_cache)


dataset_loading_functions = {
    # classification
    "airline_depdelay": load_airline_depdelay,
    "a9a": load_a9a,
    "codrnanorm": load_codrnanorm,
    "covtype": load_covtype,
    "creditcard": load_creditcard,
    "digits": load_sklearn_digits,
    "fraud": load_fraud,
    "gisette": load_gisette,
    "hepmass": load_hepmass,
    "higgs": load_higgs,
    "susy": load_susy,
    "ijcnn": load_ijcnn,
    "klaverjas": load_klaverjas,
    "cifar": load_cifar,
    "connect": load_connect,
    "covertype": load_covertype,
    "skin_segmentation": load_skin_segmentation,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "svhn": load_svhn,
    "sensit": load_sensit,
    "letters": load_letters,
    "szilard_1m": load_szilard_1m,
    "szilard_10m": load_szilard_10m,
    # regression
    "abalone": load_abalone,
    "ames_housing": load_ames_housing,
    "california_housing": load_california_housing,
    "fried": load_fried,
    "medical_charges_nominal": load_medical_charges_nominal,
    "twodplanes": load_twodplanes,
    "year_prediction_msd": load_year_prediction_msd,
    "yolanda": load_yolanda,
    "road_network": load_road_network,
    # index search
    "sift": load_sift,
    "gist": load_gist,
}
