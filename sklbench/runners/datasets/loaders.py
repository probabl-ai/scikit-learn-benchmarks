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
import re

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
Datasets used by configs/*.py (kept on top; see dataset_loading_functions for
the reference of which config file/tier uses each one).
"""


def load_ames_housing(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Ames Housing dataset (openml id 42165).

    1460 samples, 79 features, a mix of numeric and (mostly `object`-dtype)
    categorical columns, with real missing values in both. Small enough to be
    used in the "test" tier of configs.

    Regression task: predict `SalePrice`.
    """
    x, y = load_openml(42165, raw_data_cache, as_frame=True)
    x.drop(columns=['Id'], inplace=True)

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_amazon_employee_access(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Amazon Employee Access Challenge dataset (OpenML id 4135, originally a
    Kaggle competition).

    32769 samples, 9 features, all of them categorical (pandas `category`
    dtype) describing employee/resource/role attributes (e.g. RESOURCE,
    MGR_ID, ROLE_ROLLUP_1/2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC,
    ROLE_FAMILY, ROLE_CODE). Cardinality ranges widely across columns, from
    a few dozen unique values up to several thousand (RESOURCE has 7518,
    MGR_ID has 4243), so this dataset exercises categorical preprocessing
    with a mix of low- and very high-cardinality columns. No missing values.

    Classification task: predict ACTION (whether access was granted).
    n_classes = 2, notably imbalanced (~94% granted / ~6% denied).
    """
    x, y = load_openml(4135, raw_data_cache, as_frame=True)

    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_kddcup09_churn(raw_data_cache: str) -> tuple[dict, dict]:
    """
    KDD Cup 2009 "churn" task (openml id 1112).
    https://www.openml.org/d/1112

    Raw shape is 50000 rows x 230 columns (174 float64, 18 `object`, and
    38 `category` columns -- 56 categorical-like columns total). 23 of the
    230 columns are entirely useless (all-NaN or constant, i.e.
    `nunique(dropna=True) <= 1`) and are dropped here, leaving 207 columns.

    Missingness remains heavy even after dropping the useless columns: 161
    of the original 230 columns have >50% NaN. This is real, expected
    messiness for this dataset and is left untouched -- downstream
    preprocessing/models must handle NaN (HGB-style histogram trees do so
    natively; plain sklearn trees currently don't, which is a known
    limitation of this repo's tree preprocessing, not something to fix
    here).

    Categorical cardinality is bimodal: most of the 56 categorical-like
    columns are low-cardinality (many between 2 and 100 uniques), but a
    handful are very high-cardinality, e.g. Var214/Var200 (15415 uniques
    each), Var217 (13990), Var202 (5713), Var199 (5073), and
    Var198/Var220/Var222 (4291 each). This dataset is included specifically
    to exercise categorical preprocessing across both extremes.

    Classification task. n_classes = 2. Target is highly imbalanced:
    y=0 (no churn) = 46328 (92.7%), y=1 (churn) = 3672 (7.3%), so the
    default split stratifies on y to keep churn examples in both subsets.
    """
    x, y = load_openml(1112, raw_data_cache, as_frame=True)

    useless_cols = [col for col in x.columns if x[col].nunique(dropna=True) <= 1]
    x = x.drop(columns=useless_cols)

    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42, "stratify": "y"},
    }
    return {"x": x, "y": y}, data_desc


def load_kick(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Don't Get Kicked (openml id 41162).

    72983 samples, 32 columns describing used vehicles purchased at auction,
    with a realistic mix of 9 numeric columns (price/odometer/age fields like
    VehOdo, VehicleAge, several MMR* price columns, VehBCost, WarrantyCost)
    and `category`-dtype columns spanning low to high cardinality (from 2
    up to 1063 unique values for Model, 863 for SubModel, 134 for Trim).
    PRIMEUNIT and AUCGUART are ~95% missing but left as-is: the missingness
    itself is an informative low-frequency category rather than noise.

    Classification task: predict IsBadBuy (whether the purchase was a "kick").
    Imbalanced target, ~87.7% negative / 12.3% positive. n_classes = 2.
    """
    x, y = load_openml(41162, raw_data_cache, as_frame=True)

    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def undo_one_hot(x: pd.DataFrame) -> pd.DataFrame:
    """Collapse `<prefix>_<int>`-named one-hot dummy columns (as produced by
    sklearn's `as_frame=True` fetchers for originally-categorical UCI
    features) back into one `category`-dtype column per prefix."""
    x = x.copy()
    matches = [re.match(r"(.*)_[0-9]+$", col) for col in x.columns]
    col_mapping = {m.group(0): m.group(1) for m in matches if m}

    for col in set(col_mapping.values()):
        x[col] = -1
    for src_col, col in col_mapping.items():
        val = int(src_col.split("_")[-1])
        x.loc[x[src_col] > 0, col] = val
    x = x.drop(columns=list(col_mapping))
    for col in set(col_mapping.values()):
        x[col] = x[col].astype("category")
    return x


def load_covtype(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Cover type dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/covertype

    581012 samples, 12 features: 10 continuous numeric (elevation, slope,
    hillshade, distances to hydrology/roadways/fire points, ...) plus 2
    categorical features reconstructed (via `undo_one_hot`) from their
    one-hot-encoded source columns - `Wilderness_Area` (4 categories) and
    `Soil_Type` (40 categories). No missing values.

    y contains 7 unique class labels from 1 to 7 inclusive.
    Classification task. n_classes = 7.
    """
    x, y = fetch_covtype(return_X_y=True, as_frame=True, data_home=raw_data_cache)
    x = undo_one_hot(x)
    y = y.astype(int) - 1

    data_desc = {
        "n_classes": 7,
        "default_split": {"test_size": 0.2, "random_state": 77},
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
    """
    SUSY dataset from UCI machine learning repository
    https://archive.ics.uci.edu/ml/datasets/SUSY

    5,000,000 samples, 18 features, all numeric (kinematic properties of
    simulated particle collisions). No missing values. Binary
    classification, roughly balanced (~45.7% positive).
    """
    return load_higgs_susy_subsample("susy", raw_data_cache)


def load_year_prediction_msd(raw_data_cache: str) -> tuple[dict, dict]:
    """
    YearPredictionMSD dataset (subset of the Million Song Dataset) from UCI
    machine learning repository
    https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

    515345 samples, 90 features, all numeric (timbre-based audio features:
    12 average values + 78 covariance values). No missing values.

    Regression task: predict the song's release year, ranging 1922-2011.
    Inherently hard - published baselines report R2 around 0.2-0.3, so a
    low R2 here reflects the task, not an undertuned model.
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
        "YearPredictionMSD.txt.zip"
    )
    data = download_and_read_csv(url, raw_data_cache, header=None)
    x, y = data.iloc[:, 1:], data.iloc[:, 0]
    # `x`'s columns are otherwise the leftover integer labels [1..90] from
    # the headerless CSV read, which `ColumnTransformer` misreads as
    # 0-indexed positional indices (breaking `linear`/`trees`
    # preprocessing) - string-label them to remove the ambiguity.
    x.columns = [str(col) for col in x.columns]
    data_desc = {"default_split": {"test_size": 0.1, "shuffle": False}}
    return {"x": x, "y": y}, data_desc


def load_fraud(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Credit Card Fraud Detection dataset (openml id 42175).

    284807 samples, 30 features, all numeric: 28 PCA-derived components
    (V1-V28, anonymized for confidentiality) plus `Time` and `Amount`. No
    missing values.

    `Time` (seconds since the first transaction, spanning ~2 days) is
    replaced with `Time_of_day` (`Time` modulo 86400) since the raw
    elapsed-time counter isn't meaningful on its own but time-of-day is.
    For the `linear` preprocessing kind: V1-V28 are already well-conditioned
    PCA components and are passed through untouched (flagged via
    `preprocessing_defaults`), while `Time_of_day` and `Amount` get
    spline-encoded with 20 knots instead of the linear preprocessor's
    default 10.

    Rows are in chronological order (confirmed via the original `Time`
    column), so the default split is *not* shuffled: train on the first
    80% of transactions, test on the most recent 20% (417 vs 75 fraud
    cases) - this mirrors real fraud-detection deployment (train on past
    transactions, evaluate on future ones) rather than an optimistic
    random split.

    Classification task. n_classes = 2, extremely imbalanced: 492 fraud
    cases (0.17%) out of 284807 transactions.
    """
    x, y = load_openml(42175, raw_data_cache, as_frame=True)
    x = x.rename(columns={"Time": "Time_of_day"})
    x["Time_of_day"] = x["Time_of_day"] % 86400

    v_columns = [col for col in x.columns if col.startswith("V")]
    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "shuffle": False},
        "preprocessing_defaults": {
            "linear": {
                "passthrough_columns": v_columns,
                "spline_kwargs": {"n_knots": 20},
            },
        },
    }
    return {"x": x, "y": y}, data_desc


def load_medical_charges_nominal(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Medical charges dataset (openml id 42559): US hospital inpatient charges
    by diagnosis-related group (DRG) and provider.

    163065 samples, 11 features: one moderate-cardinality categorical
    (`DRG_Definition`, 100 categories) plus several very-high-cardinality
    "nominal" identifier-like categoricals for the billing provider
    (`Provider_Id`/`Provider_Name`/`Provider_Street_Address` each with
    ~3000-3300 categories, `Provider_Zip_Code` ~3053, `Provider_City` ~1977,
    `Hospital_Referral_Region_(HRR)_Description` ~306, `Provider_State` 51),
    plus 2 numeric features and 1 numeric count. No missing values. This is
    a standard benchmark for high-cardinality categorical encoding.

    Regression task: predict average total payment per DRG/provider.
    """
    x, y = load_openml(42559, raw_data_cache, as_frame=True)

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_bank_marketing(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Bank Marketing dataset (openml id 1461): UCI phone-based term-deposit
    marketing campaign data from a Portuguese bank.

    45211 samples, 16 features (columns are anonymized as V1-V16 by this
    OpenML version but correspond to the standard UCI schema: age, job,
    marital, education, default, balance, housing, loan, contact, day,
    month, duration, campaign, pdays, previous, poutcome). 9 categorical
    columns with low cardinality (2 to 12 categories), the rest numeric. No
    missing values.

    Classification task: predict whether the client subscribed to a term
    deposit. n_classes = 2, imbalanced (~88.3% no / ~11.7% yes).
    """
    x, y = load_openml(1461, raw_data_cache, as_frame=True)

    data_desc = {
        "n_classes": 2,
        "default_split": {"test_size": 0.2, "random_state": 42},
    }
    return {"x": x, "y": y}, data_desc


def load_california_housing(raw_data_cache: str) -> tuple[dict, dict]:
    """
    California Housing dataset (StatLib, from the 1990 US census).

    20640 samples, 8 features, all numeric (median income, house age, mean
    rooms/bedrooms per household, population, mean household occupancy,
    latitude/longitude), no missing values.

    Regression task: predict the median house value (in units of $100,000)
    for the census block group.
    """
    x, y = fetch_california_housing(
        return_X_y=True, as_frame=True, data_home=raw_data_cache
    )
    data_desc = {"default_split": {"test_size": 0.1, "random_state": 42}}
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


def load_fried(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(564, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_twodplanes(raw_data_cache: str) -> tuple[dict, dict]:
    x, y = load_openml(1197, raw_data_cache)
    data_desc = {"default_split": {"test_size": 0.4, "random_state": 42}}
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
Clustering datasets
"""


def load_road_network_points(raw_data_cache: str) -> tuple[dict, dict]:
    """
    3D Road Network (North Jutland, Denmark) dataset from UCI machine
    learning repository, same source file as `load_road_network` above:
    https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+%28North+Jutland%2C+Denmark%29

    434874 samples, 3 features, all numeric and real (not synthetic):
    longitude, latitude and altitude of points sampled along roads, derived
    from a real GPS/elevation survey. No missing values.

    Unlike `load_road_network` (which pairs the coordinates with the row's
    OSM way id as a regression target), this loader keeps only the 3D
    position itself, with no target - meant for clustering algorithms like
    KMeans on genuine low-dimensional spatial data rather than a synthetic
    `make_blobs` or high-dimensional dataset like MNIST.
    """
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
    data = download_and_read_csv(
        url,
        raw_data_cache,
        header=None,
        names=["osm_id", "longitude", "latitude", "altitude"],
    )
    x = data[["longitude", "latitude", "altitude"]].to_numpy(dtype=np.float64)
    y = np.zeros(x.shape[0])

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
    return {"x": x, "y": y}, data_desc


def load_airports(raw_data_cache: str) -> tuple[dict, dict]:
    """
    Airports dataset from OurAirports (public domain,
    https://ourairports.com/data/), a community-maintained worldwide
    directory of airports, heliports and seaplane bases.

    60466 samples (after filtering, see below), 3 features, all numeric and
    real: longitude, latitude and elevation (in feet) of each facility.
    `closed` facilities and rows missing `elevation_ft` are dropped (from
    85819 raw rows) - elevation is unset for a meaningful fraction of small,
    unsurveyed airstrips/heliports, and closed facilities no longer
    represent a real, currently-operating location. No target - meant for
    clustering algorithms like KMeans on genuine low-dimensional spatial
    data (e.g. grouping airports by continent/region) rather than a
    synthetic `make_blobs` or high-dimensional dataset like MNIST.
    """
    url = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    data = download_and_read_csv(url, raw_data_cache)
    data = data[data["type"] != "closed"]
    data = data.dropna(subset=["elevation_ft"])
    x = data[["longitude_deg", "latitude_deg", "elevation_ft"]].to_numpy(dtype=np.float64)
    y = np.zeros(x.shape[0])

    data_desc = {"default_split": {"test_size": 0.2, "random_state": 42}}
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
    y = np.zeros((x.shape[0],))
    return {"x": x, "y": y}, data_desc


def load_sift(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    return load_ann_dataset_template(url, raw_data_cache)


def load_gist(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://ann-benchmarks.com/gist-960-euclidean.hdf5"
    return load_ann_dataset_template(url, raw_data_cache)


def load_glove_100_l2_normalized(raw_data_cache: str) -> tuple[dict, dict]:
    """
    GloVe 100-dim word embeddings from ann-benchmarks.com
    (glove-100-angular.hdf5), L2-normalized.

    The source dataset is meant for cosine/angular-distance ANN search, so
    vectors aren't normalized as stored. L2-normalizing here makes
    Euclidean distance (and thus k-means) equivalent to cosine distance.
    """
    url = "http://ann-benchmarks.com/glove-100-angular.hdf5"
    data, data_desc = load_ann_dataset_template(url, raw_data_cache)
    x = data["x"]
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return {"x": x, "y": data["y"]}, data_desc


def load_fashion_mnist_784_euclidean(raw_data_cache: str) -> tuple[dict, dict]:
    url = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
    return load_ann_dataset_template(url, raw_data_cache)


def load_nytimes_256_l2_normalized(raw_data_cache: str) -> tuple[dict, dict]:
    """
    NYTimes 256-dim bag-of-words embeddings from ann-benchmarks.com
    (nytimes-256-angular.hdf5), L2-normalized.

    The source dataset is meant for cosine/angular-distance ANN search, so
    vectors aren't normalized as stored. L2-normalizing here makes
    Euclidean distance (and thus k-means) equivalent to cosine distance.

    248/300000 rows are exact all-zero vectors (documents with no overlap
    with the source vocabulary) - dividing those by their own (zero) norm
    would produce NaN, so they're left as zero vectors instead.
    """
    url = "http://ann-benchmarks.com/nytimes-256-angular.hdf5"
    data, data_desc = load_ann_dataset_template(url, raw_data_cache)
    x = data["x"]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.where(norms == 0, 1, norms)
    return {"x": x, "y": data["y"]}, data_desc


dataset_loading_functions = {
    # used by configs/*.py
    "ames_housing": load_ames_housing,
    "amazon_employee_access": load_amazon_employee_access,
    "kddcup09_churn": load_kddcup09_churn,
    "kick": load_kick,
    "covtype": load_covtype,
    "susy": load_susy,
    "year_prediction_msd": load_year_prediction_msd,
    "fraud": load_fraud,
    "medical_charges_nominal": load_medical_charges_nominal,
    "bank_marketing": load_bank_marketing,
    "california_housing": load_california_housing,
    "mnist": load_mnist,
    # classification
    "airline_depdelay": load_airline_depdelay,
    "a9a": load_a9a,
    "codrnanorm": load_codrnanorm,
    "creditcard": load_creditcard,
    "digits": load_sklearn_digits,
    "gisette": load_gisette,
    "hepmass": load_hepmass,
    "higgs": load_higgs,
    "ijcnn": load_ijcnn,
    "klaverjas": load_klaverjas,
    "cifar": load_cifar,
    "connect": load_connect,
    "covertype": load_covertype,
    "skin_segmentation": load_skin_segmentation,
    "fashion_mnist": load_fashion_mnist,
    "svhn": load_svhn,
    "sensit": load_sensit,
    "letters": load_letters,
    "szilard_1m": load_szilard_1m,
    "szilard_10m": load_szilard_10m,
    # regression
    "abalone": load_abalone,
    "fried": load_fried,
    "twodplanes": load_twodplanes,
    "yolanda": load_yolanda,
    "road_network": load_road_network,
    # clustering
    "road_network_points": load_road_network_points,
    "airports": load_airports,
    # index search
    "sift": load_sift,
    "gist": load_gist,
    "glove_100": load_glove_100_l2_normalized,
    "fashion_mnist_784": load_fashion_mnist_784_euclidean,
    "nytimes_256": load_nytimes_256_l2_normalized,
}
