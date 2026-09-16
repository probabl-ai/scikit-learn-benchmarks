REAL_DATASET_CASES = [
    # Trees:
    (
        ("kddcup09_churn", None),
        "RandomForestClassifier",
        {   # search space:
            "encoder": {
                "min_frequency": [5, 20, 100],
            },
            "estimator": {
                "n_jobs": -1,
                "n_estimators": [100, 200, 300],
                "max_features": [0.3, "sqrt"],
                "min_samples_split": [2, 5, 20, 100],
                "min_impurity_decrease": [3e-5, 1e-5, 1e-6]
            }
        }
    ),
    (
        ("susy", 300_000),
        "RandomForestClassifier",
        {   # search space:
            "estimator": {
                "n_jobs": -1,
                "n_estimators": 50,
                "min_samples_split": [50, 500, 5000],
                "min_impurity_decrease": [3e-4, 1e-4, 1e-5, 1e-6]
            }
        }
    ),
    (
        ("ames_housing", None),
        ["RandomForestRegressor", "ExtraTreesRegressor"],
        {   # search space:
            "encoder": {
                "min_frequency": [1, 5, 20, 100],
            },
            "estimator": {
                "n_jobs": -1,
                "n_estimators": [200, 400, 600],
                "max_features": [1., 0.5, 0.2],
                "max_leaf_nodes": [50, 200, 800],
            }
        }
    ),
    # Linear:
    # TODO:
    # - Ridge + ames_housing (same preprocessing than in real_dataset.py, skip for sklearnex because too slow)
    # - Ridge + susy (same preprocessing than in real_dataset.py too)
    # - Ridge + year_prediction_msd (same preprocessing than in real_dataset.py)
    # - Similar for logistic regression (only lbfgs): 2/3 datasets, at least one fast, at least one at 1-5s/fit
    # - 3 for kmeans too: road_network_points and sift, varying n_clusters (5 to 20).
    #   + one with more clusters (max ~5s/fit, maybe ny_times downsampled)
    # - For HGB: ames_housing, kddcup09_churn, year_prediction_msd
]


