import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def get_subset_metrics_of_estimator(
    task, stage, estimator_instance, data
) -> dict[str, float]:
    # brute kNN with transfer between training and inference stages
    # is required for recall metric calculation of search task
    global _brute_knn

    metrics = dict()
    # Note: use `x` and `y` when calling estimator methods,
    # and `x_compat` and `y_compat` for compatibility with sklearn metrics
    x, y = data
    x_compat = convert_to_numpy(x)
    y_compat = convert_to_numpy(y)
    if task == "classification":
        y_pred = convert_to_numpy(estimator_instance.predict(x))
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_compat, y_pred)),
                "balanced accuracy": float(balanced_accuracy_score(y_compat, y_pred)),
            }
        )
        if hasattr(estimator_instance, "predict_proba") and not (
            hasattr(estimator_instance, "probability")
            and getattr(estimator_instance, "probability") == False
        ):
            y_pred_proba = convert_to_numpy(estimator_instance.predict_proba(x))
            metrics.update(
                {
                    "ROC AUC": float(
                        roc_auc_score(
                            y_compat,
                            (
                                y_pred_proba
                                if y_pred_proba.shape[1] > 2
                                else y_pred_proba[:, 1]
                            ),
                            multi_class="ovr",
                        )
                    ),
                    "logloss": float(log_loss(y_compat, y_pred_proba)),
                }
            )
    elif task == "regression":
        y_pred = convert_to_numpy(estimator_instance.predict(x))
        metrics.update(
            {
                "RMSE": float(mean_squared_error(y_compat, y_pred) ** 0.5),
                "R2": float(r2_score(y_compat, y_pred)),
            }
        )
    elif task == "decomposition":
        if "PCA" in str(estimator_instance):
            if hasattr(estimator_instance, "score"):
                metrics.update(
                    {"average log-likelihood": float(estimator_instance.score(x))}
                )
            if stage == "training" and hasattr(
                estimator_instance, "explained_variance_ratio_"
            ):
                metrics.update(
                    {
                        "1st component variance ratio": float(
                            estimator_instance.explained_variance_ratio_[0]
                        )
                    }
                )
    elif task == "clustering":
        if hasattr(estimator_instance, "inertia_"):
            # compute inertia manually using distances to cluster centers
            # provided by KMeans.transform
            metrics.update(
                {
                    "inertia": float(
                        np.power(
                            convert_to_numpy(estimator_instance.transform(x)).min(axis=1),
                            2,
                        ).sum()
                    )
                }
            )
        if hasattr(estimator_instance, "predict"):
            y_pred = convert_to_numpy(estimator_instance.predict(x))
            metrics.update(
                {
                    "Davies-Bouldin score": float(davies_bouldin_score(x_compat, y_pred)),
                    "homogeneity": float(homogeneity_score(y_compat, y_pred)),
                    "completeness": float(completeness_score(y_compat, y_pred)),
                }
            )
        if "DBSCAN" in str(estimator_instance) and stage == "training":
            labels = convert_to_numpy(estimator_instance.labels_)
            clusters = len(np.unique(labels[labels != -1]))
            metrics.update({"clusters": clusters})
            if clusters > 1:
                metrics.update(
                    {
                        "Davies-Bouldin score": float(
                            davies_bouldin_score(x_compat, labels)
                        )
                    }
                )
            if len(np.unique(y_compat)) < 128:
                metrics.update(
                    {
                        "homogeneity": (
                            float(homogeneity_score(y_compat, labels))
                            if clusters > 1
                            else 0
                        ),
                        "completeness": (
                            float(completeness_score(y_compat, labels))
                            if clusters > 1
                            else 0
                        ),
                    }
                )
    elif task == "manifold":
        if hasattr(estimator_instance, "kl_divergence_") and stage == "training":
            metrics.update(
                {"Kullback-Leibler divergence": float(estimator_instance.kl_divergence_)}
            )
    elif task == "search":
        if stage == "training":
            from sklearn.neighbors import NearestNeighbors

            _brute_knn = NearestNeighbors(algorithm="brute").fit(x_compat)
        else:
            recall_degree = 10
            ground_truth_neighbors = _brute_knn.kneighbors(
                x_compat, recall_degree, return_distance=False
            )
            predicted_neighbors = convert_to_numpy(
                estimator_instance.kneighbors(x, recall_degree, return_distance=False)
            )
            n_relevant = 0
            for i in range(ground_truth_neighbors.shape[0]):
                n_relevant += len(
                    np.intersect1d(ground_truth_neighbors[i], predicted_neighbors[i])
                )
            recall = (
                n_relevant
                / ground_truth_neighbors.shape[0]
                / ground_truth_neighbors.shape[1]
            )
            metrics.update({f"recall@{recall_degree}": recall})
    if (
        hasattr(estimator_instance, "support_vectors_")
        and estimator_instance.support_vectors_ is not None
    ):
        metrics.update({"support vectors": len(estimator_instance.support_vectors_)})
    return metrics


def convert_to_numpy(a, dp_compat=False) -> np.ndarray:
    if dp_compat and ("dpctl" in str(type(a)) or "dpnp" in str(type(a))):
        return a
    if isinstance(a, np.ndarray):
        return a
    elif hasattr(a, "to_numpy"):
        return a.to_numpy()
    elif hasattr(a, "asnumpy"):
        return a.asnumpy()
    elif "dpnp" in str(type(a)):
        import dpnp

        return dpnp.asnumpy(a)
    elif "torch.Tensor" in str(type(a)):
        return a.detach().cpu().numpy()
    elif "cupy.ndarray" in str(type(a)):
        return a.get()
    else:
        raise ValueError("Unable to convert data to numpy.ndarray")
