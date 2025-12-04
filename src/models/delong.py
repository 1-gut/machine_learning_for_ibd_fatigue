import numpy as np
from scipy import stats


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Computes midranks for a 1D array of scores.

    This helper is part of the fast implementation of DeLong's test for
    comparing correlated ROC curves.
    """
    # Sort scores in ascending order
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)

    # Initialize midranks
    midranks = np.zeros(n, dtype=float)
    i = 0

    while i < n:
        j = i
        # Find the extent of ties
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1

        # Assign midrank for the tied group
        midranks[i:j] = 0.5 * (i + j - 1)
        i = j

    # Revert to original order
    reverse = np.empty(n, dtype=int)
    reverse[order] = np.arange(n)
    return midranks[reverse] + 1  # 1-based midranks for consistency with literature


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """
    Fast implementation of DeLong's covariance calculation.

    Adapted from Sun and Xu (2014) and commonly used open-source references.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    # Calculate midranks for positive and negative groups separately
    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_predictions[r, :])
        ty[r, :] = _compute_midrank(negative_predictions[r, :])

    tz = np.hstack((tx, ty))
    # Midranks across all observations
    for r in range(k):
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    # AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = (tz[:, m:] - ty) / m

    s01 = np.cov(v01)
    s10 = np.cov(v10)
    delong_cov = s01 / m + s10 / n

    return aucs, delong_cov


def _prepare_predictions(y_true: np.ndarray, predictions: np.ndarray):
    """Orders predictions so that positives come first as required by DeLong."""
    # Sort so that positive labels (1) come first
    order = np.argsort(-y_true)
    return predictions[:, order]


def delong_roc_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
):
    """
    Performs DeLong's test to compare the AUCs of two correlated ROC curves.

    Args:
        y_true: Ground truth binary labels (1 for positive, 0 for negative)
        y_pred_a: Predicted scores for model A
        y_pred_b: Predicted scores for model B

    Returns:
        dict with aucs, z_score, and two-sided p_value.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array of binary labels")

    label_1_count = int(np.sum(y_true))
    if label_1_count == 0 or label_1_count == len(y_true):
        raise ValueError("y_true must contain both positive and negative samples")

    preds = np.vstack((y_pred_a, y_pred_b))
    preds_sorted = _prepare_predictions(y_true, preds)

    aucs, delong_cov = _fast_delong(preds_sorted, label_1_count)
    auc_diff = aucs[0] - aucs[1]
    var_diff = delong_cov[0, 0] + delong_cov[1, 1] - 2 * delong_cov[0, 1]

    # Guard against degenerate variance
    if var_diff <= 0:
        return {"aucs": aucs, "z_score": np.nan, "p_value": np.nan}

    z_score = auc_diff / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z_score))

    return {"aucs": aucs, "z_score": z_score, "p_value": p_value}
