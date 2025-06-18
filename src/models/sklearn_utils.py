import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_score


def perform_grid_search(
    models_and_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    cv: GroupKFold,
) -> dict:
    """Perform grid search for multiple models.

    Args:
        models_and_params (dict): Dictionary of models and their parameter grids
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        groups (pd.Series): Group labels for cross-validation
        cv (GroupKFold): Cross-validation strategy

    Returns:
        dict: Dictionary of grid search results for each model
    """
    results = {}

    for model_name, (model, param_grid) in models_and_params.items():
        print(f"Running GridSearch for {model_name}...")
        grid = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, scoring="roc_auc"
        )
        grid.fit(X_train, y_train, groups=groups)

        results[model_name] = {
            "best_params": grid.best_params_,
            "mean_score": grid.cv_results_["mean_test_score"][grid.best_index_],
            "std_score": grid.cv_results_["std_test_score"][grid.best_index_],
            "grid": grid,
        }

    return results


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    groups: pd.Series,
    cv: GroupKFold,
) -> dict:
    """
    Evaluate a model using cross-validation and on test data.

    Args:
        model: The machine learning model to evaluate
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
        groups (pd.Series): Group labels for cross-validation
        cv (GroupKFold): Cross-validation strategy

    Returns:
        dict: Dictionary containing model evaluation metrics
        Metrics include AUC, accuracy, sensitivity, and specificity.
        FPR and TPR for ROC curve are also included.
    """
    # Perform cross-validation
    cv_auc_scores = cross_val_score(
        model, X_train, y_train, cv=cv, groups=groups, scoring="roc_auc", n_jobs=-1
    )

    # Compute mean and SD of AUC
    mean_auc = np.mean(cv_auc_scores)
    std_auc = np.std(cv_auc_scores)

    # Fit the model to the entire training set
    model.fit(X_train, y_train)

    # Predict probabilities for the test set
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC on the test set
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    test_auc = auc(fpr, tpr)

    # Calculate other metrics
    y_test_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    accuracy = accuracy_score(y_test, y_test_pred)
    sensitivity = recall_score(y_test, y_test_pred)
    specificity = tn / (tn + fp)

    return {
        "model": model,
        "fpr": fpr,
        "tpr": tpr,
        "test_auc": test_auc,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
