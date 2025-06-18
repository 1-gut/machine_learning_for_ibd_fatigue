from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.config.constants import NUMERICAL_FEATURES
from src.config.paths import DNN_DIR
from src.data.conversions import (
    capitalize_first_letter,
    convert_creatinine_from_mg_dl_to_umol_l,
    convert_urea_from_mg_dl_to_mmol_l,
)
from src.data.preprocessing import (
    fill_missing_values_with_saved_medians,
    map_season,
    map_season_australia,
)

plt.rcParams["font.family"] = ["Roboto", "Arial", "sans-serif"]


def preprocess_validation_data(
    df: pd.DataFrame,
    target_df: pd.DataFrame,
    scaler,
    country: str,
    medians_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess validation data from different countries to match the format expected by the model.

    Parameters:
    -----------
    df : pandas DataFrame
        Raw validation data
    target_df : pandas DataFrame
        Reference dataframe with expected columns and types
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to normalize numerical features
    country : str
        Country of validation data ('spain', 'scotland', 'australia')
    medians_path : Path, optional
        Path to medians.json file for filling missing values

    Returns:
    --------
    tuple
        (X_validation, y_validation) - preprocessed features and target
    """
    # Make a copy to avoid modifying the original
    df = df.copy()

    # Define fatigue outcome based on days of fatigue (≥10 days = fatigue)
    df["fatigue_outcome"] = df[
        "On how many days over the last 14 days have you felt fatigued? (0-14)"
    ].apply(lambda x: 1 if x >= 10 else 0)

    # Common column renaming
    cols_to_rename = {
        "height_cm": "height",
        "weight_kg": "weight",
        "has_symptoms_of_active_ibd": "has_active_symptoms",
        "sampling_aminosalicylates (any)": "sampling_asa",
        "sampling_antibiotics": "sampling_abx",
        "study_group": "study_group_name",
        "smoking_status": "is_smoker",
    }
    df.rename(columns=cols_to_rename, inplace=True)

    # Strip whitespace from categorical columns
    cols_to_strip_whitespace = [
        "sex",
        "montreal_cd_location",
        "montreal_cd_behaviour",
        "montreal_uc_severity",
        "montreal_uc_extent",
    ]
    for col in cols_to_strip_whitespace:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Sex transformation
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    # BMI calculation
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

    # Datetime Type Conversion
    datetime_cols = [
        "date_of_diagnosis",
        "date_recorded",
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Age at diagnosis calculation
    df["age_at_diagnosis"] = df["age"] - (2025 - df["date_of_diagnosis"].dt.year)

    # Disease duration and year features
    df["disease_duration_weeks"] = (
        df["date_recorded"] - df["date_of_diagnosis"]
    ).dt.days / 7
    df["diagnosis_year"] = df["date_of_diagnosis"].dt.year

    # Season mapping (different for Australia due to Southern Hemisphere)
    if country.lower() == "australia":
        df["season"] = df["date_recorded"].apply(map_season_australia)
    else:
        df["season"] = df["date_recorded"].apply(map_season)

    # Study group name standardization
    df["study_group_name"] = (
        df["study_group_name"].astype(str).apply(lambda x: x.upper())
    )

    # Capitalize the first letter in smoking_status
    df["is_smoker"] = df["is_smoker"].apply(capitalize_first_letter)

    # Country-specific processing
    if country.lower() == "spain":
        # Fix Spanish creatinine values (different units)
        df["creatinine"] = df["creatinine"].apply(lambda x: x / 100 if x > 50 else x)
        df["urea"] = convert_urea_from_mg_dl_to_mmol_l(df["urea"])
        df["creatinine"] = convert_creatinine_from_mg_dl_to_umol_l(df["creatinine"])

    elif country.lower() == "australia":
        df["montreal_cd_behaviour"] = df["montreal_cd_behaviour"].replace(
            "B1 Non-stricturing", "B1 Non-stricturing, non-penetrating"
        )
    elif country.lower() == "scotland":
        # Remove (vape) from suffix
        df["is_smoker"] = df["is_smoker"].str.replace(r"\s*\(vape\)", "", regex=True)
        df["is_smoker"].value_counts()

        # Map the smoking values
        df["is_smoker"] = df["is_smoker"].map(
            {
                "Non smoker": "Non-smoker",
                "Ex smoker": "Ex-smoker",
                "Current smoker": "Smoker",
                "Smoker": "Smoker",
            }
        )
        # Fill missing sex with 1
        df["sex"] = df["sex"].fillna(1)
        df["montreal_uc_extent"] = df["montreal_uc_extent"].replace(
            {
                "E2": "E2 Left-sided",
            }
        )
        # Add this columns as no patients had these features
        # WARNING: This can cause bugs in the pipeline if a new patient has S3
        cols_to_add = ["montreal_uc_severity_S3 Severe"]
        for col in cols_to_add:
            if col not in df.columns:
                df[col] = 0
                df[col] = df[col].astype(int)

    # Handle calprotectin (remove < symbol if present)
    if "calprotectin" in df.columns:
        if df["calprotectin"].dtype == object:  # Check if not numeric already
            df["calprotectin"] = df["calprotectin"].astype(str).str.replace("<", "")
            df["calprotectin"] = df["calprotectin"].astype(float)

    # Binary columns (Yes/No to 1/0)
    cols_to_map = [
        "montreal_upper_gi",
        "montreal_perianal",
        "sampling_steroids",
        "sampling_abx",
        "sampling_asa",
        "sampling_aza",
        "sampling_mp",
        "sampling_ifx",
        "sampling_ada",
        "sampling_vedo",
        "sampling_uste",
        "sampling_tofa",
        "sampling_mtx",
        "sampling_ciclosporin",
        "sampling_filgo",
        "sampling_upa",
        "sampling_risa",
        "has_active_symptoms",
    ]
    for col in cols_to_map:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encoding of categorical variables
    categorical_features_for_one_hot_encoding = [
        "study_group_name",
        "montreal_cd_location",
        "montreal_cd_behaviour",
        "montreal_uc_extent",
        "montreal_uc_severity",
        "is_smoker",
        "season",
    ]

    # Fix NaN values in categorical columns
    cols_to_fix_nans = [
        "montreal_cd_location",
        "montreal_cd_behaviour",
        "montreal_uc_severity",
        "montreal_uc_extent",
        "study_group_name",
    ]
    for col in cols_to_fix_nans:
        if col in df.columns:
            df[col] = df[col].replace(["nan", "NaN", "NAN"], np.nan)

    # One-hot encoding
    df = pd.get_dummies(
        df, columns=categorical_features_for_one_hot_encoding, dtype=int
    )

    # Add missing season columns if not present
    for season in ["autumn", "summer", "winter", "spring"]:
        if f"season_{season}" not in df.columns:
            df[f"season_{season}"] = 0
            df[f"season_{season}"] = df[f"season_{season}"].astype(int)

    # Fill missing values for binary columns
    nans_to_fill_with_zero = [
        "montreal_upper_gi",
        "montreal_perianal",
        "sampling_steroids",
        "sampling_abx",
        "sampling_asa",
        "sampling_aza",
        "sampling_mp",
        "sampling_ifx",
        "sampling_ada",
        "sampling_vedo",
        "sampling_uste",
        "sampling_tofa",
        "sampling_mtx",
        "sampling_ciclosporin",
        "sampling_filgo",
        "sampling_upa",
        "sampling_risa",
        "has_active_symptoms",
    ]
    for col in nans_to_fill_with_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill missing values for numerical features
    if medians_path is None:
        medians_path = DNN_DIR / "exports" / "medians.json"

    df = fill_missing_values_with_saved_medians(df, medians_path=medians_path)

    # Columns to drop
    cols_to_drop = [
        "Who collected this data",
        "Clinical Setting (inpatient, outpatient, endoscopy, infusion unit, etc.)",
        "Unnamed: 49",
        "On how many days over the last 14 days have you felt fatigued? (0-14)",
        "date_of_diagnosis",
        "date_recorded",
    ]
    # Drop columns if they exist
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    # Create X and y
    y_validation = df["fatigue_outcome"].copy()
    X_validation = df.drop(columns=["fatigue_outcome"]).copy()

    # Ensure columns match target_df
    X_validation = X_validation[target_df.columns]

    # Ensure datatypes match
    for col in X_validation.columns:
        X_validation[col] = X_validation[col].astype(target_df[col].dtype)

    # Scale numerical features
    numerical_features = [
        col
        for col in NUMERICAL_FEATURES
        if col not in ["haematocrit", "ada_drug_level", "ifx_drug_level"]
    ]
    X_validation[numerical_features] = scaler.transform(
        X_validation[numerical_features]
    )

    return X_validation, y_validation


def evaluate_model(
    X_validation: pd.DataFrame, y_validation: pd.Series, model
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Evaluate a model on validation data.

    Parameters:
    -----------
    X_validation : pandas DataFrame
        Validation features
    y_validation : pandas Series
        Validation target
    model : keras.Model or sklearn model
        The model to evaluate

    Returns:
    --------
    tuple
        (y_pred_proba, y_pred, roc_auc, fpr, tpr, confusion_matrix, metrics_dict)
    """
    # Get probability predictions
    if hasattr(model, "predict_proba"):
        # For sklearn models
        y_pred_proba = model.predict_proba(X_validation)[:, 1]

    else:
        # for Keras models
        y_pred_proba = model.predict(X_validation)

    # Get binary predictions using 0.5 threshold
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate ROC curve and AUC

    fpr, tpr, _ = roc_curve(y_validation, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate confusion matrix
    cm = confusion_matrix(y_validation, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute metrics
    metrics = {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
        "auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    return y_pred_proba, y_pred, roc_auc, fpr, tpr, cm, metrics


def prepare_logreg_data(X_validation: pd.DataFrame, logreg_model) -> pd.DataFrame:
    """
    Prepare data for statsmodels logistic regression prediction.

    Parameters:
    -----------
    X_validation : pandas DataFrame
        Validation features
    logreg_model : statsmodels.discrete.discrete_model.Logit
        Trained logistic regression model

    Returns:
    --------
    pandas DataFrame
        Data prepared for statsmodels prediction
    """
    # Add constant
    X_const = sm.add_constant(X_validation, has_constant="add")

    # Ensure columns match the model's parameters
    missing_cols = set(logreg_model.params.index) - set(X_const.columns)
    for c in missing_cols:
        print(f"WARNING: missing column detected: {c}")

    # Order columns according to model parameters
    X_const = X_const[logreg_model.params.index]

    return X_const


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
):
    """
    Plot ROC curve for model evaluation.

    Parameters:
    -----------
    fpr : numpy array
        False positive rates
    tpr : numpy array
        True positive rates
    roc_auc : float
        Area under the ROC curve
    title : str, optional
        Plot title
    save_path : Path, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"DNN (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray, title: str = "Confusion Matrix", save_path: Optional[Path] = None
):
    """
    Plot confusion matrix with annotations.

    Parameters:
    -----------
    cm : numpy array
        Confusion matrix (2x2)
    title : str, optional
        Plot title
    save_path : Path, optional
        Path to save the plot
    """

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fatigue (0)", "Fatigue (1)"],
        yticklabels=["No Fatigue (0)", "Fatigue (1)"],
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_metrics_dict(model_name, metrics_dict):
    """
    Create a dictionary of metrics for the model.
    Parameters
    ----------
    model_name : str
        Name of the model
    metrics_dict : dict
        Dictionary containing the metrics
    Returns
    -------
    dict
        Formatted dictionary containing the model name and metrics
    """
    return {
        "Model": model_name,
        "AUC": metrics_dict["auc"],
        "Accuracy": metrics_dict["accuracy"],
        "Sensitivity": metrics_dict["sensitivity"],
        "Specificity": metrics_dict["specificity"],
        "Positive Predictive Value": metrics_dict["ppv"],
        "Negative Predictive Value": metrics_dict["npv"],
        "True Positives": metrics_dict["tp"],
        "True Negatives": metrics_dict["tn"],
        "False Positives": metrics_dict["fp"],
        "False Negatives": metrics_dict["fn"],
    }


def plot_shap_force_plot(
    df: pd.DataFrame, scaler, explainer, max_plots: Optional[int] = None
):
    """
    Plot SHAP force plots for individual predictions.
    Args:
        df (pd.DataFrame): DataFrame containing the features for which to plot SHAP values.
        scaler (sklearn.preprocessing.StandardScaler): Scaler used to inverse transform the features.
        explainer (shap.Explainer): SHAP explainer object.
    """
    # Ensure the DataFrame is in the correct format for SHAP
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    import shap

    shap_values = explainer.shap_values(df)
    shap_values_class_1 = shap_values[:, :, 0]

    # Inverse transform the numerical features
    numerical_features = [
        col
        for col in NUMERICAL_FEATURES
        if col not in ["haematocrit", "ada_drug_level", "ifx_drug_level"]
    ]

    df[numerical_features] = scaler.inverse_transform(df[numerical_features])

    # If max_plots if provided, limit the number of plots
    number_of_plots = len(df) if max_plots is None else max_plots
    if number_of_plots > len(df):
        number_of_plots = len(df)

    # Plot Individual Predictions
    for i in range(number_of_plots):
        shap.force_plot(
            explainer.expected_value[0],
            shap_values_class_1[i],
            df.iloc[i],
            matplotlib=True,
            contribution_threshold=0.05,
            text_rotation=30,
            show=False,
            plot_cmap="PRGn",
        )
        plt.show()


def plot_combined_roc_curves(
    all_model_results, output_path, title="ROC Curves", figsize=(8, 6)
):
    """
    Plot combined ROC curves for multiple models.

    Parameters:
    -----------
    all_model_results : dict
        Dictionary containing model results with keys like "fpr", "tpr", "auc", and "color".
    output_path : str or Path
        Path to save the ROC curve plot.
    title : str, optional
        Title of the plot.
    figsize : tuple, optional
        Size of the figure.
    """
    plt.figure(figsize=figsize)

    # Assign colors to models without pre-assigned colors
    model_names = [name for name, results in all_model_results.items()]
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    color_index = 0

    for model_name, results in all_model_results.items():
        # Assign color if not already assigned
        if results["color"] is None:
            results["color"] = colors[color_index]
            color_index += 1

        # Plot ROC curve
        plt.plot(
            results["fpr"],
            results["tpr"],
            lw=2,
            label=f"{model_name} (AUC = {results['auc']:.3f})",
            color=results["color"],
        )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold", labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold", labelpad=10)
    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["bottom"].set_linewidth(2)

    # Make the ticks bold
    plt.gca().tick_params(axis="both", which="major", labelsize=12, width=2)
    plt.gca().tick_params(axis="both", which="minor", labelsize=10, width=2)

    # Make the tick labels bold
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_agreement_heatmap(
    all_predictions, output_path, title="Model Prediction Agreement", figsize=(10, 8)
):
    """
    Plot a heatmap showing the agreement between model predictions.

    Parameters:
    -----------
    all_predictions : dict
        Dictionary where keys are model names and values are arrays of predictions.
    output_path : str or Path
        Path to save the heatmap plot.
    title : str, optional
        Title of the heatmap.
    figsize : tuple, optional
        Size of the figure.
    """
    # Create a DataFrame from the predictions
    agreement_df = pd.DataFrame(all_predictions)

    # Get model names
    model_names = list(all_predictions.keys())

    # Initialize agreement matrix
    agreement_matrix = np.zeros((len(model_names), len(model_names)))

    # Calculate pairwise agreement percentages
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i, j] = 1.0  # Full agreement with self
            else:
                agreement_matrix[i, j] = np.mean(
                    agreement_df[model1] == agreement_df[model2]
                )

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=model_names,
        yticklabels=model_names,
    )
    plt.title(title, fontsize=14)
    plt.tight_layout()

    # Save the heatmap
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
