from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams["font.family"] = ["Roboto", "Arial", "sans-serif"]


def plot_training_validation_auc(history_dict, output_path: Optional[str] = None):
    try:
        auc_values = history_dict["auc"]
        val_auc_values = history_dict["val_auc"]
    except KeyError:
        try:
            auc_values = history_dict["auc_1"]
            val_auc_values = history_dict["val_auc_1"]
        except KeyError:
            auc_values = history_dict["auc_2"]
            val_auc_values = history_dict["val_auc_2"]
    epochs = range(1, len(auc_values) + 1)

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, auc_values, "bo", label="Training AUC")
    plt.plot(epochs, val_auc_values, "b", label="Validation AUC")
    plt.title("Training and Validation AUC", fontsize=24, fontweight="bold", pad=40)
    plt.xlabel("Epochs", fontsize=20, fontweight="bold")
    plt.ylabel("AUC", fontsize=20, fontweight="bold")
    plt.legend(fontsize=14)
    plt.ylim(0, 1)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", labelsize=16, width=2)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_validation_loss(history_dict, output_path: Optional[str] = None):
    plt.figure(figsize=(10, 8))
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss", fontsize=24, fontweight="bold", pad=40)
    plt.xlabel("Epochs", fontsize=20, fontweight="bold")
    plt.ylabel("Loss", fontsize=20, fontweight="bold")
    plt.legend(fontsize=14)
    plt.ylim(0, 1)
    # Remove the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", labelsize=16, width=2)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    fpr,
    tpr,
    test_auc,
    output_path: Optional[str] = None,
    title: str = "Performance of Keras DNN on Test Set",
):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=20, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=20, fontweight="bold")
    plt.title(label=title, fontsize=24, fontweight="bold", pad=40)
    plt.legend(loc="lower right", fontsize=14)

    # Remove the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", labelsize=16, width=2)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_cucq_trajectory(
    df, title: str = "Longitudinal CUCQ Trajectories", output_path: Optional[str] = None
):
    """Plot CUCQ score trajectories over time.

    Args:
        df: DataFrame containing CUCQ data
        title: Plot title
        output_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 8))

    # Plot individual trajectories
    for study_id in df["study_id"].unique():
        patient_data = df[df["study_id"] == study_id]
        plt.plot(
            "redcap_event_name",
            "cucq_total",
            data=patient_data,
            marker="o",
            linestyle="-",
            alpha=0.3,
            color="gray",
        )

    # Calculate and plot mean trajectory
    mean_trajectory = df.groupby("redcap_event_name")["cucq_total"].mean()
    plt.plot(
        mean_trajectory.index,
        mean_trajectory.values,
        color="red",
        linewidth=3,
        label="Mean CUCQ",
    )

    plt.title(title, fontsize=24, fontweight="bold")
    plt.xlabel("Timepoint", fontsize=20, fontweight="bold", labelpad=20)
    plt.ylabel("CUCQ Score", fontsize=20, fontweight="bold", labelpad=20)
    # Map axis labels
    timepoint_labels = {
        "timepoint_1": "Baseline",
        "timepoint_2": "3 months",
        "timepoint_3": "6 months",
        "timepoint_4": "9 months",
        "timepoint_5": "12 months",
    }
    plt.xticks(
        ticks=list(timepoint_labels.keys()),
        labels=list(timepoint_labels.values()),
        fontsize=16,
    )
    plt.yticks(fontsize=16)
    # Remove the top and right spines
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot confusion matrix with annotations and save to file.

    Args:
        y_true: Ground truth (correct) target values
        y_pred: Estimated targets as returned by a classifier
        save_path: Path where the plot will be saved
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure and axes
    plt.figure(figsize=(10, 8))

    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Fatigue", "Fatigue"],
        yticklabels=["No Fatigue", "Fatigue"],
        annot_kws={"size": 20},
        # cbar=False,
    )

    # Add labels and title
    plt.title("Confusion Matrix", fontsize=24, pad=20)
    plt.xlabel("Predicted Label", fontsize=20, labelpad=10)
    plt.ylabel("True Label", fontsize=20, labelpad=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curves(
    roc_results: dict,
    dnn_fpr: np.ndarray = None,
    dnn_tpr: np.ndarray = None,
    dnn_metrics: dict = None,
    biochemical_remission=False,
    output_path: Path = None,
    title: str = None,
):
    """
    Plot ROC curves for multiple models from sklearn,
    optionally including DNN comparison from TensorFlow.

    Args:
        roc_results (dict): Dictionary containing ROC data for multiple models
        dnn_fpr (array, optional): False positive rates for DNN model
        dnn_tpr (array, optional): True positive rates for DNN model
        dnn_metrics (dict, optional): Dictionary containing DNN metrics
        biochemical_remission (bool): Whether analysis is for biochemical remission cohort
        output_path (Path): Directory to save the plot
        title (str, optional): Custom title for the plot
    """
    plt.figure(figsize=(8, 6))

    # Assign colors to models
    model_names = list(roc_results.keys())
    n_models = len(model_names)
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))
    color_index = 0

    # Plot ROC curves for sklearn models
    for model_name, roc_data in roc_results.items():
        test_auc = roc_data["test_auc"]
        plt.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            lw=2,
            label=f"{model_name} (AUC = {test_auc:.3f})",
            color=colors[color_index],
        )
        color_index += 1

    # Add DNN model if data is available
    dnn_included = False
    if (
        dnn_fpr is not None
        and dnn_tpr is not None
        and dnn_metrics is not None
        and dnn_metrics.get("auc") is not None
    ):
        dnn_auc = dnn_metrics["auc"]
        plt.plot(
            dnn_fpr,
            dnn_tpr,
            lw=2,
            label=f"Deep Neural Network (AUC = {dnn_auc:.3f})",
            color="darkorange",
        )
        dnn_included = True
    else:
        print("DNN ROC data not available, skipping DNN plot.")

    # Add diagonal line
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance (AUC = 0.50)"
    )

    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16, fontweight="bold", labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=16, fontweight="bold", labelpad=10)

    # Set title based on parameters
    if title:
        plt.title(title, fontsize=18, fontweight="bold", pad=20)
    else:
        if dnn_included:
            model_prefix = "DNN Outperforms Other ML Algorithms on Test Set"
        else:
            model_prefix = "ML Algorithm Comparison"
        if biochemical_remission:
            plt.title(
                f"{model_prefix} (Biochemical Remission)",
                fontsize=18,
                fontweight="bold",
                pad=20,
            )
        else:
            plt.title(
                f"{model_prefix} (All IBD)", fontsize=18, fontweight="bold", pad=20
            )

    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Remove top and right spines, thicken left/bottom
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(axis="both", which="major", labelsize=12, width=2)
    ax.tick_params(axis="both", which="minor", labelsize=10, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(12)

    # Save the plot if output path is provided
    if output_path:
        save_path = output_path / "plots" / "combined_roc_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")

    print("ROC curves plotted successfully. DNN data included:", dnn_included)
    plt.show()


def plot_missing_data_heatmap(df: pd.DataFrame):
    """
    Plot a heatmap of missing data in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to visualize missing data.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Missing Data Heatmap", fontsize=24, fontweight="bold")
    plt.xlabel("Features", fontsize=20, fontweight="bold")
    plt.ylabel("Samples", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
