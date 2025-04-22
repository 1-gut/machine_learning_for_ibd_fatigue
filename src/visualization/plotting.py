from typing import Optional

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


def plot_roc_curve(fpr, tpr, test_auc, output_path: Optional[str] = None):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=20, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=20, fontweight="bold")
    plt.title(
        "Performance of Keras DNN on Test Set", fontsize=24, fontweight="bold", pad=40
    )
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
