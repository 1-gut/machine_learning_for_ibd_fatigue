from matplotlib import pyplot as plt

plt.rcParams["font.family"] = ["Roboto", "Arial", "sans-serif"]


def plot_training_validation_auc(history_dict, output_path, file_prefix):
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

    save_path = output_path + file_prefix + "_training_vs_validation_auc.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_validation_loss(history_dict, output_path, file_prefix):
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

    save_path = output_path + file_prefix + "_training_vs_validation_loss.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curve(fpr, tpr, test_auc, output_path, file_prefix):
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

    # Define the absolute path and save the plot
    save_path = f"{output_path}{file_prefix}_roc_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
