import math
from pathlib import Path

import matplotlib.pyplot as plt
import shap
from PIL import Image

plt.rcParams["font.family"] = ["Roboto", "Arial", "sans-serif"]


def create_shap_plot(
    model, X_test, model_name, cmap, output_path, sample_size=None, model_type="tree"
):
    """Create and save SHAP plots for the given model.

    Args:
        model: Trained model
        X_test: Test data
        model_name: Name of the model for plot title and filename
        cmap: Colormap to use
        output_path: Path to save the plot
        sample_size: Optional size to sample X_test (for KernelExplainer)
        model_type: Type of model ('tree', 'linear', 'kernel', or 'xgboost')
    """
    plt.figure(figsize=(10, 8))

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap_values_class_1 = shap_values[:, :, 1]
        feature_names = X_test.columns.tolist()
        shap.summary_plot(
            shap_values_class_1,
            X_test,
            feature_names=feature_names,
            show=False,
            cmap=cmap,
        )
    elif model_type == "linear":
        masker = shap.maskers.Independent(data=X_test)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(
            shap_values, X_test, feature_names=X_test.columns, show=False, cmap=cmap
        )
        plt.xlim(-1, 1)
    elif model_type == "kernel":
        X_sample = shap.sample(X_test, sample_size)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap_values_class_1 = shap_values[:, :, 1]
        shap.summary_plot(
            shap_values_class_1,
            X_sample,
            feature_names=X_sample.columns,
            show=False,
            cmap=cmap,
        )
    elif model_type == "xgboost":
        explainer = shap.Explainer(model, X_test)
        shap_values = explainer(X_test)
        shap.plots.beeswarm(
            shap_values, max_display=20, show=False, color=plt.get_cmap(cmap)
        )

    # Add title and save
    plt.title(f"{model_name}", fontsize=20, pad=20, loc="left")
    plt.tight_layout()
    save_path = (
        output_path / "plots" / f"shap_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_shap_grid(load_path: Path, save_path: Path):
    """
    Arranges SHAP plot images into a grid and saves it as a single image.
    This function assumes that SHAP plot images are saved in the "plots" directory and are named with the prefix "shap_".
    It creates a grid layout based on the number of images and saves the resulting image to the specified path.
    The grid will have a maximum of 2 columns, and the number of rows will be determined by the number of images.
    The images are pasted into the grid in the order they are found in the directory.
    If no images are found, a message is printed and the function exits without creating a grid.

    Args:
        load_path (Path): Path to the directory containing SHAP plot images.
        save_path (Path): Path to save the grid image.
    """

    # Get all the SHAP plot images
    image_paths = list(load_path.glob("shap_*.png"))
    image_paths = [path for path in image_paths if "grid" not in path.name]

    if not image_paths:
        print("No SHAP plot images found to create grid.")
        return
    images = [Image.open(img_path) for img_path in image_paths]

    # Calculate grid dimensions
    n_images = len(images)
    n_cols = min(2, n_images)
    n_rows = math.ceil(n_images / n_cols)

    # Get dimensions of the first image
    width, height = images[0].size

    # Create a new blank image
    grid_image = Image.new("RGB", (n_cols * width, n_rows * height), color="white")

    # Paste the images into the grid
    for i, img in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        grid_image.paste(img, (col * width, row * height))

    grid_image.save(save_path)
    print(f"Grid image created at {save_path}")
