from pathlib import Path

# Data paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
WORKING_DIR = Path("working_data")

# Model output paths
MODEL_OUTPUT_PATHS = {
    "all_ibd": OUTPUT_DIR / "all_ibd",
    "biochem_remission": OUTPUT_DIR / "biochem_remission",
    "dnn": OUTPUT_DIR / "dnn",
    "dnn_biochem_remission": OUTPUT_DIR / "dnn_biochem_remission",
    "dnn_clinical_remission": OUTPUT_DIR / "dnn_clinical_remission",
}


# Model training parameters


# Plotting parameters
PLOT_CONFIG = {
    "all_ibd": {"cmap": "seismic", "file_prefix": "all_ibd"},
    "biochem_remission": {"cmap": "berlin", "file_prefix": "biochem_remission"},
}
