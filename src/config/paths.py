from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"

# Demographics
DEMOGRAPHICS_DIR = OUTPUT_DIR / "demographics"

# All IBD
DNN_DIR = OUTPUT_DIR / "all_ibd" / "dnn"
SKLEARN_DIR = OUTPUT_DIR / "all_ibd" / "sklearn"
VALIDATION_SPAIN_DIR = OUTPUT_DIR / "all_ibd" / "validation" / "spain"
VALIDATION_AUSTRALIA_DIR = OUTPUT_DIR / "all_ibd" / "validation" / "australia"

# Biochemical remission subgroup
DNN_BIOCHEM_REMISSION_DIR = OUTPUT_DIR / "biochemical_remission" / "dnn"
SKLEARN_BIOCHEM_REMISSION_DIR = OUTPUT_DIR / "biochemical_remission" / "sklearn"

# Create common directories: exports, plots, metrics


def ensure_output_dirs():
    """Create output directories if they don't exist"""
    dirs = [
        OUTPUT_DIR,
        DEMOGRAPHICS_DIR,
        DNN_DIR,
        SKLEARN_DIR,
        VALIDATION_SPAIN_DIR,
        VALIDATION_AUSTRALIA_DIR,
        DNN_BIOCHEM_REMISSION_DIR,
        SKLEARN_BIOCHEM_REMISSION_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for model outputs
    model_dirs = [
        DNN_DIR,
        SKLEARN_DIR,
        DNN_BIOCHEM_REMISSION_DIR,
        SKLEARN_BIOCHEM_REMISSION_DIR,
    ]
    subdirs = ["exports", "plots", "metrics", "evaluation"]

    for dir in model_dirs:
        for subdir in subdirs:
            (dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create forceplots directory for DNN models only
    dnn_dirs = [DNN_DIR, DNN_BIOCHEM_REMISSION_DIR]
    for dir in dnn_dirs:
        (dir / "plots" / "forceplots").mkdir(parents=True, exist_ok=True)
