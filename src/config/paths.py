from pathlib import Path

# This file sets up the output directory structure for the project.
# You can use ensure_output_dirs() in notebooks to ensure the output structure is created.
#
# The output directory structure is as follows:
# ├── output
# │   ├── demographics
# │   ├── all_ibd
# │   │   ├── dnn
# │   │   │   ├── exports
# │   │   │   ├── plots
# │   │   │   │   ├── forceplots
# │   │   │   ├── metrics
# │   │   │   ├── evaluation
# │   │   ├── sklearn
# │   │   │   ├── exports
# │   │   │   ├── plots
# │   │   │   ├── metrics
# │   │   │   ├── evaluation
# │   ├── biochemical_remission
# │   │   ├── dnn
# │   │   │   ├── exports
# │   │   │   ├── plots
# │   │   │   │   ├── forceplots
# │   │   │   ├── metrics
# │   │   │   ├── evaluation
# │   │   ├── sklearn
# │   │   │   ├── exports
# │   │   │   ├── plots
# │   │   │   ├── metrics
# │   │   │   ├── evaluation
# │   ├── validation
# │   │   ├── spain
# │   │   ├── australia
# │   │   ├── scotland
# │   │   ├── norway
# │   │   ├── combined

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"

# Demographics
DEMOGRAPHICS_DIR = OUTPUT_DIR / "demographics"

# All IBD
DNN_DIR = OUTPUT_DIR / "all_ibd" / "dnn"
SKLEARN_DIR = OUTPUT_DIR / "all_ibd" / "sklearn"
BENCHMARKING_DIR = OUTPUT_DIR / "all_ibd" / "benchmarking"

# Validation directories
VALIDATION_SPAIN_DIR = OUTPUT_DIR / "validation" / "spain"
VALIDATION_AUSTRALIA_DIR = OUTPUT_DIR / "validation" / "australia"
VALIDATION_SCOTLAND_DIR = OUTPUT_DIR / "validation" / "scotland"
VALIDATION_NORWAY_DIR = OUTPUT_DIR / "validation" / "norway"
VALIDATION_COMBINED_DIR = OUTPUT_DIR / "validation" / "combined"

# Biochemical remission subgroup
DNN_BIOCHEM_REMISSION_DIR = OUTPUT_DIR / "biochemical_remission" / "dnn"
SKLEARN_BIOCHEM_REMISSION_DIR = OUTPUT_DIR / "biochemical_remission" / "sklearn"

# Clustering directories
CLUSTERING_DIR = OUTPUT_DIR / "clustering"

# Create common directories: exports, plots, metrics


def ensure_output_dirs():
    """Create output directories if they don't exist"""
    dirs = [
        OUTPUT_DIR,
        DEMOGRAPHICS_DIR,
        DNN_DIR,
        SKLEARN_DIR,
        BENCHMARKING_DIR,
        VALIDATION_SPAIN_DIR,
        VALIDATION_AUSTRALIA_DIR,
        VALIDATION_SCOTLAND_DIR,
        VALIDATION_NORWAY_DIR,
        VALIDATION_COMBINED_DIR,
        DNN_BIOCHEM_REMISSION_DIR,
        SKLEARN_BIOCHEM_REMISSION_DIR,
        CLUSTERING_DIR,
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
