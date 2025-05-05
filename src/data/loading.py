import json
from pathlib import Path

import joblib
import pandas as pd


def load_fatigue_dataset() -> pd.DataFrame:
    """
    Load the fatigue modelling dataset from the specified CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    # Read the data
    df = pd.read_csv("data/gidamps_music_combined_fatigue_df_111224_with_cmh.csv")

    return df


def load_online_survey_dataset() -> pd.DataFrame:
    """
    Load the online survey dataset from the specified CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    return pd.read_csv("data/cohort_online_cucq32.csv")


def load_sklearn_models_and_metadata(models_dir):
    """Load all saved models and metadata from the specified directory."""
    models_dir = Path(models_dir)

    # Load models
    models = {}
    for model_file in models_dir.glob("*_model.joblib"):
        model_name = model_file.stem.replace("_model", "").replace("_", " ").title()
        models[model_name] = joblib.load(model_file)

    # Load scaler
    scaler = joblib.load(models_dir / "scaler.pkl")

    # Load feature names
    with open(models_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)["features"]

    return models, scaler, feature_names
