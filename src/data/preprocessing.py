import datetime
import json
from pathlib import Path

import pandas as pd

from src.config.preprocessing import (
    CATEGORICAL_FEATURES_FOR_ONE_HOT_ENCODING,
    COLUMNS_TO_FILL_MEDIAN,
    COLUMNS_TO_FILL_ZERO,
    UNUSED_COLUMNS,
)
from src.data.loading import load_fatigue_dataset


def map_fatigue_outcome(score: int) -> int:
    """
    Score is the answer to the question: On how many days over the last 14 days have you felt fatigued? (0-14)
    We found a threshold of >=10 days to be the cut off (median derived from the patient survey dataset).
    If the score is greater than or equal to 10, it returns 1 (indicating fatigue).
    If the score is less than 10, it returns 0 (indicating no fatigue).
    """
    return 1 if score >= 10 else 0


def map_season(date: datetime.date) -> str:
    """
    Takes in a date and returns the corresponding season.
    The seasons are defined as follows:
    - Winter: December, January, February
    - Spring: March, April, May
    - Summer: June, July, August
    - Autumn: September, October, November
    If the month is not recognized, it returns "no_data".

    Parameters
    ----------
    date : datetime.date
        The date to map to a season.

    Returns
    -------
    str
        The season corresponding to the date.
    """
    month = date.month
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "no_data"


def map_season_australia(date: datetime.date) -> str:
    """
    Takes in a date and returns the corresponding season in Australia.
    The seasons are defined as follows:
    - Summer: December, January, February
    - Autumn: March, April, May
    - Winter: June, July, August
    - Spring: September, October, November
    If the month is not recognized, it returns "no_data".

    Parameters
    ----------
    date : datetime.date
        The date to map to a season.

    Returns
    -------
    str
        The season corresponding to the date.
    """
    month = date.month
    if month in [12, 1, 2]:
        return "summer"
    elif month in [3, 4, 5]:
        return "autumn"
    elif month in [6, 7, 8]:
        return "winter"
    elif month in [9, 10, 11]:
        return "spring"
    else:
        return "no_data"


def fill_missing_values(
    df: pd.DataFrame, save_medians=False, medians_path: Path = None
) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame and optionally save median values to a JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to fill missing values in
    save_medians : bool, default=False
        If True, saves the computed median values to a JSON file
    medians_path : Path, optional
        Path where to save the JSON file with median values.

    Returns
    -------
    pd.DataFrame
        The dataframe with filled values
    """
    for column in COLUMNS_TO_FILL_ZERO:
        df[column] = df[column].fillna(0)

    # Calculate and store median values
    median_values = {}
    for column in COLUMNS_TO_FILL_MEDIAN:
        median_val = df[column].median()
        median_values[column] = float(
            median_val
        )  # Convert numpy.float to Python float for JSON serialization
        df[column] = df[column].fillna(median_val)

    # Save median values to JSON if requested
    if save_medians:
        if medians_path is None:
            raise ValueError("medians_path must be provided if save_medians is True")
        with open(medians_path, "w") as f:
            json.dump(median_values, f, indent=4)
            print(f"Median values saved to {medians_path}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns from the DataFrame."""
    df.drop(columns=UNUSED_COLUMNS, inplace=True)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df["montreal_cd_location"] = df["montreal_cd_location"].replace(
        "L3 Ileocoloni", "L3 Ileocolonic"
    )  # Fix typo in the data

    df = pd.get_dummies(
        df, columns=CATEGORICAL_FEATURES_FOR_ONE_HOT_ENCODING, dtype=int
    )

    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Create the target variable for the model."""
    df["fatigue_outcome"] = df["cucq_5"].apply(map_fatigue_outcome)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering
    Creates the following new columns:
    1. Disease duration in weeks: Float
    2. Season of cucq date: String
    3. Year of diagnosis: Integer
    4. ADA and IFX drug level present: 0 or 1
    """

    df["cucq_date"] = pd.to_datetime(df["cucq_date"])
    df["date_of_diagnosis"] = pd.to_datetime(df["date_of_diagnosis"])

    df["disease_duration_weeks"] = (
        df["cucq_date"] - df["date_of_diagnosis"]
    ).dt.days / 7
    df["season"] = df["cucq_date"].apply(map_season)
    df["diagnosis_year"] = df["date_of_diagnosis"].dt.year

    # Create these 2 columns so we can zero out the drug level column where data is not available
    df["ada_drug_level_present"] = df["ada_drug_level"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )
    df["ifx_drug_level_present"] = df["ifx_drug_level"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )

    return df


def convert_categorical_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Define columns to drop
    unwanted_columns = [
        "aggregate_disease_activity_Active",
        "aggregate_disease_activity_Biochemical remission",
        "aggregate_disease_activity_Remission",
        "season_no_data",
        "study",
        "redcap_event_name_timepoint_1",
        "redcap_event_name_timepoint_2",
        "redcap_event_name_timepoint_3",
        "redcap_event_name_timepoint_4",
        "redcap_event_name_timepoint_5",
        "baseline_eims_pyoderma_gangrenosum",  # all columns are 0
    ]

    # Some dummy columns may be absent (e.g., if a category does not appear in a subset),
    # so ignore missing columns instead of raising errors.
    df.drop(columns=unwanted_columns, inplace=True, errors="ignore")
    return df


def remove_low_value_features(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        "baseline_eims_arthralgia_arthritis",
        "baseline_eims_ankylosing_spondylitis",
        "baseline_eims_erythema_nodosum",
        "baseline_eims_uveitis",
        "baseline_eims_scleritis_episclerities",
        "is_smoker_smokeryn1",
        "study_group_name_Await Dx",
        "ifx_drug_level",
        "ada_drug_level",
        "ifx_drug_level_present",
        "ada_drug_level_present",
        "ifx_antibody_present",
        "ada_antibody_present",
        "haematocrit",
    ]

    df.drop(columns=columns_to_drop, inplace=True)
    return df


def fix_numerical_features_for_production(numerical_features: list) -> list:
    numerical_features = [
        col
        for col in numerical_features
        if col not in ["haematocrit", "ada_drug_level", "ifx_drug_level"]
    ]
    return numerical_features


def preprocess_data(
    df: pd.DataFrame,
    biochemical_remission=False,
    save_medians=False,
    medians_path: Path = None,
) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    Please note the order of operations is important.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        biochemical_remission (bool): If True, filter the DataFrame for patients in biochemical remission.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = create_target_variable(df)
    df = convert_categorical_columns_to_numeric(df)
    df = feature_engineering(df)
    df = fill_missing_values(df, save_medians=save_medians, medians_path=medians_path)
    df = drop_unused_columns(df)

    # If you wish to analyse the dataframe in R without one hot encoding,
    # uncomment the line below to save the dataframe as a csv file
    # df.to_csv("working_data/all_ibd_R_analysis.csv", index=False)

    df = one_hot_encode(df)

    if biochemical_remission:
        df = df[df["aggregate_disease_activity_Biochemical remission"] == 1]

    # this operation can only happen after subsetting for disease activity
    df = drop_unwanted_columns(df)

    return df


def load_and_preprocess_data(
    biochemical_remission=False, save_medians=False, medians_path: Path = None
) -> pd.DataFrame:
    """
    Load and preprocess the fatigue dataset.

    Args:
        biochemical_remission (bool): If True, filter the DataFrame for patients in biochemical remission.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    """
    df = load_fatigue_dataset()
    df = preprocess_data(
        df,
        biochemical_remission=biochemical_remission,
        save_medians=save_medians,
        medians_path=medians_path,
    )
    return df


def fill_missing_values_with_saved_medians(
    df: pd.DataFrame, medians_path: Path
) -> pd.DataFrame:
    with open(medians_path, "r") as f:
        median_values = json.load(f)

    for column in COLUMNS_TO_FILL_ZERO:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    for column, value in median_values.items():
        if column in df.columns:
            df[column] = df[column].fillna(value)

    return df
