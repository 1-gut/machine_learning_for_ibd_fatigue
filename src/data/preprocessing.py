import datetime

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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame."""
    for column in COLUMNS_TO_FILL_ZERO:
        df[column] = df[column].fillna(0)

    for column in COLUMNS_TO_FILL_MEDIAN:
        df[column] = df[column].fillna(df[column].median())
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

    df.drop(columns=unwanted_columns, inplace=True)
    return df


def preprocess_data(df: pd.DataFrame, biochemical_remission=False) -> pd.DataFrame:
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
    df = fill_missing_values(df)
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


def load_and_preprocess_data(biochemical_remission=False) -> pd.DataFrame:
    """
    Load and preprocess the fatigue dataset.

    Args:
        biochemical_remission (bool): If True, filter the DataFrame for patients in biochemical remission.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    """
    df = load_fatigue_dataset()
    df = preprocess_data(df, biochemical_remission=biochemical_remission)
    return df
