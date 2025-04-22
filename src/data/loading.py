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
