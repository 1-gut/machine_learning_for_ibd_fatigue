import pandas as pd


def load_and_preprocess_data(
    filepath="working_data/all_ibd_ml_input.csv", biochem=False
):
    """
    Load and preprocess the data.
    By default, the data is loaded from the working_data/all_ibd_ml_input.csv file.
    If biochem is True, the data is filtered to only include patients in biochemical remission.

    Parameters
    ----------
    filepath : str, optional
        The path to the input data file, by default "working_data/all_ibd_ml_input.csv"
    biochem : bool, optional
        Whether to filter the data to only include patients in biochemical remission, by default False

    Returns
    -------
    pd.DataFrame
        The preprocessed data
    """
    # Read the data
    df = pd.read_csv(filepath)

    # Optionally filter by biochemical remission
    if biochem:
        df = df[df["aggregate_disease_activity_Biochemical remission"] == 1]

    # Convert categorical columns to numerical
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["fatigue_outcome"] = df["fatigue_outcome"].map({"fatigue": 1, "no_fatigue": 0})

    # Define columns to drop
    drop_columns = [
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

    # Drop unwanted columns
    df.drop(columns=drop_columns, inplace=True)

    return df
