import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config.constants import RANDOM_SEED, SKLEARN_TEST_SIZE


def split_data_for_sklearn(df: pd.DataFrame) -> tuple:
    """
    Splits the input DataFrame into training and testing sets using GroupShuffleSplit.
    The split is stratified based on the 'study_id' column to ensure that the same study
    is not present in both training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be split.

    Returns:
        tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test, groups).
    """

    # GroupShuffleSplit
    splitter = GroupShuffleSplit(
        test_size=SKLEARN_TEST_SIZE, n_splits=1, random_state=RANDOM_SEED
    )

    # Perform the split
    for train_idx, test_idx in splitter.split(df, groups=df["study_id"]):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

    # Drop 'study_id' from X_train and X_test as it's not a feature
    X_train = train_data.drop(columns=["fatigue_outcome", "study_id"])
    y_train = train_data["fatigue_outcome"]

    X_test = test_data.drop(columns=["fatigue_outcome", "study_id"])
    y_test = test_data["fatigue_outcome"]

    groups = train_data["study_id"]

    return X_train, X_test, y_train, y_test, groups


def split_data_for_keras(df: pd.DataFrame) -> tuple:
    """
    Splits the input DataFrame into training, validation, and testing sets using GroupShuffleSplit.
    The split is stratified based on the 'study_id' column to ensure that the same study
    is not present in both training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be split.
    Returns:
        tuple: A tuple containing the training, validation, and testing sets (X_train, X_val, X_test, y_train, y_val, y_test, groups).
    """
    # In this we split the data with 20% testing, 80% training similar to above.
    # From the 80% training, we derive 20% of it as validation to simulate the 5-fold cross-validation done in sklearn.
    # Final ratios of the dataset are: 65% training, 15% validation, 20% testing.
    # This is why we went with 80/20 splits for above as well to make both pipelines as similar as possible.

    # GroupShuffleSplit
    splitter = GroupShuffleSplit(test_size=0.36, n_splits=1, random_state=RANDOM_SEED)

    # Perform the split
    for train_idx, test_idx in splitter.split(df, groups=df["study_id"]):
        train_data = df.iloc[train_idx]
        temp_data = df.iloc[test_idx]

    # Drop 'study_id' from X_train and X_test as it's not a feature
    X_train = train_data.drop(columns=["fatigue_outcome", "study_id"])
    y_train = train_data["fatigue_outcome"]

    groups = train_data["study_id"]  # Group variable for GroupKFold cross-validation

    temp_data_splitter = GroupShuffleSplit(
        test_size=0.56, n_splits=1, random_state=RANDOM_SEED
    )

    # Perform the split
    for val_idx, test_idx in temp_data_splitter.split(
        temp_data, groups=temp_data["study_id"]
    ):
        val_data = df.iloc[val_idx]
        test_data = df.iloc[test_idx]

    X_val = val_data.drop(columns=["fatigue_outcome", "study_id"])
    y_val = val_data["fatigue_outcome"]

    X_test = test_data.drop(columns=["fatigue_outcome", "study_id"])
    y_test = test_data["fatigue_outcome"]

    return X_train, X_val, X_test, y_train, y_val, y_test, groups
