import numpy as np
import pandas as pd

from src.data.preprocessing import fill_missing_values, preprocess_data


def test_fill_missing_values():
    """Test missing value imputation."""
    df = pd.DataFrame({"zero_fill": [1, np.nan, 3], "median_fill": [10, np.nan, 30]})

    result = fill_missing_values(df)

    assert result["zero_fill"].isna().sum() == 0
    assert result["median_fill"].median() == 20.0


def test_preprocess_data():
    """Test full preprocessing pipeline."""
    sample_data = pd.DataFrame(
        {
            "cucq_5": [12, 8, 15],
            "height": [170, np.nan, 180],
            "montreal_cd_location": ["L1 Ileal", "L2 Colonic", "L3 Ileocolonic"],
        }
    )

    result = preprocess_data(sample_data)

    assert "fatigue_outcome" in result.columns
    assert result["height"].isna().sum() == 0
