# tests/test_data_preprocessing.py

import pandas as pd
import pytest
from src.data_preprocessing import ComplaintPreprocessor

# -----------------------------
# Fixture for a sample DataFrame
# -----------------------------
@pytest.fixture
def sample_df():
    data = {
        "Consumer complaint narrative": [
            "This is a TEST complaint!! Please ignore xxxx.",
            "Contact me at john@example.com or +1 (555) 123-4567.",
            None,
            "   "
        ]
    }
    return pd.DataFrame(data)

# -----------------------------
# Tests for individual methods
# -----------------------------
def test_lowercase_text(sample_df):
    df = ComplaintPreprocessor._lowercase_text(sample_df.copy(), "Consumer complaint narrative")
    assert df["Consumer complaint narrative"].iloc[0].islower()

def test_clean_text_removes_email_phone():
    text = "Email me at test@example.com or call +1 555-123-4567."
    cleaned = ComplaintPreprocessor._clean_text(text)
    assert "@" not in cleaned
    assert "+" not in cleaned
    assert "555" not in cleaned

def test_remove_placeholders():
    text = "Please ignore xxxx and xxxxxx this."
    cleaned = ComplaintPreprocessor._remove_placeholders(text)
    assert "xxxx" not in cleaned
    assert "xxxxxx" not in cleaned

# -----------------------------
# Test the full preprocessing pipeline
# -----------------------------
def test_preprocess_pipeline(sample_df):
    preprocessor = ComplaintPreprocessor(verbose=False)
    df_cleaned = preprocessor.preprocess(sample_df.copy(), "Consumer complaint narrative")

    # Check that None and blank rows are removed
    assert df_cleaned.shape[0] == 2  # Only 2 non-empty rows should remain

    # Check that placeholders are removed
    assert "xxxx" not in df_cleaned.iloc[0, 0]

    # Check that text is lowercase
    assert df_cleaned.iloc[0, 0].islower()

    # Check that the final output is string
    assert isinstance(df_cleaned.iloc[0, 0], str)

