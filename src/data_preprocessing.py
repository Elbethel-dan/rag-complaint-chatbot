# data_preprocessing.py

import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from .data_loader import load_data

class ComplaintPreprocessor:
    """
    Fully-featured text preprocessing class for complaint narratives.
    Features:
    - Lowercase text
    - Remove emails, phone numbers, special characters
    - Remove boilerplate sentences
    - Remove stopwords
    - Lemmatize words
    - Remove placeholders
    - Drop empty rows
    - Informative logging of cleaning steps
    """

    PLACEHOLDERS = ["xxxx", "xxxxx", "xxxxxx", "---", "n/a", "na", "unknown"]

    def __init__(self, boilerplate_file: str = None, stopwords_file: str = None, verbose: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.verbose = verbose

        # Load boilerplate sentences
        if boilerplate_file:
            raw = load_data(boilerplate_file)
            if isinstance(raw, str):
                self.boilerplate = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                raise ValueError(f"Expected a text file for boilerplate, got {type(raw)}")
        else:
            self.boilerplate = []

        # Load stopwords
        if stopwords_file:
            raw = load_data(stopwords_file)
            if isinstance(raw, str):
                self.stop_words = [line.strip() for line in raw.splitlines() if line.strip()]
            else:
                raise ValueError(f"Expected a text file for stopwords, got {type(raw)}")
        else:
            self.stop_words = []

    # -------------------------
    # Text cleaning methods
    # -------------------------
    @staticmethod
    def _lowercase_text(df: pd.DataFrame, column: str) -> pd.DataFrame:
        df[column] = df[column].str.lower()
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove emails
        text = re.sub(r"(\+?\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}", "", text)  # Remove phones
        text = re.sub(r"[^a-zA-Z0-9\s\.?]", " ", text)  # Keep letters, numbers, space, '.' and '?'
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _remove_boilerplate(self, text: str) -> str:
        if pd.isna(text):
            return ""
        for sentence in self.boilerplate:
            text = re.sub(re.escape(sentence), "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def _remove_stopwords(self, text: str) -> str:
        if pd.isna(text):
            return ""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)

    def _lemmatize_text(self, text: str) -> str:
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    @classmethod
    def _remove_placeholders(cls, text: str) -> str:
        if pd.isna(text):
            return ""
        for ph in cls.PLACEHOLDERS:
            text = text.replace(ph, "")
        return re.sub(r"\s+", " ", text).strip()

    # -------------------------
    # Main preprocessing pipeline
    # -------------------------
    def preprocess(self, df: pd.DataFrame, column: str, sample_size: int = 3) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if self.verbose:
            print("="*70)
            print("STARTING PREPROCESSING PIPELINE")
            print(f"Initial number of rows: {len(df):,}")
            print("="*70)
            print(f"Sample before any cleaning ({sample_size} rows):")
            print(df[column].head(sample_size))
            print("-"*70)

        before_rows = len(df)

        # Lowercase
        df = self._lowercase_text(df, column)
        if self.verbose:
            print(f"After LOWERCASE ({sample_size} rows):")
            print(df[column].head(sample_size))
            print("-"*70)

        # Clean text
        df[column] = df[column].apply(self._clean_text)
        if self.verbose:
            print(f"After CLEAN_TEXT ({sample_size} rows):")
            print(df[column].head(sample_size))
            print("-"*70)

        # Remove boilerplate
        if self.boilerplate:
            df[column] = df[column].apply(self._remove_boilerplate)
            if self.verbose:
                print(f"After BOILERPLATE REMOVAL ({sample_size} rows):")
                print(df[column].head(sample_size))
                print("-"*70)

        # Remove stopwords
        if self.stop_words:
            df[column] = df[column].apply(self._remove_stopwords)
            if self.verbose:
                print(f"After STOPWORDS REMOVAL ({sample_size} rows):")
                print(df[column].head(sample_size))
                print("-"*70)

        # Lemmatize
        df[column] = df[column].apply(self._lemmatize_text)
        if self.verbose:
            print(f"After LEMMATIZATION ({sample_size} rows):")
            print(df[column].head(sample_size))
            print("-"*70)

        # Remove placeholders
        if self.verbose:
            placeholder_count_before = df[column].str.contains("|".join(self.PLACEHOLDERS), case=False, na=False).sum()
        df[column] = df[column].apply(self._remove_placeholders)
        if self.verbose:
            placeholder_count_after = df[column].str.contains("|".join(self.PLACEHOLDERS), case=False, na=False).sum()
            removed_placeholders = placeholder_count_before - placeholder_count_after
            print(f"Number of complaints containing placeholders removed: {removed_placeholders:,}")
            print("-"*70)

        # Drop empty rows
        df = df[df[column].str.strip() != ""].copy()
        after_rows = len(df)
        if self.verbose:
            print(f"Rows removed because they were empty after cleaning: {before_rows - after_rows:,}")
            print(f"Remaining rows: {after_rows:,}")
            print("="*70)
            print("PREPROCESSING PIPELINE COMPLETED")
            print("="*70)

        return df
