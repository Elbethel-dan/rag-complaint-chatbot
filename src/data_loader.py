import pandas as pd
from typing import Union


def load_data(file_path: str) -> Union[pd.DataFrame, str]:
    """
    Load data from CSV, Excel, JSON, or TXT files based on file extension.

    Supported formats:
    - .csv  → pandas DataFrame
    - .xlsx, .xls → pandas DataFrame
    - .json → pandas DataFrame
    - .txt  → raw text (str)
    """

    file_ext = file_path.lower().split('.')[-1]

    try:
        if file_ext == "csv":
            return pd.read_csv(file_path)

        elif file_ext in ("xlsx", "xls"):
            return pd.read_excel(file_path)

        elif file_ext == "json":
            # Support both line-delimited and standard JSON
            try:
                return pd.read_json(file_path, lines=True)
            except ValueError:
                return pd.read_json(file_path)

        elif file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise ValueError(f"Unsupported file type: .{file_ext}")

    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")
