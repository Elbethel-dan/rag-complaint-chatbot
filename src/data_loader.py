"""
Data loading utilities for various file formats.
Supports CSV, Excel, JSON, and plain text files.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> Union[pd.DataFrame, str]:
    """
    Load data from CSV, Excel, JSON, or TXT files based on file extension.

    Args:
        file_path: Path to the file to load. Supports:
            - .csv  → pandas DataFrame
            - .xlsx, .xls → pandas DataFrame
            - .json → pandas DataFrame (supports both line-delimited and standard JSON)
            - .txt  → raw text (str)

    Returns:
        pandas DataFrame for structured data files, or str for text files.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file type is unsupported or loading fails.

    Examples:
        >>> df = load_data("data.csv")
        >>> text = load_data("document.txt")
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Get file extension
    file_ext = file_path.suffix.lower().lstrip('.')
    
    if not file_ext:
        raise ValueError(f"File has no extension: {file_path}")

    try:
        if file_ext == "csv":
            logger.info(f"Loading CSV file: {file_path}")
            return pd.read_csv(file_path)

        elif file_ext in ("xlsx", "xls"):
            logger.info(f"Loading Excel file: {file_path}")
            return pd.read_excel(file_path)

        elif file_ext == "json":
            logger.info(f"Loading JSON file: {file_path}")
            # Try line-delimited JSON first (more common for large datasets)
            try:
                return pd.read_json(file_path, lines=True)
            except ValueError:
                # Fall back to standard JSON
                return pd.read_json(file_path)

        elif file_ext == "txt":
            logger.info(f"Loading text file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            supported_formats = ["csv", "xlsx", "xls", "json", "txt"]
            raise ValueError(
                f"Unsupported file type: .{file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

    except pd.errors.EmptyDataError as e:
        raise ValueError(f"File is empty: {file_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse file '{file_path}': {e}") from e
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode file '{file_path}'. Try specifying encoding.") from e
    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}") from e
