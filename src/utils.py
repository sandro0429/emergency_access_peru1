"""
utils.py — Helper functions for the emergency access analysis.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def ensure_dir(path: str | Path) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return str(path)


def pad_ubigeo(series: pd.Series) -> pd.Series:
    """
    Standardize UBIGEO codes to 6-digit zero-padded strings.
    """
    return series.astype(str).str.strip().str.zfill(6)


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize text columns: uppercase, strip whitespace, remove accents."""
    import unicodedata

    def _clean(text):
        if pd.isna(text):
            return text
        text = str(text).strip().upper()
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    return series.apply(_clean)


def minmax_scale(series: pd.Series) -> pd.Series:
    """Min-max normalization to [0, 1] range."""
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series(0.5, index=series.index)
    return (series - smin) / (smax - smin)


def rank_percentile(series: pd.Series) -> pd.Series:
    """Convert values to percentile ranks [0, 1]."""
    return series.rank(pct=True)


def safe_log(series: pd.Series) -> pd.Series:
    """Log transform with log(1 + x) to handle zeros."""
    return np.log1p(series)


def print_summary(df: pd.DataFrame, name: str) -> None:
    """Print a summary of a DataFrame for documentation."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"  Dtypes:\n{df.dtypes}")