"""Data loading and schema parsing.

Fill in once the challenge files are available. Keep this module side-effect
free: functions return DataFrames / metadata, callers decide what to do.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw challenge dataset (CSV/parquet auto-detected)."""
    raise NotImplementedError("Wire up once dataset format is known.")


def load_sample_submission(path: str | Path | None = None) -> pd.DataFrame:
    """Load the sample submission to lock the exact output schema."""
    raise NotImplementedError("Wire up once submission format is known.")


def describe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, non-null count, n_unique, example value — first look."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "n_missing": df.isna().sum(),
            "n_unique": df.nunique(dropna=True),
            "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        }
    )
