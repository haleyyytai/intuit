"""Cleaning, diagnostics, and train/validation/test splitting.

Diagnostic helpers here are meant to be run BEFORE modeling to satisfy Stage 1
(leakage, missingness, nonstationarity, duplicates, impossible values,
temporal-ordering, distribution shift).
"""
from __future__ import annotations

import pandas as pd


def basic_health_checks(df: pd.DataFrame) -> dict:
    """Return a dict of cheap red-flag diagnostics for the Stage 1 report."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_by_col": df.isna().mean().sort_values(ascending=False).to_dict(),
        "constant_cols": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }


def check_stationarity(series: pd.Series) -> dict:
    """ADF / KPSS style stationarity test (fill in for temporal data)."""
    raise NotImplementedError("Use statsmodels adfuller/kpss once data type known.")


def make_split(df: pd.DataFrame, strategy: str = "temporal"):
    """Return (train, valid, test) indices. strategy in {temporal, panel, random}."""
    raise NotImplementedError("Choose split after data type is classified.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply agreed cleaning steps (impute, clip impossible values, dedup)."""
    raise NotImplementedError("Define after inspecting the data.")
