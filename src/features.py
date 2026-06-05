"""Feature construction: lags, rolling stats, time features, interactions.

Kept separate from preprocessing so feature choices are auditable AND so the
identical transform can be replayed on intervention-modified inputs (Stage 5)
without leakage. All lag/rolling features use only PAST information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_lags(df: pd.DataFrame, cols, lags, group: str | None = None) -> pd.DataFrame:
    """Append lagged copies of `cols`. Per-entity when `group` is given (panel)."""
    out = df.copy()
    for c in cols:
        for L in lags:
            name = f"{c}_lag{L}"
            out[name] = out.groupby(group)[c].shift(L) if group else out[c].shift(L)
    return out


def add_rolling(df: pd.DataFrame, cols, windows, group: str | None = None,
                stats=("mean", "std")) -> pd.DataFrame:
    """Rolling stats over PAST values (shift(1) first to exclude the current row)."""
    out = df.copy()
    for c in cols:
        base = out.groupby(group)[c].shift(1) if group else out[c].shift(1)
        for w in windows:
            r = base.rolling(w, min_periods=max(2, w // 2))
            for s in stats:
                out[f"{c}_roll{w}_{s}"] = getattr(r, s)()
        out[f"{c}_diff1"] = (out.groupby(group)[c].diff(1) if group else out[c].diff(1))
    return out


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Calendar + cyclical (sin/cos) features when the time column is a real datetime."""
    out = df.copy()
    t = out[time_col]
    if not pd.api.types.is_datetime64_any_dtype(t):
        t = pd.to_datetime(t, errors="coerce")
    if t.notna().mean() > 0.5:
        out["dow"] = t.dt.dayofweek
        out["month"] = t.dt.month
        out["woy"] = t.dt.isocalendar().week.astype("float")
        out["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365.25)
    out["trend_idx"] = np.arange(len(out))
    return out


def build_design_matrix(df: pd.DataFrame, target: str, feature_cols=None, dropna: bool = True):
    """Return (X, y, used_cols) with no target column leaking into X."""
    feature_cols = feature_cols or [c for c in df.columns if c != target]
    feature_cols = [c for c in feature_cols if c != target]
    data = df[feature_cols + [target]]
    if dropna:
        data = data.dropna()
    X = data[feature_cols].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    return X, y, feature_cols
