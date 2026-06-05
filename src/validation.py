"""Validation splitters — getting this wrong invalidates everything.

Hard rule: never random-shuffle time-ordered data. Provide temporal, blocked
(rolling-origin), grouped, and stratified schemes; choose by task type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def temporal_split(n: int, frac_train: float = 0.7, frac_valid: float = 0.15):
    """Contiguous train / valid / test by position. Test is the most recent block."""
    i1 = int(n * frac_train)
    i2 = int(n * (frac_train + frac_valid))
    return np.arange(0, i1), np.arange(i1, i2), np.arange(i2, n)


def rolling_origin(n: int, n_splits: int = 5, horizon: int | None = None, min_train: int | None = None):
    """Expanding-window CV: train [0:t], validate [t:t+h]. Yields (train_idx, val_idx)."""
    horizon = horizon or max(1, n // (n_splits + 1))
    min_train = min_train or horizon
    for k in range(n_splits):
        t = min_train + k * horizon
        if t + horizon > n:
            break
        yield np.arange(0, t), np.arange(t, min(t + horizon, n))


def grouped_temporal_split(df: pd.DataFrame, time_col: str, frac_train: float = 0.7,
                           frac_valid: float = 0.15):
    """Panel split: cut on the GLOBAL time axis so all entities share the boundary.

    Prevents leakage of an entity's future into another entity's training window.
    """
    order = df[time_col].rank(method="dense")
    q_tr, q_va = order.quantile(frac_train), order.quantile(frac_train + frac_valid)
    tr = np.where(order <= q_tr)[0]
    va = np.where((order > q_tr) & (order <= q_va))[0]
    te = np.where(order > q_va)[0]
    return tr, va, te


def kfold(n: int, n_splits: int = 5, y=None, stratify: bool = False, random_state: int = 0):
    """i.i.d. tabular CV. Stratified if a (binary/categorical) target is given."""
    idx = np.arange(n)
    if stratify and y is not None:
        for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(idx, y):
            yield tr, va
    else:
        for tr, va in KFold(n_splits, shuffle=True, random_state=random_state).split(idx):
            yield tr, va


def choose_validation(task_primary: str):
    """Return a description + a callable factory matching the task type."""
    return {
        "timeseries": ("temporal / rolling-origin", rolling_origin),
        "panel": ("grouped-temporal (global time boundary)", grouped_temporal_split),
        "treatment": ("stratified k-fold by treatment", kfold),
        "tabular": ("k-fold", kfold),
    }.get(task_primary, ("k-fold", kfold))
