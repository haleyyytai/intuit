"""Probability calibration + interval coverage tuning for Deliverables A/B.

Two jobs:
  1. POINT calibration: map raw model PD -> calibrated PD so predicted probabilities
     match empirical default frequencies (isotonic regression). Scored directly.
  2. INTERVAL calibration: tune a single width multiplier so the 90% intervals attain
     ~90% empirical coverage, assessed by a binned-reliability check on held-out
     labeled data (the true latent PD is unobservable, so we check whether each PD
     bin's empirical default rate falls inside that bin's averaged interval).

Honest assessment uses 2-fold cross-fitting on the labeled validation set; the final
submission refits isotonic on all labeled validation rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def fit_isotonic(p_raw, y) -> IsotonicRegression:
    """Monotone calibration map raw_pd -> P(default). Clips out-of-range inputs."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        np.asarray(p_raw, float), np.asarray(y, float))


def ece(p, y, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower = better calibrated)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def binned_coverage(p, lo, hi, y, n_bins: int = 10) -> float:
    """Fraction of PD bins whose empirical default rate lies in the bin's mean interval."""
    df = pd.DataFrame({"p": p, "lo": lo, "hi": hi, "y": y})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return np.nan
    covered = 0; total = 0
    for _, g in df.groupby("bin", observed=True):
        emp = g["y"].mean()
        if g["lo"].mean() - 1e-9 <= emp <= g["hi"].mean() + 1e-9:
            covered += 1
        total += 1
    return covered / max(1, total)


def tune_width_multiplier(p_cal, base_half, y, target: float = 0.90,
                          grid=None, n_bins: int = 10) -> float:
    """Pick the smallest width multiplier whose binned coverage >= target."""
    grid = grid if grid is not None else np.linspace(0.3, 4.0, 38)
    best = grid[-1]
    for m in grid:
        cov = binned_coverage(p_cal, np.clip(p_cal - m * base_half, 0, 1),
                              np.clip(p_cal + m * base_half, 0, 1), y, n_bins)
        if not np.isnan(cov) and cov >= target:
            best = m
            break
    return float(best)


def crossfit_report(p_raw, base_half, y, seed: int = 0) -> dict:
    """2-fold cross-fit: fit isotonic + tune width on one fold, assess on the other.

    Returns honest before/after ECE/Brier and interval coverage.
    """
    p_raw, base_half, y = map(lambda a: np.asarray(a, float), (p_raw, base_half, y))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    folds = np.array_split(order, 2)
    cal_cov, cal_ece, raw_ece, cal_brier, raw_brier = [], [], [], [], []
    for i in range(2):
        te = folds[i]; trn = folds[1 - i]
        iso = fit_isotonic(p_raw[trn], y[trn])
        m = tune_width_multiplier(iso.predict(p_raw[trn]), base_half[trn], y[trn])
        p_cal = iso.predict(p_raw[te])
        lo = np.clip(p_cal - m * base_half[te], 0, 1)
        hi = np.clip(p_cal + m * base_half[te], 0, 1)
        cal_cov.append(binned_coverage(p_cal, lo, hi, y[te]))
        raw_ece.append(ece(p_raw[te], y[te])); cal_ece.append(ece(p_cal, y[te]))
        raw_brier.append(float(np.mean((p_raw[te] - y[te]) ** 2)))
        cal_brier.append(float(np.mean((p_cal - y[te]) ** 2)))
    return {"ece_raw": np.mean(raw_ece), "ece_cal": np.mean(cal_ece),
            "brier_raw": np.mean(raw_brier), "brier_cal": np.mean(cal_brier),
            "interval_coverage": np.nanmean(cal_cov)}
