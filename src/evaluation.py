"""Metrics and calibration diagnostics.

Point metrics + proper scoring rules for intervals (pinball / interval score),
plus calibration tables. The official competition metric is wired into
`competition_metric` once the rules are read; until then it defaults to RMSE so
the pipeline is runnable end-to-end on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    return float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(y_pred - y_true) / denom)))


# --------------------------------------------------------------------------- #
# Proper scoring rules for probabilistic / interval forecasts
# --------------------------------------------------------------------------- #
def pinball_loss(y_true, y_pred_q, q: float) -> float:
    """Quantile (pinball) loss at level q. Lower is better."""
    y_true, y_pred_q = np.asarray(y_true, float), np.asarray(y_pred_q, float)
    diff = y_true - y_pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_score(y_true, lower, upper, alpha: float = 0.1) -> float:
    """Winkler/interval score for a central (1-alpha) interval. Lower is better.

    Penalizes width plus 2/alpha * miss distance — the standard proper score that
    rewards *calibrated* intervals rather than just narrow ones.
    """
    y_true = np.asarray(y_true, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    width = upper - lower
    below = (lower - y_true) * (y_true < lower)
    above = (y_true - upper) * (y_true > upper)
    return float(np.mean(width + (2 / alpha) * (below + above)))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_table(y_true, lower, upper, target_coverage: float) -> dict:
    """One row of the calibration table consumed by the report."""
    from uncertainty import coverage, mean_width  # local import keeps modules decoupled

    return {
        "target_coverage": target_coverage,
        "empirical_coverage": coverage(y_true, lower, upper),
        "mean_width": mean_width(lower, upper),
    }


def coverage_curve(y_true, quantile_preds: dict) -> pd.DataFrame:
    """Given {q: predictions}, return nominal-vs-empirical coverage per quantile.

    For a one-sided check: empirical = P(y <= pred_q) should track q.
    """
    y_true = np.asarray(y_true, float)
    rows = []
    for q, pred in sorted(quantile_preds.items()):
        emp = float(np.mean(y_true <= np.asarray(pred, float)))
        rows.append({"nominal_q": q, "empirical": emp, "gap": emp - q})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Official metric hook
# --------------------------------------------------------------------------- #
def competition_metric(y_true, y_pred):
    """Placeholder for the official metric — set after reading the rules.

    Defaults to RMSE so the end-to-end pipeline runs before rules are known.
    """
    return rmse(y_true, y_pred)
