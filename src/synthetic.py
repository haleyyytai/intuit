"""Synthetic causal time-series generator — a known-ground-truth mini-challenge.

Used to rehearse the full pipeline before the real data exists, and to validate
that causal recovery / intervention forecasting return the TRUE answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ground-truth lagged structure (your §16 design):
#   x1_t = 0.7 x1_{t-1}                              + e
#   x2_t = 0.5 x2_{t-1} + 0.8 x1_{t-1}               + e   (x1 -> x2)
#   x3_t = 0.4 x3_{t-1} - 0.6 x4_{t-2}               + e   (x4 -> x3, lag 2)
#   x4_t = 0.6 x4_{t-1}                              + e
#   y_t  = 0.6 y_{t-1} + 1.2 x2_{t-1} - 0.5 x3_{t-2} + e   (x2 -> y, x3 -> y lag2)
TRUE_EDGES = [
    ("x1", "x2", 1, +1), ("x4", "x3", 2, -1),
    ("x2", "y", 1, +1), ("x3", "y", 2, -1),
]


def generate(n: int = 2000, noise: float = 0.3, seed: int = 0,
             intervention: dict | None = None) -> pd.DataFrame:
    """Generate the system. `intervention` like {'x1': +1.0} adds a persistent shift."""
    rng = np.random.RandomState(seed)
    e = rng.normal(0, noise, size=(n, 5))
    x1 = np.zeros(n); x2 = np.zeros(n); x3 = np.zeros(n); x4 = np.zeros(n); y = np.zeros(n)
    iv = intervention or {}
    for t in range(2, n):
        x1[t] = 0.7 * x1[t-1] + e[t, 0] + iv.get("x1", 0.0)
        x4[t] = 0.6 * x4[t-1] + e[t, 3] + iv.get("x4", 0.0)
        x2[t] = 0.5 * x2[t-1] + 0.8 * x1[t-1] + e[t, 1] + iv.get("x2", 0.0)
        x3[t] = 0.4 * x3[t-1] - 0.6 * x4[t-2] + e[t, 2] + iv.get("x3", 0.0)
        y[t]  = 0.6 * y[t-1] + 1.2 * x2[t-1] - 0.5 * x3[t-2] + e[t, 4]
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y}).iloc[2:].reset_index(drop=True)


def true_effect_on_y(intervention: dict, n: int = 4000, noise: float = 0.3, seed: int = 0) -> float:
    """Monte-Carlo ground-truth average effect on y of a persistent intervention."""
    base = generate(n, noise, seed)["y"].mean()
    do = generate(n, noise, seed, intervention=intervention)["y"].mean()
    return float(do - base)
