"""Discrete-time hazard (survival) model — Deliverable B (time-to-default).

Why discrete-time hazard rather than a binary default classifier: a day-5 default
and a day-55 default both have y=1 but very different cashflows. Modelling the
per-day hazard h_t(x)=P(T=t | T>=t, x) gives the full time-to-default distribution,
hence survival S(t|x), cumulative default rate CDR by week, AND the inputs the
E[NPV] decision rule needs.

Implementation: person-period expansion + a standard classifier on (features, day).
This is the textbook discrete-time survival trick — no special library needed, and
it directly yields a calibrated hazard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class DiscreteTimeHazard:
    """Fit P(default on day t | survived to t, x) over the loan term."""

    def __init__(self, term_days: int = 60, model: str = "gbm", random_state: int = 0):
        self.term_days = term_days
        self.model = model
        self.random_state = random_state

    # --- person-period expansion ------------------------------------------- #
    def _expand(self, df, features, time_col, event_col):
        """One row per (loan, day) up to its default day (or full term if censored)."""
        rows_X, rows_d, rows_e = [], [], []
        Xf = df[features].to_numpy(float)
        t = df[time_col].to_numpy(int)
        y = df[event_col].to_numpy(int)
        for i in range(len(df)):
            last = min(t[i], self.term_days)        # observed up to default or term end
            for day in range(1, last + 1):
                rows_X.append(Xf[i])
                rows_d.append(day)
                rows_e.append(1 if (y[i] == 1 and day == t[i]) else 0)
        X = np.column_stack([np.asarray(rows_X, float), np.asarray(rows_d, float)])
        return X, np.asarray(rows_e, int)

    def fit(self, df, features, time_col="t", event_col="y"):
        self.features_ = list(features)
        X, e = self._expand(df, self.features_, time_col, event_col)
        if self.model == "linear":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=self.random_state))
        else:
            clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300,
                                                 learning_rate=0.08, l2_regularization=1.0,
                                                 random_state=self.random_state)
        clf.fit(X, e)
        self.clf_ = clf
        return self

    # --- prediction -------------------------------------------------------- #
    def predict_hazard(self, X_df) -> np.ndarray:
        """Return (n, T) per-day hazard matrix."""
        Xf = np.asarray(X_df[self.features_], float)
        n = len(Xf)
        days = np.arange(1, self.term_days + 1)
        H = np.empty((n, self.term_days))
        for j, day in enumerate(days):
            Xd = np.column_stack([Xf, np.full(n, day, float)])
            H[:, j] = self.clf_.predict_proba(Xd)[:, 1]
        return np.clip(H, 1e-6, 1 - 1e-6)

    def predict_survival(self, X_df) -> np.ndarray:
        """Return (n, T) survival S_t = prod_{k<=t}(1-h_k)."""
        return np.cumprod(1 - self.predict_hazard(X_df), axis=1)

    def predict_default_prob(self, X_df) -> np.ndarray:
        """P(default within the term) = 1 - S_T  — Deliverable A point estimate."""
        return 1.0 - self.predict_survival(X_df)[:, -1]

    def predict_cdr(self, X_df, weeks=(1, 2, 4, 8)) -> pd.DataFrame:
        """Cumulative default rate by week a: P(T <= 7a) — Deliverable B curve."""
        S = self.predict_survival(X_df)
        out = {}
        for a in weeks:
            day = min(7 * a, self.term_days)
            out[f"cdr_w{a}"] = 1.0 - S[:, day - 1]
        return pd.DataFrame(out, index=getattr(X_df, "index", None))
