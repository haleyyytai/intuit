"""Calibrated uncertainty quantification.

Design goals:
  * distribution-free coverage via split conformal (no library needed)
  * quantile regression as an ensemble alternative (sklearn HistGBM quantile loss)
  * OOD-aware inflation: intervals widen when an input (esp. an intervention) falls
    outside the empirical support of the training distribution.

Everything here is data-agnostic: it operates on arrays / fitted estimators, so it
is testable before the challenge data exists. Narrow intervals are NOT the goal —
believable coverage is. Always report coverage AND width.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #
@dataclass
class ConformalModel:
    """Wraps a fitted point estimator with split-conformal residual intervals.

    Fit the base model on the training split, then call `.calibrate` on a held-out
    calibration split. Intervals carry a finite-sample (1 - alpha) coverage
    guarantee under exchangeability.
    """

    model: object
    alpha: float = 0.1
    q_: float | None = None  # conformal residual quantile

    def calibrate(self, X_calib, y_calib) -> "ConformalModel":
        preds = np.asarray(self.model.predict(X_calib), dtype=float)
        resid = np.abs(np.asarray(y_calib, dtype=float) - preds)
        n = len(resid)
        # finite-sample correction: ceil((n+1)(1-alpha)) / n quantile
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_ = float(np.quantile(resid, level, method="higher"))
        return self

    def predict_interval(self, X, scale: np.ndarray | None = None):
        """Return (point, lower, upper). `scale` (>=1 per row) inflates width OOD."""
        if self.q_ is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        point = np.asarray(self.model.predict(X), dtype=float)
        half = np.full(len(point), self.q_, dtype=float)
        if scale is not None:
            half = half * np.asarray(scale, dtype=float)
        return point, point - half, point + half


def conformal_intervals(model, X_calib, y_calib, X_test, alpha: float = 0.1,
                        scale: np.ndarray | None = None):
    """Convenience one-shot: calibrate `model` and return (point, lower, upper)."""
    cm = ConformalModel(model=model, alpha=alpha).calibrate(X_calib, y_calib)
    return cm.predict_interval(X_test, scale=scale)


# --------------------------------------------------------------------------- #
# Quantile regression intervals (ensemble alternative)
# --------------------------------------------------------------------------- #
def quantile_intervals(X_train, y_train, X_test, quantiles=(0.05, 0.5, 0.95),
                       **gbm_kwargs):
    """Fit one HistGBM per quantile; return dict {q: prediction array}.

    Note: pinball-optimal but NOT coverage-guaranteed and can cross; conformal is
    the primary. Use this for adaptive (heteroscedastic) width or as an ensemble.
    """
    out = {}
    for q in quantiles:
        gbm = HistGradientBoostingRegressor(loss="quantile", quantile=q, **gbm_kwargs)
        gbm.fit(X_train, y_train)
        out[q] = np.asarray(gbm.predict(X_test), dtype=float)
    return out


# --------------------------------------------------------------------------- #
# OOD / intervention support scoring  -> uncertainty inflation
# --------------------------------------------------------------------------- #
class SupportScorer:
    """Quantifies how far a row sits from the training support.

    Combines a robust Mahalanobis distance and an IsolationForest anomaly score
    into a per-row inflation factor >= 1. Interventions pushed outside observed
    support get wider intervals — the "causal humility" guardrail.
    """

    def __init__(self, robust: bool = True, random_state: int = 0):
        self.robust = robust
        self.random_state = random_state

    def fit(self, X) -> "SupportScorer":
        X = np.asarray(X, dtype=float)
        cov = MinCovDet(random_state=self.random_state) if self.robust else EmpiricalCovariance()
        try:
            self.cov_ = cov.fit(X)
        except Exception:  # MinCovDet can fail on tiny/degenerate data
            self.cov_ = EmpiricalCovariance().fit(X)
        self._iso = IsolationForest(random_state=self.random_state, n_estimators=200).fit(X)
        # reference distribution of Mahalanobis distances on training data
        self._train_maha = np.sqrt(np.maximum(self.cov_.mahalanobis(X), 0.0))
        self._maha_ref = np.quantile(self._train_maha, 0.95)
        self._dim = X.shape[1]
        return self

    def mahalanobis(self, X) -> np.ndarray:
        return np.sqrt(np.maximum(self.cov_.mahalanobis(np.asarray(X, dtype=float)), 0.0))

    def support_score(self, X) -> np.ndarray:
        """In [0, 1]-ish: 0 = squarely in support, larger = more OOD."""
        X = np.asarray(X, dtype=float)
        maha = self.mahalanobis(X)
        maha_norm = maha / (self._maha_ref + 1e-9)
        # IsolationForest: higher score_samples = more normal; flip & shift to [0,1]
        iso = -self._iso.score_samples(X)
        iso_norm = (iso - iso.min()) / (np.ptp(iso) + 1e-9) if len(iso) > 1 else np.zeros_like(iso)
        return np.maximum(maha_norm - 1.0, 0.0) + iso_norm

    def inflation_factor(self, X, k: float = 0.5, cap: float = 4.0) -> np.ndarray:
        """Map support score -> interval multiplier in [1, cap]."""
        s = self.support_score(X)
        return np.clip(1.0 + k * s, 1.0, cap)


def percentile_rank_violation(X_train, X_query) -> np.ndarray:
    """Per-row fraction of features whose value is outside the train min/max range.

    Cheap, interpretable OOD flag for the intervention sanity-check table.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    out = (X_query < lo) | (X_query > hi)
    return out.mean(axis=1)


# --------------------------------------------------------------------------- #
# Weighted (covariate-shift-robust) conformal  —  Tibshirani et al. 2019,
# Lei & Candes 2021. The principled version of "OOD inflation": instead of
# multiplying the half-width by a heuristic factor, we REWEIGHT the calibration
# residuals by the density ratio w(x) = dP_test / dP_calib and take a weighted
# quantile. Under weighted exchangeability this restores the coverage guarantee
# when the test/intervention distribution differs from calibration.
# --------------------------------------------------------------------------- #
def density_ratio_weights(X_calib, X_query, clip: float = 50.0, random_state: int = 0,
                          classifier: str = "auto"):
    """Estimate w(x) = dP_query/dP_calib via a probabilistic classifier (calib vs query).

    Classifier-based density-ratio estimation: train c(x)=P(query|x); then
    w(x) = c(x)/(1-c(x)) * (n_calib/n_query). Cross-fitted (out-of-fold probs) so
    weights aren't optimistic. A linear classifier misses nonlinear shifts (e.g. a
    tail/variance shift), so the default is a small gradient-boosted classifier.
    Returns (w_calib, w_query) arrays.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    Xc, Xq = np.asarray(X_calib, float), np.asarray(X_query, float)
    nc, nq = len(Xc), len(Xq)
    X = np.vstack([Xc, Xq])
    y = np.concatenate([np.zeros(nc), np.ones(nq)])
    if classifier == "linear":
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, random_state=random_state))
    else:  # 'auto' / nonlinear: captures tail & variance shifts a linear model can't
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                             learning_rate=0.1, l2_regularization=1.0,
                                             random_state=random_state)
    # out-of-fold probabilities avoid over-fit (over-confident) weights
    p = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
    p = np.clip(p, 1e-4, 1 - 1e-4)
    w = (p / (1 - p)) * (nc / max(1, nq))
    w = np.clip(w, 1.0 / clip, clip)
    return w[:nc], w[nc:]


def _weighted_conformal_q(scores, w_calib, w_test, alpha: float) -> float:
    """Smallest calibration score whose cumulative *normalized* weight >= 1-alpha.

    The test point holds mass w_test/(sum w_calib + w_test) at +inf; if the
    calibration mass cannot reach 1-alpha, the quantile is +inf (maximally OOD ->
    unbounded interval), which we cap downstream.
    """
    order = np.argsort(scores)
    s, w = np.asarray(scores)[order], np.asarray(w_calib)[order]
    total = w.sum() + w_test
    cum = np.cumsum(w) / total
    idx = int(np.searchsorted(cum, 1 - alpha, side="left"))
    return float(s[idx]) if idx < len(s) else np.inf


@dataclass
class WeightedConformalModel:
    """Split conformal with covariate-shift weighting. Per-test-point intervals
    widen automatically where the test distribution is under-represented in
    calibration — no hand-tuned inflation factor.
    """

    model: object
    alpha: float = 0.1
    random_state: int = 0

    def calibrate(self, X_calib, y_calib) -> "WeightedConformalModel":
        self.Xc_ = np.asarray(X_calib, float)
        preds = np.asarray(self.model.predict(X_calib), float)
        self.scores_ = np.abs(np.asarray(y_calib, float) - preds)
        return self

    def predict_interval(self, X_test, cap_width: float | None = None):
        Xt = np.asarray(X_test, float)
        w_calib, w_test = density_ratio_weights(self.Xc_, Xt, random_state=self.random_state)
        point = np.asarray(self.model.predict(Xt), float)
        half = np.array([_weighted_conformal_q(self.scores_, w_calib, wt, self.alpha)
                         for wt in w_test])
        if cap_width is None:
            cap_width = 5.0 * float(np.quantile(self.scores_, 0.99))
        half = np.minimum(half, cap_width)
        return point, point - half, point + half


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def coverage(y_true, lower, upper) -> float:
    """Empirical coverage = fraction of y_true within [lower, upper]."""
    y_true, lower, upper = map(lambda a: np.asarray(a, dtype=float), (y_true, lower, upper))
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))
