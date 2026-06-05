"""Forecasting models: baselines through final forecaster.

Implemented on the stock scientific-Python stack (sklearn / statsmodels) so the
pipeline runs with no fragile dependencies. LightGBM/CatBoost can be swapped in as
optional accelerators later; HistGradientBoostingRegressor is a LightGBM-class
booster and is the default GBM here.

All point models expose fit(X, y) / predict(X) so they are interchangeable.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NaiveBaseline:
    """Train-mean predictor (or persistence when used on lag-1 features)."""

    def fit(self, X, y):
        self._mean = float(np.asarray(y, float).mean())
        return self

    def predict(self, X):
        return np.full(len(X), self._mean)


def fit_linear(X, y, random_state: int = 0):
    """Standardized ElasticNet with CV-selected penalty — interpretable baseline."""
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, random_state=random_state,
                     max_iter=5000),
    )
    model.fit(X, y)
    return model


def fit_gbm(X, y, random_state: int = 0, **kwargs):
    """HistGradientBoosting point forecaster (LightGBM-class)."""
    params = dict(learning_rate=0.05, max_iter=400, max_depth=None,
                  l2_regularization=1.0, early_stopping=True,
                  validation_fraction=0.15, random_state=random_state)
    params.update(kwargs)
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    return model


def fit_var(df, maxlags: int = 5):
    """Reduced-form VAR for temporal data (statsmodels). Returns fitted results."""
    from statsmodels.tsa.api import VAR

    res = VAR(np.asarray(df, float)).fit(maxlags=maxlags, ic="aic")
    return res


class BootstrapEnsemble:
    """Bag of a base estimator over bootstrap resamples.

    Gives a model-disagreement signal (predict_std) that feeds adaptive
    uncertainty, complementing conformal's marginal guarantee.
    """

    def __init__(self, fit_fn=fit_gbm, n_estimators: int = 20, random_state: int = 0):
        self.fit_fn = fit_fn
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        rng = np.random.RandomState(self.random_state)
        self.models_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n)  # bootstrap resample
            self.models_.append(self.fit_fn(X[idx], y[idx]))
        return self

    def _stack(self, X):
        return np.column_stack([m.predict(X) for m in self.models_])

    def predict(self, X):
        return self._stack(X).mean(axis=1)

    def predict_std(self, X):
        return self._stack(X).std(axis=1)
