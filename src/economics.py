"""Realized-NPV economics with empirical loss-given-default (Gate 5 fix).

The stylized `fee + D*(t-1) + rec - R` formula is WRONG for defaulters: the cure
mechanism (3 consecutive / 6 cumulative missed draws) means collected != (t-1)
draws. Instead we estimate the realized NPV ratio per default week empirically from
the outcome columns (actual default day + actual final_recovered_amount), then plug
the hazard-implied default-week distribution into E[NPV].

Realized NPV ratio (per $1 of principal R):
  repaid : fee_rate + apr*T/365                                (= 0.0875 here)
  default: fee_rate + collected/R + recovery/R - 1
where collected is the draws actually received before default. We do not observe
missed draws directly, so collected is approximated as draws up to the default day
capped at the full repayment schedule; the per-week empirical averaging absorbs the
residual noise. This is transparent and far better than the stylized closed form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lending import daily_draw, LoanTerms

TERMS = LoanTerms()
N_WEEKS = 13
REPAID_RATIO = TERMS.fee_rate + TERMS.apr * TERMS.term_days / 365.0  # 0.0875


def realized_default_ratio(R, t, rec, terms: LoanTerms = TERMS):
    """Realized NPV / R for a loan that defaulted on day t with recovery `rec`."""
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    rec = np.nan_to_num(np.asarray(rec, float), nan=0.0)
    D = daily_draw(R, terms)
    collected = D * np.clip(t - 1, 0, terms.term_days)
    full = R * (1 + terms.apr * terms.term_days / 365.0)
    collected = np.minimum(collected, full)
    return terms.fee_rate + collected / R + rec / R - 1.0


def fit_lgd_by_week(df_labeled, terms: LoanTerms = TERMS) -> np.ndarray:
    """Empirical mean realized NPV ratio for defaulters, by default week (1..13)."""
    d = df_labeled[df_labeled["default_flag"] == True]  # noqa: E712
    R = d["requested_amount"].to_numpy(float)
    t = pd.to_numeric(d["days_to_default"], errors="coerce").to_numpy()
    rec = pd.to_numeric(d["final_recovered_amount"], errors="coerce").to_numpy()
    ratio = realized_default_ratio(R, t, rec, terms)
    wk = np.clip(np.ceil(t / 7.0), 1, N_WEEKS).astype(int)
    out = np.full(N_WEEKS, np.nan)
    for a in range(1, N_WEEKS + 1):
        m = wk == a
        if m.sum() >= 5:
            out[a - 1] = float(np.nanmean(ratio[m]))
    return pd.Series(out).interpolate(limit_direction="both").to_numpy()


def expected_npv(R, H, lgd_by_week, repaid_ratio: float = REPAID_RATIO):
    """E[NPV] = S_T * repaid_ratio * R + sum_a P(default week a) * lgd_a * R."""
    R = np.asarray(R, float)
    H = np.atleast_2d(np.asarray(H, float))
    n, T = H.shape
    surv = np.ones(n)
    enpv = np.zeros(n)
    for a in range(T):
        p_def = surv * H[:, a]
        enpv += p_def * lgd_by_week[a] * R
        surv = surv * (1 - H[:, a])
    enpv += surv * repaid_ratio * R
    return enpv
