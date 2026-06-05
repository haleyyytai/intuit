"""Lending economics + synthetic loan data with KNOWN default-timing structure.

Implements the challenge's NPV cashflow model exactly, and a data-generating
process where time-to-default depends on applicant traits AND loan terms (with the
lender setting terms partly from traits -> confounding, which the causal layer must
handle). Used to validate the survival model, the E[NPV] decision rule, and the
counterfactual estimator before the real data arrives.

NPV (per the spec):
  repaid  (y=0): NPV = F + R * r * T/365
  default (y=1 at day t): NPV = F + D*(t-1) + rec - R
where
  R   = loan amount (principal)
  F   = origination fee = fee_rate * R         (fee_rate = 0.03)
  r   = APR = 0.35,  T = 60 days
  D   = daily draw   = R * (1 + r*T/365) / T
  rec = recovery     = recovery_rate * outstanding principal at default
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# --- fixed challenge parameters -------------------------------------------- #
APR = 0.35
TERM_DAYS = 60
FEE_RATE = 0.03


@dataclass(frozen=True)
class LoanTerms:
    apr: float = APR
    term_days: int = TERM_DAYS
    fee_rate: float = FEE_RATE


def daily_draw(R, terms: LoanTerms = LoanTerms()):
    """Fixed daily payment so that T payments repay principal + interest."""
    R = np.asarray(R, float)
    return R * (1 + terms.apr * terms.term_days / 365.0) / terms.term_days


def npv_repaid(R, terms: LoanTerms = LoanTerms()):
    """Profit if the loan is fully repaid: fee + interest."""
    R = np.asarray(R, float)
    return terms.fee_rate * R + R * terms.apr * terms.term_days / 365.0


def npv_default(R, t, recovery_rate: float = 0.30, terms: LoanTerms = LoanTerms()):
    """Profit if the loan defaults on day t (1..T): fee + collected draws + recovery - principal."""
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    D = daily_draw(R, terms)
    collected = D * (t - 1)                      # paid for (t-1) days before defaulting
    outstanding = np.maximum(R - collected, 0.0)
    rec = recovery_rate * outstanding
    return terms.fee_rate * R + collected + rec - R


def expected_npv_from_hazard(R, hazard, recovery_rate: float = 0.30,
                             terms: LoanTerms = LoanTerms()):
    """E[NPV] given a per-day hazard vector h_1..h_T (the survival model's output).

    E[NPV] = S_T * NPV_repaid + sum_t P(default day t) * NPV_default(t)
    with S_t = prod_{k<=t}(1-h_k), P(default day t) = S_{t-1} * h_t.
    `R` scalar, `hazard` shape (T,) or (n, T) for a batch.
    """
    hazard = np.atleast_2d(np.asarray(hazard, float))            # (n, T)
    R = np.atleast_1d(np.asarray(R, float))                      # (n,)
    n, T = hazard.shape
    surv_prev = np.ones(n)                                       # S_{t-1}
    exp_npv = np.zeros(n)
    days = np.arange(1, T + 1)
    for j, t in enumerate(days):
        p_def_t = surv_prev * hazard[:, j]                       # P(default exactly day t)
        exp_npv += p_def_t * npv_default(R, t, recovery_rate, terms)
        surv_prev = surv_prev * (1 - hazard[:, j])
    exp_npv += surv_prev * npv_repaid(R, terms)                  # survived all T days = repaid
    return exp_npv if len(exp_npv) > 1 else float(exp_npv[0])


# --------------------------------------------------------------------------- #
# Synthetic loan portfolio with known causal/timing structure
# --------------------------------------------------------------------------- #
def generate_portfolio(n: int = 5000, seed: int = 0, terms: LoanTerms = LoanTerms()):
    """Generate applicants, lender-set loan terms (confounded by traits), and the
    realized default DAY via a discrete-time hazard process.

    Returns a DataFrame with traits, loan_amount, default day `t` (TERM+1 = repaid),
    binary `y` (default within term), and the realized NPV.
    """
    rng = np.random.RandomState(seed)
    # --- applicant traits (pre-treatment covariates) ---
    credit = rng.normal(0.0, 1.0, n)                              # standardized credit quality
    biz_age = rng.gamma(2.0, 2.0, n)                              # years in business
    revenue = np.exp(rng.normal(11.0, 0.8, n))                   # annual revenue ($)
    industry = rng.randint(0, 5, n)                               # 5 industries
    ind_risk = np.array([0.0, 0.3, -0.2, 0.5, -0.1])[industry]   # industry baseline risk

    # --- lender sets loan amount partly FROM traits (this is the confounding) ---
    base_amt = 5000 + 0.04 * revenue + 1500 * credit + 300 * biz_age
    loan_amount = np.clip(base_amt + rng.normal(0, 1500, n), 1000, 60000)

    # leverage = how big the loan is relative to revenue (a key default driver)
    leverage = loan_amount / (revenue + 1e-6)

    # --- discrete-time hazard: worse credit / higher leverage / risky industry -> higher daily hazard ---
    # intercept/scales tuned so the term default rate is realistic (~10-20%), not degenerate
    lin = (-6.5 - 0.9 * credit + 1.2 * leverage + 0.8 * ind_risk - 0.05 * biz_age)
    base_h = 1.0 / (1.0 + np.exp(-lin))                          # per-day hazard ~ small
    base_h = np.clip(base_h, 1e-5, 0.05)

    # realize default day via sequential Bernoulli over the term
    t = np.full(n, terms.term_days + 1)                          # TERM+1 == survived (repaid)
    U = rng.uniform(size=(n, terms.term_days))
    fired = U < base_h[:, None]                                  # constant hazard per applicant
    has_def = fired.any(axis=1)
    t[has_def] = fired.argmax(axis=1)[has_def] + 1               # first day hazard fires (1-indexed)
    y = (t <= terms.term_days).astype(int)

    # realized NPV
    npv = np.where(y == 0,
                   npv_repaid(loan_amount, terms),
                   npv_default(loan_amount, t, terms=terms))

    return pd.DataFrame({
        "credit": credit, "biz_age": biz_age, "revenue": revenue,
        "industry": industry, "leverage": leverage,
        "loan_amount": loan_amount, "t": t, "y": y, "npv": npv,
        "_true_hazard": base_h,                                   # ground truth (for validation only)
    })


FEATURES = ["credit", "biz_age", "revenue", "industry", "leverage", "loan_amount"]
