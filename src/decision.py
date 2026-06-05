"""Decision layer — approve to maximize portfolio NPV, not to minimize default.

The challenge's first trap: a naive policy approves when default prob < tau. But a
$100 loan at 5% default can be worth less than a $50k loan at 8% default. The
correct policy approves when EXPECTED PROFIT is positive:

    approve_i = 1[ E[NPV_i | x_i] > 0 ]

E[NPV] is computed from the survival model's hazard curve and the loan amount, so
the decision layer consumes Deliverable B directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lending import expected_npv_from_hazard, LoanTerms


def expected_npv(loan_amount, hazard, recovery_rate: float = 0.30,
                 terms: LoanTerms = LoanTerms()) -> np.ndarray:
    """Per-applicant E[NPV] from predicted per-day hazards (n, T) and loan amounts (n,)."""
    R = np.asarray(loan_amount, float)
    H = np.atleast_2d(np.asarray(hazard, float))
    out = np.array([expected_npv_from_hazard(R[i], H[i], recovery_rate, terms)
                    for i in range(len(R))])
    return out


def policy_threshold(default_prob, tau: float) -> np.ndarray:
    """Naive policy: approve if predicted default probability < tau."""
    return (np.asarray(default_prob, float) < tau).astype(int)


def policy_expected_npv(exp_npv, min_npv: float = 0.0) -> np.ndarray:
    """Profit-aware policy: approve if expected NPV exceeds a floor (default 0)."""
    return (np.asarray(exp_npv, float) > min_npv).astype(int)


def portfolio_profit(realized_npv, approve_mask) -> float:
    """Total realized NPV over approved loans (the business objective)."""
    return float(np.sum(np.asarray(realized_npv, float) * np.asarray(approve_mask, int)))


def evaluate_policies(df, exp_npv, default_prob, taus=(0.05, 0.10, 0.15, 0.20, 0.30),
                      realized_col="npv") -> pd.DataFrame:
    """Compare the E[NPV] policy against a sweep of threshold policies + approve-all."""
    realized = df[realized_col].to_numpy(float)
    rows = []
    # approve everyone
    m = np.ones(len(df), int)
    rows.append({"policy": "approve_all", "approved": int(m.sum()),
                 "portfolio_profit": portfolio_profit(realized, m),
                 "approval_rate": float(m.mean())})
    # threshold sweep
    for tau in taus:
        m = policy_threshold(default_prob, tau)
        rows.append({"policy": f"threshold p<{tau}", "approved": int(m.sum()),
                     "portfolio_profit": portfolio_profit(realized, m),
                     "approval_rate": float(m.mean())})
    # E[NPV] policy
    m = policy_expected_npv(exp_npv)
    rows.append({"policy": "E[NPV]>0", "approved": int(m.sum()),
                 "portfolio_profit": portfolio_profit(realized, m),
                 "approval_rate": float(m.mean())})
    return pd.DataFrame(rows).sort_values("portfolio_profit", ascending=False).reset_index(drop=True)
