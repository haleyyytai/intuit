"""Real-data interventional sanity check: take actual validation applicants, sweep one
feature across its observed range (recomputing deterministic descendants), and confirm the
C-model's predicted PD moves in the economically sensible direction and monotonically.

We cannot score these against ground truth (counterfactuals are unobservable on real data),
so this checks RESPONSE PLAUSIBILITY, not accuracy. Accuracy lives in c_validation.py.

Run:  python src/perturb_sanity.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import build_features, feature_columns, DATA, recompute_descendants, DESCENDANT_TRIGGERS  # noqa: E402

# feature -> expected direction of PD as the feature INCREASES (+1 risk up, -1 risk down)
EXPECT = {
    "invoice_payment_delinquency_rate": +1,
    "aggregate_credit_utilization": +1,
    "observed_revenue_volatility": +1,
    "observed_overdraft_count_3mo": +1,
    "requested_amount": +1,
    "observed_revenue_trend_3mo": -1,
    "observed_cash_balance_p10": -1,
    "stated_time_in_business": -1,
}


def main():
    tr = pd.read_csv(DATA / "train.csv"); va = pd.read_csv(DATA / "validation.csv")
    feat = feature_columns(tr)
    featC = [c for c in feat if c != "prior_underwriter_score"]      # C feature set
    miss = [c for c in featC if tr[c].isna().mean() > 0.05]
    lab = tr[tr.default_flag.notna()]
    M = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                       l2_regularization=1.0, random_state=0).fit(
        build_features(lab, featC, miss), (lab.default_flag == True).astype(int))  # noqa: E712

    base = va.sample(1500, random_state=1).reset_index(drop=True)   # real applicants
    print(f"swept on {len(base)} real validation applicants; baseline mean PD = "
          f"{M.predict_proba(build_features(base, featC, miss))[:,1].mean():.3f}\n")
    print(f"{'feature':34s} {'p10->p90 PD':>16s}  exp  got  ok")
    n_ok = 0
    for f, exp in EXPECT.items():
        col = pd.to_numeric(tr[f], errors="coerce")
        lo, hi = np.nanpercentile(col, 10), np.nanpercentile(col, 90)
        pds = []
        for v in [lo, hi]:
            cf = base.copy(); cf[f] = v
            if f in DESCENDANT_TRIGGERS:
                cf = recompute_descendants(cf, np.ones(len(cf), bool))
            pds.append(M.predict_proba(build_features(cf, featC, miss))[:, 1].mean())
        got = int(np.sign(pds[1] - pds[0])) or 1
        ok = (got == exp)
        n_ok += ok
        print(f"{f:34s} {pds[0]:.3f} -> {pds[1]:.3f}   {('+' if exp>0 else '-')}    "
              f"{('+' if got>0 else '-')}   {'OK' if ok else 'XX'}")
    print(f"\ndirection sanity: {n_ok}/{len(EXPECT)} features move the economically expected way")


if __name__ == "__main__":
    main()
