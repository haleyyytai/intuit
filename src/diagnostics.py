"""Phase 1/2 diagnostics on labeled validation: per-age trajectory calibration (2B),
A-interval coverage vs width (2C), and the E[NPV] lambda sweep (2A). Analysis only.

Run:  python src/diagnostics.py
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
from challenge_pipeline import (build_features, feature_columns, default_week,  # noqa: E402
                                expand_person_period, WeeklyHazard, DATA, N_WEEKS)
import economics as econ  # noqa: E402
from calibration import fit_isotonic, binned_coverage  # noqa: E402


def main():
    tr = pd.read_csv(DATA / "train.csv"); va = pd.read_csv(DATA / "validation.csv")
    feat = feature_columns(tr); miss = [c for c in feat if tr[c].isna().mean() > 0.05]
    lgd = econ.fit_lgd_by_week(tr[tr.default_flag.notna()])

    a_old = (tr.prior_decision == 1).astype(int).to_numpy()
    Xtr = build_features(tr, feat, miss)
    e = HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=0).fit(Xtr, a_old)
    e_tr = np.clip(e.predict_proba(Xtr)[:, 1], 1e-3, 1 - 1e-3)
    lab = tr.default_flag.notna().to_numpy()
    Xlab = Xtr[lab].reset_index(drop=True)
    dwk = default_week(tr[lab].reset_index(drop=True))
    w = np.clip(a_old.mean() / e_tr[lab], 0.1, 10.0)
    PP, ev, wv = expand_person_period(Xlab, dwk, w)
    rng = np.random.RandomState(0)
    models = [WeeklyHazard(random_state=k).fit(PP.iloc[(i := rng.randint(0, len(PP), len(PP)))].reset_index(drop=True), ev[i], wv[i]) for k in range(5)]

    def curves(X):
        ps, cs, hs = [], [], []
        for m in models:
            p, c, h = m.curves(X.copy()); ps.append(p); cs.append(c); hs.append(h)
        return np.mean(ps, 0), np.std(ps, 0), np.mean(cs, 0), np.mean(hs, 0)

    vlab = va.default_flag.notna()
    Xv = build_features(va[vlab], feat, miss)
    pdv, sdv, cdrv, Hv = curves(Xv)
    yv = (va.loc[vlab, "default_flag"] == True).astype(int).to_numpy()  # noqa: E712
    dv = pd.to_numeric(va.loc[vlab, "days_to_default"], errors="coerce").to_numpy()
    R = va.loc[vlab, "requested_amount"].to_numpy(float)
    iso = fit_isotonic(pdv, yv)
    pd_cal = np.clip(iso.predict(pdv), 0, 1)
    factor = pd_cal / np.clip(pdv, 1e-6, 1)
    cdr_cal = np.clip(cdrv * factor[:, None], 0, 1)
    H_cal = np.clip(Hv * factor[:, None], 1e-6, 1 - 1e-6)

    # ---- 2B per-age trajectory calibration ----
    print("[2B] per-age cumulative default: predicted vs realized (labeled validation)")
    print("  age  pred   real   gap")
    for a in range(1, N_WEEKS + 1):
        pred = cdr_cal[:, a - 1].mean()
        real = float(np.mean((yv == 1) & (dv <= 7 * a)))
        print(f"  {a:2d}   {pred:.3f}  {real:.3f}  {pred-real:+.3f}")

    # ---- 2C A-interval coverage vs width ----
    print("\n[2C] A interval: binned coverage vs mean width (target ~0.90)")
    for mult in [0.5, 1.0, 1.5, 2.0]:
        half = mult * (1.645 * sdv) + 0.02
        cov = binned_coverage(pd_cal, np.clip(pd_cal - half, 0, 1), np.clip(pd_cal + half, 0, 1), yv)
        print(f"  mult={mult:.1f}  coverage={cov:.2f}  mean_width={2*half.mean():.3f}")

    # ---- 2A lambda sweep ----
    print("\n[2A] E[NPV] lambda sweep (realized validation profit, exact formula)")
    realized = np.where(yv == 0, econ.REPAID_RATIO * R, econ.realized_default_ratio(R, dv, pd.to_numeric(va.loc[vlab, "final_recovered_amount"], errors="coerce").to_numpy()) * R)
    enpv = econ.expected_npv(R, H_cal, lgd)
    for lam in [0.0, 0.005, 0.01, 0.02]:
        dec = (enpv > lam * R).astype(int)
        print(f"  lambda={lam:.3f}R  profit=${(realized*dec).sum():>11,.0f}  approve={dec.mean():.2%}")


if __name__ == "__main__":
    main()
