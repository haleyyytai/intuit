"""Gate 1 — data audit. Produces the concrete numbers behind Deliverable D Section 1
(selection on observables, positivity, missingness, timing). Saves tables to
<challenge>/audit/ and prints a summary.

Run:  python src/audit.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import (build_features, feature_columns,  # noqa: E402
                                DATA, CH)
AUD = CH / "audit"


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA / "train.csv")
    va = pd.read_csv(DATA / "validation.csv")
    te = pd.read_csv(DATA / "test.csv")
    feat_cols = feature_columns(tr)
    miss_cols = [c for c in feat_cols if tr[c].isna().mean() > 0.05]

    # 1. label availability
    lab = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [len(tr), len(va), len(te)],
        "labeled": [tr.default_flag.notna().sum(), va.default_flag.notna().sum(),
                    te.default_flag.notna().sum()],
    })
    lab.to_csv(AUD / "label_availability.csv", index=False)
    print("[1] label availability\n", lab.to_string(index=False))

    # 2. selection gap: standardized mean difference approved vs declined
    appr = tr["prior_decision"] == 1
    num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].dtype != bool]
    smd = []
    for c in num:
        a, d = tr.loc[appr, c], tr.loc[~appr, c]
        sd = np.sqrt((a.var() + d.var()) / 2)
        if sd and np.isfinite(sd):
            smd.append((c, (a.mean() - d.mean()) / sd))
    smd = pd.DataFrame(smd, columns=["feature", "smd"]).sort_values("smd", key=abs, ascending=False)
    smd.to_csv(AUD / "selection_smd.csv", index=False)
    print("\n[2] top selection gaps (|SMD|):\n", smd.head(6).to_string(index=False))

    # 3. approval predictability (positivity risk, NOT ignorability)
    X = build_features(tr, feat_cols, miss_cols)
    y_appr = appr.astype(int).to_numpy()
    e_hat = cross_val_predict(
        HistGradientBoostingClassifier(max_depth=3, max_iter=250, random_state=0),
        X, y_appr, cv=3, method="predict_proba")[:, 1]
    auc = roc_auc_score(y_appr, e_hat)
    print(f"\n[3] approval-propensity AUC = {auc:.3f}  (high -> strong positivity risk)")

    # 4. positivity on the decision set
    dec = pd.concat([va, te], ignore_index=True)
    e_dec = HistGradientBoostingClassifier(max_depth=3, max_iter=250, random_state=0)\
        .fit(X, y_appr).predict_proba(build_features(dec, feat_cols, miss_cols))[:, 1]
    pos = pd.DataFrame({"threshold": [0.01, 0.05, 0.1, 0.2],
                        "frac_decision_below": [(e_dec < t).mean() for t in [0.01, 0.05, 0.1, 0.2]]})
    pos.to_csv(AUD / "positivity.csv", index=False)
    print("\n[4] positivity (fraction of decision set with low e(x)):\n", pos.to_string(index=False))

    # 5. partial ignorability probe: does prior_underwriter_score retain signal for Y
    #    among APPROVED labeled rows after the other features?
    labm = tr.default_flag.notna()
    yL = (tr.loc[labm, "default_flag"] == True).astype(int).to_numpy()  # noqa: E712
    XL_full = build_features(tr[labm], feat_cols, miss_cols)
    XL_no = build_features(tr[labm], [c for c in feat_cols if c != "prior_underwriter_score"], miss_cols)
    auc_full = roc_auc_score(yL, cross_val_predict(
        HistGradientBoostingClassifier(max_depth=3, max_iter=250, random_state=0),
        XL_full, yL, cv=3, method="predict_proba")[:, 1])
    auc_no = roc_auc_score(yL, cross_val_predict(
        HistGradientBoostingClassifier(max_depth=3, max_iter=250, random_state=0),
        XL_no, yL, cv=3, method="predict_proba")[:, 1])
    print(f"\n[5] risk AUC with prior_underwriter_score={auc_full:.3f}  without={auc_no:.3f}  "
          f"(gap {auc_full-auc_no:+.3f} => it carries residual signal worth conditioning on)")

    # 6. timing
    d = pd.to_numeric(va.loc[va.default_flag == True, "days_to_default"], errors="coerce")  # noqa: E712
    print(f"\n[6] days_to_default: median={d.median():.0f}  frac>60d={ (d>60).mean():.3f}  max={d.max():.0f}")

    # 7. default rate by prior_underwriter_score decile
    m = tr[labm].copy()
    m["dec"] = pd.qcut(m["prior_underwriter_score"], 10, labels=False, duplicates="drop")
    by = m.groupby("dec")["default_flag"].apply(lambda s: (s == True).mean())  # noqa: E712
    by.to_csv(AUD / "default_by_score_decile.csv")
    print("\n[7] default rate by prior_underwriter_score decile (monotone?):")
    print("   ", [round(float(v), 3) for v in by.values])
    print(f"\nsaved audit tables to {AUD}")


if __name__ == "__main__":
    main()
