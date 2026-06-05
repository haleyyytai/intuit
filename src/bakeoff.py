"""Task B - controlled backbone bake-off (selection on an INTERNAL train time-split).

Discipline: fixed robust defaults (no grid search); select on calibrated Brier/logloss
on a held-out LATER time slice of train; reserve official validation for final
calibration only. Incumbent = HistGBM/LightGBM weekly hazard; only swap on a decisive
win (>2% relative calibrated Brier, or AUC by >0.01 with no worse calibration), with a
thumb on the scale for the hazard backbone because Deliverable C reuses it.

Run:  python src/bakeoff.py   -> writes challenge/.../audit/bakeoff.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import (build_features, feature_columns, default_week,  # noqa: E402
                                expand_person_period, DATA, CH, N_WEEKS)
import economics as econ  # noqa: E402
from calibration import fit_isotonic, ece  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def brier(p, y): return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": brier(p, y), "logloss": float(log_loss(y, p)),
            "auc": float(roc_auc_score(y, p)), "ece": ece(p, y)}


def main():
    tr = pd.read_csv(DATA / "train.csv")
    feat = feature_columns(tr)
    miss = [c for c in feat if tr[c].isna().mean() > 0.05]
    lgd = econ.fit_lgd_by_week(tr[tr.default_flag.notna()])

    # labeled only, ordered by application time -> early(fit) / cal / late(select)
    lab = tr[tr.default_flag.notna()].copy()
    lab["ts"] = pd.to_datetime(lab["application_timestamp"], errors="coerce")
    lab = lab.sort_values("ts").reset_index(drop=True)
    n = len(lab)
    fit_idx = np.arange(0, int(0.56 * n))
    cal_idx = np.arange(int(0.56 * n), int(0.70 * n))
    late_idx = np.arange(int(0.70 * n), n)
    print(f"time-split: fit={len(fit_idx)} cal={len(cal_idx)} late={len(late_idx)}")

    Xf = build_features(lab.iloc[fit_idx], feat, miss)
    Xc = build_features(lab.iloc[cal_idx], feat, miss)
    Xl = build_features(lab.iloc[late_idx], feat, miss)
    yf = (lab.iloc[fit_idx].default_flag == True).astype(int).to_numpy()  # noqa: E712
    yc = (lab.iloc[cal_idx].default_flag == True).astype(int).to_numpy()  # noqa: E712
    yl = (lab.iloc[late_idx].default_flag == True).astype(int).to_numpy()  # noqa: E712

    rows = []

    # ---- binary PD90 learners ---------------------------------------------- #
    def eval_binary(name, clf, needs_impute=False):
        clf.fit(Xf, yf)
        pc = clf.predict_proba(Xc)[:, 1]
        pl = clf.predict_proba(Xl)[:, 1]
        iso = fit_isotonic(pc, yc)
        m = metrics(np.clip(iso.predict(pl), 0, 1), yl)
        m.update(name=name, kind="binary")
        rows.append(m); print(f"  {name:26s} Brier={m['brier']:.4f} logloss={m['logloss']:.4f} "
                              f"AUC={m['auc']:.3f} ECE={m['ece']:.4f}")

    eval_binary("logistic PD90", make_pipeline(SimpleImputer(strategy="median"),
                StandardScaler(), LogisticRegression(max_iter=2000)))
    eval_binary("randomforest PD90", make_pipeline(SimpleImputer(strategy="median"),
                RandomForestClassifier(n_estimators=300, min_samples_leaf=50, random_state=0)))
    eval_binary("lightgbm-binary PD90", LGBMClassifier(
        n_estimators=600, num_leaves=47, learning_rate=0.04, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1))

    # ---- weekly-hazard learners -------------------------------------------- #
    dwk = default_week(lab.iloc[fit_idx].reset_index(drop=True))
    PP, ev, wv = expand_person_period(Xf, dwk, np.ones(len(Xf)))
    print(f"person-period (fit): {len(PP)} rows")

    def hazard_pd90(clf, X):
        n = len(X); H = np.empty((n, N_WEEKS))
        for a in range(1, N_WEEKS + 1):
            Xa = X.copy(); Xa["age_week"] = a
            H[:, a - 1] = clf.predict_proba(Xa[PP.columns])[:, 1]
        H = np.clip(H, 1e-6, 1 - 1e-6)
        return 1 - np.cumprod(1 - H, axis=1)[:, -1], H

    def eval_hazard(name, clf):
        clf.fit(PP, ev)
        pdc, _ = hazard_pd90(clf, Xc)
        pdl, Hl = hazard_pd90(clf, Xl)
        iso = fit_isotonic(pdc, yc)
        pcal = np.clip(iso.predict(pdl), 0, 1)
        m = metrics(pcal, yl); m.update(name=name, kind="hazard")
        # policy NPV on late (hazard has native timing)
        R = lab.iloc[late_idx].requested_amount.to_numpy(float)
        t = pd.to_numeric(lab.iloc[late_idx].days_to_default, errors="coerce").to_numpy()
        realized = np.where(yl == 0, econ.REPAID_RATIO * R, econ.realized_default_ratio(R, t, None) * R)
        enpv = econ.expected_npv(R, Hl, lgd)
        m["policy_npv"] = float((realized * (enpv > 0)).sum())
        rows.append(m); print(f"  {name:26s} Brier={m['brier']:.4f} logloss={m['logloss']:.4f} "
                              f"AUC={m['auc']:.3f} ECE={m['ece']:.4f} policyNPV={m['policy_npv']:,.0f}")

    eval_hazard("histgbm hazard (incumbent)", HistGradientBoostingClassifier(
        max_depth=3, max_iter=350, learning_rate=0.07, l2_regularization=1.0, random_state=0))
    eval_hazard("lightgbm hazard", LGBMClassifier(
        n_estimators=500, num_leaves=47, learning_rate=0.04, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1))
    eval_hazard("catboost hazard", CatBoostClassifier(
        depth=6, iterations=400, learning_rate=0.04, l2_leaf_reg=3.0,
        loss_function="Logloss", verbose=0, random_seed=0))

    res = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    # ---- selection rule ---------------------------------------------------- #
    inc = res[res.name.str.startswith("histgbm")].iloc[0]
    haz = res[res.kind == "hazard"].sort_values("brier").iloc[0]
    rel = (inc.brier - haz.brier) / inc.brier
    if haz.name != inc.name and rel > 0.02:
        chosen, why = haz["name"], f"won calibrated Brier by {rel*100:.1f}% (> 2% rule), hazard backbone"
    else:
        chosen, why = inc["name"], "incumbent retained (no hazard challenger beat it by >2% Brier); C reuses it"

    md = ["# Task B - backbone bake-off (internal train time-split)\n",
          f"Selection on a held-out LATER time slice of train (fit={len(fit_idx)}, "
          f"cal={len(cal_idx)}, late={len(late_idx)}). Official validation reserved for final "
          "calibration. Fixed robust defaults, no grid search.\n",
          "| backbone | kind | cal-Brier | logloss | AUC | ECE | policyNPV |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        md.append(f"| {r['name']} | {r['kind']} | {r['brier']:.4f} | {r['logloss']:.4f} | "
                  f"{r['auc']:.3f} | {r['ece']:.4f} | {r.get('policy_npv', float('nan')):,.0f} |")
    md += [f"\n**Chosen backbone:** {chosen}", f"\n**Rationale:** {why}.",
           "\nBackbone is now FROZEN. Bootstrap kept only for interval uncertainty, not performance."]
    (CH / "audit" / "bakeoff.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[-3:]))
    print(f"\nsaved -> audit/bakeoff.md")


if __name__ == "__main__":
    main()
