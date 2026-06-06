"""Textbook credit-risk baseline that the writeup names: pooled discrete-time hazard with
a GLM link (logit and complementary-log-log) on the person-period expansion, benchmarked
against the incumbent HistGBM hazard on the SAME internal train time-split. Decides whether
the GLM hazard should displace the GBM (pre-registered: >2% relative calibrated Brier).

Run:  python src/glm_hazard.py   -> audit/glm_hazard.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import (build_features, feature_columns, default_week,  # noqa: E402
                                expand_person_period, WeeklyHazard, DATA, CH, N_WEEKS)
from calibration import fit_isotonic, ece  # noqa: E402


def brier(p, y): return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": brier(p, y), "logloss": float(log_loss(y, p)),
            "auc": float(roc_auc_score(y, p)), "ece": ece(p, y)}


def main():
    tr = pd.read_csv(DATA / "train.csv")
    feat = feature_columns(tr); miss = [c for c in feat if tr[c].isna().mean() > 0.05]
    lab = tr[tr.default_flag.notna()].copy()
    lab["ts"] = pd.to_datetime(lab["application_timestamp"], errors="coerce")
    lab = lab.sort_values("ts").reset_index(drop=True)
    n = len(lab)
    fit_i = np.arange(0, int(0.56 * n)); cal_i = np.arange(int(0.56 * n), int(0.70 * n)); late_i = np.arange(int(0.70 * n), n)

    Xf = build_features(lab.iloc[fit_i].reset_index(drop=True), feat, miss)
    dwk = default_week(lab.iloc[fit_i].reset_index(drop=True))
    PP, ev, _ = expand_person_period(Xf, dwk, np.ones(len(Xf)))
    cols = list(PP.columns)
    print(f"person-period rows (fit): {len(PP)}; weekly event rate {ev.mean():.4f}")

    Xc = build_features(lab.iloc[cal_i], feat, miss)
    Xl = build_features(lab.iloc[late_i], feat, miss)
    yc = (lab.iloc[cal_i].default_flag == True).astype(int).to_numpy()  # noqa: E712
    yl = (lab.iloc[late_i].default_flag == True).astype(int).to_numpy()  # noqa: E712

    # shared preprocessing for the GLMs (impute + scale)
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(PP)
    Z = prep.transform(PP)
    logit = LogisticRegression(max_iter=2000, C=1.0).fit(Z, ev)
    cloglog = sm.GLM(ev, sm.add_constant(Z, has_constant="add"),
                     family=sm.families.Binomial(sm.families.links.CLogLog())).fit()
    histgbm = WeeklyHazard(random_state=0).fit(PP, ev, np.ones(len(PP)))
    print("fitted: logit / cloglog / histgbm hazards")

    def pd90(kind, X):
        nn = len(X); H = np.empty((nn, N_WEEKS))
        for a in range(1, N_WEEKS + 1):
            row = X.copy(); row["age_week"] = a
            if kind == "histgbm":
                H[:, a - 1] = histgbm.clf_.predict_proba(row[cols])[:, 1]
            else:
                Zr = prep.transform(row[cols])
                H[:, a - 1] = (logit.predict_proba(Zr)[:, 1] if kind == "logit"
                               else cloglog.predict(sm.add_constant(Zr, has_constant="add")))
        H = np.clip(H, 1e-6, 1 - 1e-6)
        return 1 - np.cumprod(1 - H, axis=1)[:, -1]

    rows = []
    for kind in ["histgbm", "logit", "cloglog"]:
        iso = fit_isotonic(pd90(kind, Xc), yc)
        m = metrics(np.clip(iso.predict(pd90(kind, Xl)), 0, 1), yl)
        m["model"] = kind; rows.append(m)
        print(f"  {kind:9s} cal-Brier={m['brier']:.4f} logloss={m['logloss']:.4f} AUC={m['auc']:.3f} ECE={m['ece']:.4f}")

    res = pd.DataFrame(rows).set_index("model")
    inc = res.loc["histgbm", "brier"]; best_glm = res.loc[["logit", "cloglog"], "brier"].min()
    rel = (inc - best_glm) / inc
    decision = (f"a GLM hazard beat HistGBM by {rel*100:.1f}% relative Brier (>2%) -> consider switching"
                if rel > 0.02 else
                f"HistGBM retained (best GLM within {abs(rel)*100:.1f}% Brier, < 2% bar); GLM is the calibrated reference baseline")
    md = ["# GLM discrete-time hazard baseline (logit / cloglog) vs HistGBM\n",
          f"Internal train time-split (fit {len(fit_i)} / cal {len(cal_i)} / late {len(late_i)}); "
          "calibrated on the cal slice, evaluated on the later block.\n",
          "| hazard model | cal-Brier | logloss | AUC | ECE |", "|---|---|---|---|---|"]
    for k in ["histgbm", "logit", "cloglog"]:
        r = res.loc[k]; md.append(f"| {k} | {r.brier:.4f} | {r.logloss:.4f} | {r.auc:.3f} | {r.ece:.4f} |")
    md += [f"\n**Decision:** {decision}."]
    (CH / "audit" / "glm_hazard.md").write_text("\n".join(md))
    print(f"\nDECISION: {decision}")
    print("saved -> audit/glm_hazard.md")


if __name__ == "__main__":
    main()
