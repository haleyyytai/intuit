"""Local scoring harness: trains on an INTERNAL train time-split (early->fit, mid->
calibrate, late->evaluate) and reports the score components, so experiments are ranked
by measured impact, not debate. Directional (labeled rows are approved-only), used to
rank options, not chase tiny deltas.

Run:  python src/harness.py            # baseline vs E6 credibility features
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import (build_features, feature_columns, default_week,  # noqa: E402
                                expand_person_period, WeeklyHazard, DATA, N_WEEKS)
import economics as econ  # noqa: E402
from calibration import fit_isotonic, ece, binned_coverage  # noqa: E402

TR = pd.read_csv(DATA / "train.csv")


def add_credibility(X, df):
    """E6: stated-minus-observed revenue gap (self-reported optimism signal)."""
    X = X.copy()
    stated = pd.to_numeric(df["stated_annual_revenue"], errors="coerce").to_numpy()
    obs_ann = pd.to_numeric(df["observed_monthly_revenue_avg_3mo"], errors="coerce").to_numpy() * 12
    X["cred_gap"] = np.log1p(np.clip(stated, 0, None)) - np.log1p(np.clip(obs_ann, 0, None))
    X["cred_ratio"] = stated / (obs_ann + 1.0)
    return X


def evaluate(credibility=False, seed=0):
    feat = feature_columns(TR); miss = [c for c in feat if TR[c].isna().mean() > 0.05]
    lab = TR[TR.default_flag.notna()].copy()
    lab["ts"] = pd.to_datetime(lab["application_timestamp"], errors="coerce")
    lab = lab.sort_values("ts").reset_index(drop=True)
    n = len(lab)
    fit_i = np.arange(0, int(0.60 * n)); cal_i = np.arange(int(0.60 * n), int(0.75 * n)); late_i = np.arange(int(0.75 * n), n)

    def X(df):
        x = build_features(df, feat, miss)
        return add_credibility(x, df) if credibility else x

    Xf = X(lab.iloc[fit_i]); Xc = X(lab.iloc[cal_i]); Xl = X(lab.iloc[late_i])
    lgd = econ.fit_lgd_by_week(lab.iloc[fit_i])
    # IPW
    a_old = (TR.prior_decision == 1).astype(int).to_numpy()
    e = HistGradientBoostingClassifier(max_depth=3, max_iter=250, random_state=0).fit(
        (add_credibility(build_features(TR, feat, miss), TR) if credibility else build_features(TR, feat, miss)), a_old)
    eL = np.clip(e.predict_proba(Xf)[:, 1], 1e-3, 1 - 1e-3)
    w = np.clip(a_old.mean() / eL, 0.1, 10.0)
    dwk = default_week(lab.iloc[fit_i].reset_index(drop=True))
    PP, ev, wv = expand_person_period(Xf, dwk, w)
    rng = np.random.RandomState(seed)
    models = [WeeklyHazard(random_state=k).fit(PP.iloc[(i := rng.randint(0, len(PP), len(PP)))].reset_index(drop=True), ev[i], wv[i]) for k in range(5)]

    def curves(Xa):
        ps, cs, hs = zip(*[((r := m.curves(Xa.copy()))[0], r[1], r[2]) for m in models])
        return np.mean(ps, 0), np.std(ps, 0), np.mean(cs, 0), np.mean(hs, 0)

    pc, _, _, _ = curves(Xc)
    yc = (lab.iloc[cal_i].default_flag == True).astype(int).to_numpy()  # noqa: E712
    iso = fit_isotonic(pc, yc)
    pdl, sdl, cdrl, Hl = curves(Xl)
    pcal = np.clip(iso.predict(pdl), 0, 1)
    factor = pcal / np.clip(pdl, 1e-6, 1)
    Hcal = np.clip(Hl * factor[:, None], 1e-6, 1 - 1e-6); cdrcal = np.clip(cdrl * factor[:, None], 0, 1)

    yl = (lab.iloc[late_i].default_flag == True).astype(int).to_numpy()  # noqa: E712
    dl = pd.to_numeric(lab.iloc[late_i].days_to_default, errors="coerce").to_numpy()
    R = lab.iloc[late_i].requested_amount.to_numpy(float)
    rec = pd.to_numeric(lab.iloc[late_i].final_recovered_amount, errors="coerce").to_numpy()
    realized = np.where(yl == 0, econ.REPAID_RATIO * R, econ.realized_default_ratio(R, dl, rec) * R)
    enpv = econ.expected_npv(R, Hcal, lgd)
    pnl = float((realized * (enpv > 0)).sum())
    traj_mae = float(np.mean([abs(cdrcal[:, a - 1].mean() - np.mean((yl == 1) & (dl <= 7 * a))) for a in range(1, N_WEEKS + 1)]))
    half = 1.645 * sdl + 0.01
    cov = binned_coverage(pcal, np.clip(pcal - half, 0, 1), np.clip(pcal + half, 0, 1), yl)
    return {"S_PnL": pnl, "traj_MAE": traj_mae, "AUC": roc_auc_score(yl, pcal),
            "ECE": ece(pcal, yl), "cal_cov": cov, "approve": float((enpv > 0).mean())}


def main():
    base = evaluate(credibility=False)
    cred = evaluate(credibility=True)
    print(f"{'metric':10s} {'baseline':>14s} {'+credibility(E6)':>18s}")
    for k in ["S_PnL", "traj_MAE", "AUC", "ECE", "cal_cov", "approve"]:
        print(f"{k:10s} {base[k]:>14.4f} {cred[k]:>18.4f}")
    rel = (cred["S_PnL"] - base["S_PnL"]) / base["S_PnL"]
    print(f"\nE6 S_P&L delta: {rel:+.2%}  -> {'PROMOTE' if rel > 0.01 else 'KEEP BASELINE (no clear win)'}")


if __name__ == "__main__":
    main()
