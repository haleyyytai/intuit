"""5-fold CV of the PD backbone on the labeled training set, two ways:
  (A) random stratified 5-fold  -> in-distribution generalization (what was requested)
  (B) rolling-origin 5-fold      -> temporally honest (no future->past leakage)
Reports AUC / ECE / Brier mean +/- std so strategy decisions rest on folds, not one split.

Run:  python src/kfold_cv.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from challenge_pipeline import build_features, feature_columns, DATA  # noqa: E402
from calibration import fit_isotonic, ece  # noqa: E402


def brier(p, y): return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def metrics(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return roc_auc_score(y, p), ece(p, y), brier(p, y)


def new_model():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.07,
                                          l2_regularization=1.0, random_state=0)


def main():
    tr = pd.read_csv(DATA / "train.csv")
    feat = feature_columns(tr); miss = [c for c in feat if tr[c].isna().mean() > 0.05]
    lab = tr[tr.default_flag.notna()].reset_index(drop=True)
    lab = lab.assign(_ts=pd.to_datetime(lab["application_timestamp"], errors="coerce")).sort_values("_ts").reset_index(drop=True)
    X = build_features(lab, feat, miss)
    y = (lab.default_flag == True).astype(int).to_numpy()  # noqa: E712

    def fit_eval(tr_i, te_i):
        m = new_model().fit(X.iloc[tr_i], y[tr_i])
        # calibrate on a 20% inner slice of the train fold, evaluate on the test fold
        n = len(tr_i); inner = tr_i[int(0.8 * n):]; core = tr_i[:int(0.8 * n)]
        m = new_model().fit(X.iloc[core], y[core])
        iso = fit_isotonic(m.predict_proba(X.iloc[inner])[:, 1], y[inner])
        pcal = np.clip(iso.predict(m.predict_proba(X.iloc[te_i])[:, 1]), 0, 1)
        return metrics(pcal, y[te_i])

    # (A) random stratified 5-fold
    print("(A) random stratified 5-fold (in-distribution):")
    rs = []
    for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
        a, e, b = fit_eval(tri, tei); rs.append((a, e, b))
        print(f"   fold {k+1}: AUC={a:.3f}  ECE={e:.4f}  Brier={b:.4f}")
    rs = np.array(rs)
    print(f"   MEAN: AUC={rs[:,0].mean():.3f}+/-{rs[:,0].std():.3f}  "
          f"ECE={rs[:,1].mean():.4f}+/-{rs[:,1].std():.4f}  Brier={rs[:,2].mean():.4f}+/-{rs[:,2].std():.4f}")

    # (B) rolling-origin 5-fold (temporally honest)
    print("\n(B) rolling-origin 5-fold (time-blocked, no leakage):")
    n = len(X); ro = []
    for k, (fe, ee) in enumerate([(0.50, 0.60), (0.58, 0.68), (0.66, 0.76), (0.74, 0.85), (0.82, 1.00)]):
        tri = np.arange(0, int(fe * n)); tei = np.arange(int(fe * n), int(ee * n))
        a, e, b = fit_eval(tri, tei); ro.append((a, e, b))
        print(f"   fold {k+1} (fit<{fe} eval{fe}-{ee}): AUC={a:.3f}  ECE={e:.4f}  Brier={b:.4f}")
    ro = np.array(ro)
    print(f"   MEAN: AUC={ro[:,0].mean():.3f}+/-{ro[:,0].std():.3f}  "
          f"ECE={ro[:,1].mean():.4f}+/-{ro[:,1].std():.4f}  Brier={ro[:,2].mean():.4f}+/-{ro[:,2].std():.4f}")

    print(f"\nverdict: random-5fold AUC {rs[:,0].mean():.3f} vs rolling-5fold {ro[:,0].mean():.3f} "
          f"-> {'AGREE (robust both ways)' if abs(rs[:,0].mean()-ro[:,0].mean())<0.02 else 'DIVERGE (temporal effect)'}")


if __name__ == "__main__":
    main()
