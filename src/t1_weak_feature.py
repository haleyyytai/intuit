"""T1 GATE: measure whether the tree C-model recovers a KNOWN effect of a WEAK
(low-importance) feature, or flattens it. The real-data flatness of e.g.
stated_time_in_business could mean (a) the feature truly has ~0 effect (correct), or
(b) the tree attenuates a real effect (a deficit). This injects a known moderate
effect and measures recovery, mirroring the strong-feature protocol in c_validation.py.

Decision: recovery >= ~70% of injected slope -> flatness is cheap, SKIP T2.
         recovery <  ~30% -> real deficit, proceed to T2.

Run:  python src/t1_weak_feature.py
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RNG = np.random.RandomState(11)
THETA_STRONG = 0.45     # strong driver (leverage ratio)
BETA_WEAK = -0.35       # weak-but-REAL driver (tenure): more tenure -> less default


def zscore(x):
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def generate(n=12000):
    credit = RNG.normal(0, 1, n)
    monthly_rev = np.exp(RNG.normal(9.4, 0.5, n))
    ra = np.clip(8000 + 0.25 * monthly_rev + 3000 * credit + RNG.normal(0, 2500, n), 5000, 50000)
    ratio = ra / monthly_rev
    tenure = RNG.gamma(2.0, 3.0, n)                                   # the WEAK feature
    logit = (-1.6 + THETA_STRONG * ratio - 0.9 * credit + BETA_WEAK * zscore(tenure))
    p = 1 / (1 + np.exp(-logit))
    y = (RNG.uniform(size=n) < p).astype(int)
    appr = (1 / (1 + np.exp(-(2.0 + 1.4 * credit - 0.5 * ratio + RNG.normal(0, 0.5, n))))) > 0.5
    return pd.DataFrame({"credit": credit, "log_rev": np.log(monthly_rev), "log_ra": np.log(ra),
                         "ratio": ratio, "tenure": tenure, "monthly_rev": monthly_rev,
                         "y": y, "approved": appr.astype(int)})


FEATS = ["credit", "log_rev", "log_ra", "ratio", "tenure"]


def main():
    df = generate()
    appr = df[df.approved == 1]
    print(f"weak-feature (tenure) true logit coef = {BETA_WEAK} (z-scored); "
          f"approved {len(appr)}/{len(df)}")
    MODELS = {
        "histgbm": HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                                  l2_regularization=1.0, random_state=0).fit(appr[FEATS], appr.y),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(appr[FEATS], appr.y),
    }
    base = df.sample(3000, random_state=2).reset_index(drop=True)

    # weak feature (tenure) sweep over p10..p90
    tlo, thi = np.percentile(df.tenure, [10, 90]); grid = np.linspace(tlo, thi, 20)
    tmean, tstd = df.tenure.mean(), df.tenure.std()
    truth_w = np.array([(1 / (1 + np.exp(-(-1.6 + THETA_STRONG * base.ratio.values
              - 0.9 * base.credit.values + BETA_WEAK * ((v - tmean) / tstd))))).mean() for v in grid])
    # strong feature (requested_amount -> ratio) sweep
    rlo, rhi = np.percentile(df.log_ra, [10, 90]); grid2 = np.linspace(rlo, rhi, 20)
    truth_s = np.array([(1 / (1 + np.exp(-(-1.6 + THETA_STRONG * (np.exp(lv) / base.monthly_rev.values)
              - 0.9 * base.credit.values + BETA_WEAK * zscore(base.tenure.values))))).mean() for lv in grid2])
    span = lambda y, g: np.polyfit(g, y, 1)[0] * (g[-1] - g[0])

    print(f"\n{'model':10s} {'weak-recov':>11s} {'weak-MAE':>9s} {'strong-recov':>13s}")
    res = {}
    for name, model in MODELS.items():
        mw = np.array([model.predict_proba(base.assign(tenure=v)[FEATS])[:, 1].mean() for v in grid])
        ms = np.array([model.predict_proba(base.assign(log_ra=lv, ratio=np.exp(lv) / base.monthly_rev.values)[FEATS])[:, 1].mean() for lv in grid2])
        rw = span(mw, grid) / span(truth_w, grid); rs = span(ms, grid2) / span(truth_s, grid2)
        mae_w = float(np.mean(np.abs(mw - truth_w)))
        res[name] = (rw, mae_w, rs)
        print(f"{name:10s} {rw:>10.0%} {mae_w:>9.4f} {rs:>12.0%}")

    rw_h, rw_l = res["histgbm"][0], res["logistic"][0]
    rs_h, rs_l = res["histgbm"][2], res["logistic"][2]
    print(f"\nGATE: weak-feature recovery (HistGBM) = {rw_h:.0%} -> "
          f"{'flatness CHEAP' if rw_h >= 0.70 else 'deficit'}")
    print(f"T2-2c: logistic vs HistGBM recovery  weak {rw_l:.0%} vs {rw_h:.0%} | "
          f"strong {rs_l:.0%} vs {rs_h:.0%}")


if __name__ == "__main__":
    main()
