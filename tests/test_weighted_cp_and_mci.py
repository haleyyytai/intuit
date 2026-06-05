"""Validate the two literature-driven upgrades:
  (A) weighted conformal (covariate-shift robust) vs plain split conformal under
      an intervention-induced distribution shift;
  (B) conditional-Granger (MCI-essence) pruning of the indirect x1->y artifact.

Run:  python tests/test_weighted_cp_and_mci.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import synthetic                                                  # noqa: E402
from features import add_lags                                     # noqa: E402
from forecasting import fit_gbm                                   # noqa: E402
from uncertainty import (ConformalModel, WeightedConformalModel,  # noqa: E402
                         SupportScorer, coverage, mean_width)
from causal_discovery import (discover_temporal, stability_scores,  # noqa: E402
                              select_stable, prune_indirect_edges,
                              conditional_granger_pvalue)

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
ok = []
def check(name, cond, detail=""):
    ok.append(cond); print(f"  [{PASS if cond else FAIL}] {name}{(' — ' + detail) if detail else ''}")


# === (A) Weighted conformal under covariate shift ============================ #
print("[A] weighted conformal vs plain split conformal under shift")
rng = np.random.RandomState(0)
n = 5000
X = rng.uniform(-3, 3, size=(n, 4))
y = np.sin(X[:, 0]) + 0.6 * X[:, 1] ** 2 - X[:, 2] + rng.normal(0, 0.4 + 0.15 * np.abs(X[:, 1]))
tr, ca = slice(0, 2500), slice(2500, 4000)
model = fit_gbm(X[tr], y[tr])

# shifted TEST set: sample preferentially from the high-|x1| (heteroscedastic, tail) region
pool = X[4000:]
yp = y[4000:]
shift_w = np.exp(0.9 * np.abs(pool[:, 1]))  # over-weight large |x1| -> covariate shift
shift_w /= shift_w.sum()
sel = rng.choice(len(pool), size=600, replace=True, p=shift_w)
Xte, yte = pool[sel], yp[sel]

# plain split conformal (marginal)
plain = ConformalModel(model=model, alpha=0.1).calibrate(X[ca], y[ca])
_, lo_p, hi_p = plain.predict_interval(Xte)
cov_p, w_p = coverage(yte, lo_p, hi_p), mean_width(lo_p, hi_p)

# weighted conformal (covariate-shift robust)
wc = WeightedConformalModel(model=model, alpha=0.1, random_state=0).calibrate(X[ca], y[ca])
_, lo_w, hi_w = wc.predict_interval(Xte)
cov_w, w_w = coverage(yte, lo_w, hi_w), mean_width(lo_w, hi_w)

print(f"   plain    -> coverage={cov_p:.3f} width={w_p:.3f}")
print(f"   weighted -> coverage={cov_w:.3f} width={w_w:.3f}")
check("weighted conformal coverage closer to 0.90 than plain",
      abs(cov_w - 0.90) <= abs(cov_p - 0.90) + 0.005, f"|w-.9|={abs(cov_w-.9):.3f} |p-.9|={abs(cov_p-.9):.3f}")
check("weighted conformal reaches >= 0.88 coverage under shift", cov_w >= 0.88, f"cov_w={cov_w:.3f}")
check("weighting adapts width (wider than plain in shifted tail)", w_w >= w_p - 1e-6,
      f"w_w={w_w:.3f} w_p={w_p:.3f}")


# === (B) Conditional-Granger pruning of the indirect edge ==================== #
print("\n[B] conditional-Granger removes indirect x1->y, keeps direct edges")
df = synthetic.generate(n=3000, noise=0.3, seed=1)
vars5 = ["x1", "x2", "x3", "x4", "y"]

# candidate edges via pairwise stability selection (knowingly includes indirect x1->y)
stab = stability_scores(df[vars5], lambda d: discover_temporal(d, max_lag=2, alpha=0.05),
                        n_boot=25, block=60, random_state=2)
cand = select_stable(stab, min_stability=0.6)
cand_set = set(zip(cand["cause"], cand["effect"]))
print("   candidate edges:", sorted(cand_set))

# direct p-values for the two competing edges into y
p_direct_x2 = conditional_granger_pvalue(df, "x2", "y", vars5, max_lag=2)
p_indirect_x1 = conditional_granger_pvalue(df, "x1", "y", vars5, max_lag=2)
print(f"   p(x2->y | rest) = {p_direct_x2:.4g}   (true DIRECT, expect small)")
print(f"   p(x1->y | rest) = {p_indirect_x1:.4g}   (indirect via x2, expect large)")
check("direct x2->y is significant after conditioning", p_direct_x2 < 0.05)
check("indirect x1->y is NOT significant after conditioning", p_indirect_x1 > 0.05)

pruned = prune_indirect_edges(df, cand, max_lag=2, alpha=0.05)
pruned_set = set(zip(pruned["cause"], pruned["effect"]))
print("   pruned (direct) edges:", sorted(pruned_set))
check("pruning drops x1->y artifact", ("x1", "y") not in pruned_set)
check("pruning keeps x2->y", ("x2", "y") in pruned_set)
check("pruning keeps x3->y", ("x3", "y") in pruned_set)

print("\n" + "=" * 60)
print(f"  {sum(ok)}/{len(ok)} checks passed")
print("=" * 60)
sys.exit(0 if sum(ok) == len(ok) else 1)
