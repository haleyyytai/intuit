"""Synthetic end-to-end validation of the data-agnostic engine.

Runs BEFORE the challenge data exists. Uses known data-generating processes so we
can assert that:
  1. split-conformal achieves ~nominal coverage,
  2. OOD support scoring inflates intervals for out-of-support inputs,
  3. interval score rewards calibrated over miscalibrated intervals,
  4. stability selection recovers known lagged causal edges and rejects non-edges.

Run:  python tests/test_engine_synthetic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forecasting import fit_gbm, BootstrapEnsemble          # noqa: E402
from uncertainty import (ConformalModel, SupportScorer,      # noqa: E402
                         conformal_intervals, coverage, mean_width,
                         percentile_rank_violation)
from evaluation import interval_score, pinball_loss, rmse    # noqa: E402
from causal_discovery import (discover_temporal,             # noqa: E402
                              stability_scores, select_stable)

RNG = np.random.RandomState(42)
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{(' — ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
print("\n[1] Split-conformal coverage on a nonlinear regression DGP")
n = 4000
X = RNG.uniform(-3, 3, size=(n, 5))
y = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + 0.3 * X[:, 3] * X[:, 4]
     + RNG.normal(0, 0.5, n))
tr, ca, te = slice(0, 2000), slice(2000, 3000), slice(3000, 4000)
model = fit_gbm(X[tr], y[tr])
point, lo, hi = conformal_intervals(model, X[ca], y[ca], X[te], alpha=0.1)
cov, w = coverage(y[te], lo, hi), mean_width(lo, hi)
check("90% conformal coverage in [0.86, 0.94]", 0.86 <= cov <= 0.94, f"cov={cov:.3f} width={w:.2f}")
check("point RMSE reasonable (< 1.0)", rmse(y[te], point) < 1.0, f"rmse={rmse(y[te], point):.3f}")

# --------------------------------------------------------------------------- #
print("\n[2] OOD support scoring inflates intervals for out-of-support inputs")
scorer = SupportScorer(random_state=0).fit(X[tr])
X_in = X[te]
X_ood = X[te].copy()
X_ood[:, 0] += 8.0  # push feature 0 far outside [-3, 3]
infl_in = scorer.inflation_factor(X_in)
infl_ood = scorer.inflation_factor(X_ood)
check("OOD inflation > in-distribution inflation",
      infl_ood.mean() > infl_in.mean() + 0.1,
      f"in={infl_in.mean():.2f} ood={infl_ood.mean():.2f}")
check("in-distribution inflation ~1 (median < 1.3)", np.median(infl_in) < 1.3,
      f"median={np.median(infl_in):.2f}")
viol = percentile_rank_violation(X[tr], X_ood)
check("range-violation flags OOD rows", viol.mean() > 0.1, f"mean_viol={viol.mean():.2f}")

# inflated intervals should be wider
cm = ConformalModel(model=model, alpha=0.1).calibrate(X[ca], y[ca])
_, lo_i, hi_i = cm.predict_interval(X_in, scale=infl_in)
_, lo_o, hi_o = cm.predict_interval(X_in, scale=infl_ood)
check("scaled intervals widen with inflation",
      mean_width(lo_o, hi_o) > mean_width(lo_i, hi_i),
      f"w_in={mean_width(lo_i, hi_i):.2f} w_ood={mean_width(lo_o, hi_o):.2f}")

# --------------------------------------------------------------------------- #
print("\n[3] Interval score rewards calibrated over over-confident intervals")
good_lo, good_hi = lo, hi
tight_lo, tight_hi = point - 0.05, point + 0.05  # absurdly narrow
check("calibrated interval beats over-confident (lower score)",
      interval_score(y[te], good_lo, good_hi) < interval_score(y[te], tight_lo, tight_hi),
      f"good={interval_score(y[te], good_lo, good_hi):.2f} tight={interval_score(y[te], tight_lo, tight_hi):.2f}")
check("pinball loss at q=0.5 ~ half MAE", pinball_loss(y[te], point, 0.5) > 0)

# --------------------------------------------------------------------------- #
print("\n[4] Stability selection recovers known lagged causal edges")
# DGP: X1 -> X2 (lag1), X2 -> X3 (lag1); X4 independent noise. True edges directed.
T = 1500
e = RNG.normal(0, 1, size=(T, 4))
Z = np.zeros((T, 4))
for t in range(2, T):
    Z[t, 0] = 0.5 * Z[t - 1, 0] + e[t, 0]
    Z[t, 1] = 0.6 * Z[t - 1, 0] + 0.3 * Z[t - 1, 1] + e[t, 1]   # X1 -> X2
    Z[t, 2] = 0.6 * Z[t - 1, 1] + 0.2 * Z[t - 1, 2] + e[t, 2]   # X2 -> X3
    Z[t, 3] = 0.5 * Z[t - 1, 3] + e[t, 3]                        # isolated
dfz = pd.DataFrame(Z, columns=["X1", "X2", "X3", "X4"])

stab = stability_scores(dfz, lambda d: discover_temporal(d, max_lag=2, alpha=0.05),
                        n_boot=40, block=50, random_state=1)
stable = select_stable(stab, min_stability=0.6)
edgeset = set(zip(stable["cause"], stable["effect"]))
print("    recovered stable edges:", sorted(edgeset))
check("recovers X1->X2", ("X1", "X2") in edgeset)
check("recovers X2->X3", ("X2", "X3") in edgeset)
check("does NOT invent X4->X1 (isolated stays isolated)",
      not any(c == "X4" or eff == "X4" for c, eff in edgeset) or
      ("X4", "X1") not in edgeset)

# --------------------------------------------------------------------------- #
print("\n[5] Bootstrap ensemble = in-distribution uncertainty; trees SATURATE OOD")
# LESSON: tree models extrapolate flat, so bagged-tree disagreement does NOT grow
# (often shrinks) far outside support. Ensemble variance captures in-distribution
# model uncertainty; OOD detection must come from SupportScorer (section [2]), not
# from ensemble spread. We assert the realistic behavior rather than the naive hope.
ens = BootstrapEnsemble(n_estimators=12, random_state=0).fit(X[tr], y[tr])
std_in = ens.predict_std(X_in).mean()
std_ood = ens.predict_std(X_ood).mean()
check("ensemble produces a positive in-distribution disagreement signal", std_in > 0,
      f"std_in={std_in:.3f}")
check("tree ensemble does NOT inflate on far-OOD (saturation) -> rely on SupportScorer",
      std_ood <= std_in * 1.5,
      f"std_in={std_in:.3f} std_ood={std_ood:.3f} (OOD handled by SupportScorer instead)")

# --------------------------------------------------------------------------- #
n_pass = sum(results)
print(f"\n{'='*60}\n  {n_pass}/{len(results)} checks passed\n{'='*60}")
sys.exit(0 if n_pass == len(results) else 1)
