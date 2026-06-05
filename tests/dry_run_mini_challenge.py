"""Full-pipeline rehearsal on the synthetic mini-challenge (known ground truth).

Exercises: task-type inference -> features -> temporal split -> baselines ->
stability-selected causal graph -> intervention forecast (naive vs graph-constrained
vs ground truth) -> conformal + OOD uncertainty. Asserts we recover the truth.

Run:  python tests/dry_run_mini_challenge.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import synthetic                                          # noqa: E402
from infer_task_type import infer_task_type, recommended_recipe  # noqa: E402
from features import add_lags, build_design_matrix        # noqa: E402
from validation import temporal_split                     # noqa: E402
from forecasting import NaiveBaseline, fit_linear, fit_gbm, BootstrapEnsemble  # noqa: E402
from causal_discovery import discover_temporal, stability_scores, select_stable  # noqa: E402
from uncertainty import ConformalModel, SupportScorer, coverage, mean_width  # noqa: E402
from evaluation import rmse, interval_score              # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
ok = []
def check(name, cond, detail=""):
    ok.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}{(' — ' + detail) if detail else ''}")

print("=" * 64)
print("MINI-CHALLENGE DRY RUN")
print("=" * 64)

# --- Stage 1: data + task type -------------------------------------------------
df = synthetic.generate(n=2500, noise=0.3, seed=0)
df.insert(0, "t", np.arange(len(df)))  # add a time index column for detection
print("\n[Stage 1] task-type inference")
verdict = infer_task_type(df, target="y")
print("   " + str(verdict).replace("\n", "\n   "))
check("detects time column 't'", verdict.time_col == "t")
check("classifies as timeseries", verdict.primary == "timeseries")
recipe = recommended_recipe(verdict)
print("   recipe:", recipe["forecaster"], "|", recipe["causal"])

# --- Stage: features + split ---------------------------------------------------
print("\n[Stage 3] features + temporal split + baselines")
feat_cols = ["x1", "x2", "x3", "x4", "y"]
dff = add_lags(df, feat_cols, lags=[1, 2])
dff = dff.dropna().reset_index(drop=True)
# predict y_t from LAGGED features only (no contemporaneous leakage)
lag_features = [c for c in dff.columns if "_lag" in c]
X, y, used = build_design_matrix(dff, target="y", feature_cols=lag_features, dropna=False)
tr, va, te = temporal_split(len(y), 0.7, 0.15)

scores = {}
for name, fitter in [("naive", lambda Xt, yt: NaiveBaseline().fit(Xt, yt)),
                     ("linear", fit_linear), ("gbm", fit_gbm)]:
    m = fitter(X[tr], y[tr])
    scores[name] = rmse(y[te], m.predict(X[te]))
for k, v in scores.items():
    print(f"   RMSE {k:7s} = {v:.4f}")
check("linear & gbm beat naive baseline",
      scores["linear"] < scores["naive"] and scores["gbm"] < scores["naive"],
      f"naive={scores['naive']:.3f} linear={scores['linear']:.3f} gbm={scores['gbm']:.3f}")

# --- Stage 4: stability-selected causal graph ---------------------------------
print("\n[Stage 4] stability-selected lagged causal graph")
stab = stability_scores(df[feat_cols], lambda d: discover_temporal(d, max_lag=2, alpha=0.05),
                        n_boot=30, block=60, random_state=1)
stable = select_stable(stab, min_stability=0.6)
print(stable[["cause", "effect", "lag", "stability", "sign"]].to_string(index=False))
edgeset = set(zip(stable["cause"], stable["effect"]))
# ground-truth directed pairs (collapsing lag): x1->x2, x4->x3, x2->y, x3->y
for c, eff in [("x1", "x2"), ("x2", "y"), ("x4", "x3"), ("x3", "y")]:
    check(f"recovers {c}->{eff}", (c, eff) in edgeset)
check("y is not (spuriously) a cause of its own drivers",
      ("y", "x1") not in edgeset and ("y", "x4") not in edgeset)

# --- Stage 5: intervention forecast vs ground truth ---------------------------
# THE central lesson: naive L1 feature-replacement vs structural L3 roll-forward.
print("\n[Stage 5] intervention effect: persistent do(x1 += 1.0) on y")
truth = synthetic.true_effect_on_y({"x1": 1.0}, n=6000, seed=0)
gbm = fit_gbm(X[tr], y[tr])

# L1 naive: shift x1 lag features by +1 in a one-step y-model, re-predict.
base_pred = gbm.predict(X[te]).mean()
X_do = X.copy()
for j, col in enumerate(used):
    if col.startswith("x1_lag"):
        X_do[:, j] = X_do[:, j] + 1.0
l1_effect = gbm.predict(X_do[te]).mean() - base_pred

# L3 structural roll-forward: fit a per-variable linear SCM on lag1/lag2 of ALL
# vars, then simulate the system forward with and without the persistent shift.
from sklearn.linear_model import LinearRegression
VARS = ["x1", "x2", "x3", "x4", "y"]
scm_data = add_lags(df, VARS, lags=[1, 2]).dropna().reset_index(drop=True)
lagcols = [c for c in scm_data.columns if "_lag" in c]
scm = {v: LinearRegression().fit(scm_data[lagcols], scm_data[v]) for v in VARS}

def rollout(shift, steps=200, seed=0):
    rng = np.random.RandomState(seed)
    hist = {f"{v}_lag1": 0.0 for v in VARS}
    hist.update({f"{v}_lag2": 0.0 for v in VARS})
    cur = {v: 0.0 for v in VARS}
    prev = dict(cur)
    ys = []
    for _ in range(steps):
        row = {f"{v}_lag1": cur[v] for v in VARS}
        row.update({f"{v}_lag2": prev[v] for v in VARS})
        feat = pd.DataFrame([row])[lagcols]
        nxt = {v: float(scm[v].predict(feat)[0]) + rng.normal(0, 0.3) for v in VARS}
        nxt["x1"] += shift.get("x1", 0.0)  # persistent intervention enters the mechanism
        prev, cur = cur, nxt
        ys.append(nxt["y"])
    return np.mean(ys[steps // 2:])  # average over the back half (post-burn-in)

l3_effect = rollout({"x1": 1.0}) - rollout({})
print(f"   ground-truth (MC, full system)   = {truth:+.3f}")
print(f"   L1 naive feature replacement     = {l1_effect:+.3f}   <- structurally blind to multi-step path")
print(f"   L3 structural roll-forward       = {l3_effect:+.3f}")
check("L1 naive massively UNDER-estimates (the documented failure mode)",
      abs(l1_effect) < 0.2 * abs(truth), f"|L1|={abs(l1_effect):.2f} vs truth={truth:.1f}")
check("L3 structural recovers the effect within 25% of truth",
      abs(l3_effect - truth) < 0.25 * abs(truth), f"L3={l3_effect:.2f} truth={truth:.2f}")
check("both methods agree on sign", np.sign(l1_effect) == np.sign(truth) == np.sign(l3_effect))

# --- Stage 6: conformal + OOD uncertainty -------------------------------------
print("\n[Stage 6] conformal coverage (raw) + OOD inflation")
cm = ConformalModel(model=gbm, alpha=0.1).calibrate(X[va], y[va])
scorer = SupportScorer(random_state=0).fit(X[tr])
_, lo_raw, hi_raw = cm.predict_interval(X[te])
cov_raw, w_raw = coverage(y[te], lo_raw, hi_raw), mean_width(lo_raw, hi_raw)
infl = scorer.inflation_factor(X[te])
_, lo, hi = cm.predict_interval(X[te], scale=infl)
cov, w = coverage(y[te], lo, hi), mean_width(lo, hi)
print(f"   raw      90% -> coverage={cov_raw:.3f} width={w_raw:.3f} interval_score={interval_score(y[te], lo_raw, hi_raw):.3f}")
print(f"   inflated 90% -> coverage={cov:.3f} width={w:.3f}")
check("RAW conformal coverage in [0.85, 0.95]", 0.85 <= cov_raw <= 0.95, f"cov={cov_raw:.3f}")
check("OOD inflation widens intervals & raises coverage (safety trade)",
      w >= w_raw and cov >= cov_raw, f"w_raw={w_raw:.2f}->{w:.2f} cov_raw={cov_raw:.2f}->{cov:.2f}")
infl_do = scorer.inflation_factor(X_do[te])  # x1 shifted -> more OOD
check("OOD inflation rises under the intervention", infl_do.mean() >= infl.mean(),
      f"base={infl.mean():.2f} do={infl_do.mean():.2f}")

print("\n" + "=" * 64)
print(f"  {sum(ok)}/{len(ok)} checks passed")
print("=" * 64)
sys.exit(0 if sum(ok) == len(ok) else 1)
