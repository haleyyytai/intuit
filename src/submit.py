"""End-to-end driver: load -> infer task -> features -> split -> baselines ->
causal discovery -> intervention forecasts -> uncertainty -> write artifacts.

Single reproducible entry point.

    python src/submit.py --demo                 # run on synthetic mini-challenge
    python src/submit.py --data path --target y # run on real challenge data

Writes to outputs/: predictions/, causal_graph/, tables/, figures/.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"

import synthetic                                                  # noqa: E402
from infer_task_type import infer_task_type, recommended_recipe   # noqa: E402
from features import add_lags                                     # noqa: E402
from validation import temporal_split                             # noqa: E402
from forecasting import NaiveBaseline, fit_linear, fit_gbm        # noqa: E402
from causal_discovery import (discover_temporal, discover_tabular,  # noqa: E402
                              stability_scores, select_stable, to_graphml,
                              prune_indirect_edges)
from intervention import classify_roles                           # noqa: E402
from uncertainty import (ConformalModel, WeightedConformalModel,  # noqa: E402
                         SupportScorer, coverage, mean_width)
from evaluation import rmse, mae, interval_score, competition_metric  # noqa: E402


def log(msg): print(f"[submit] {msg}", flush=True)


def load(args):
    if args.demo or not args.data:
        log("loading synthetic mini-challenge (no --data given)")
        df = synthetic.generate(n=2500, noise=0.3, seed=0)
        df.insert(0, "t", np.arange(len(df)))
        return df, "y"
    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)
    return df, args.target


def main(args):
    for d in ["predictions", "causal_graph", "tables", "figures"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    df, target = load(args)
    log(f"data: {df.shape[0]} rows x {df.shape[1]} cols; target='{target}'")

    # --- Stage 1: task type ---------------------------------------------------
    verdict = infer_task_type(df, target=target)
    log(f"task type = {verdict.primary} (flags: {verdict.flags or 'none'})")
    recipe = recommended_recipe(verdict)
    temporal = verdict.primary in ("timeseries", "panel")

    # --- Stage 3: features + split + baselines --------------------------------
    base_vars = [c for c in df.columns if c not in (verdict.time_col, verdict.entity_col)]
    if temporal:
        dff = add_lags(df, base_vars, lags=[1, 2]).dropna().reset_index(drop=True)
        feat_cols = [c for c in dff.columns if "_lag" in c]
    else:
        dff, feat_cols = df.dropna().reset_index(drop=True), [c for c in base_vars if c != target]
    X = dff[feat_cols].to_numpy(float)
    y = dff[target].to_numpy(float)
    tr, va, te = temporal_split(len(y), 0.7, 0.15) if temporal else temporal_split(len(y))

    models = {"naive": NaiveBaseline().fit(X[tr], y[tr]),
              "linear": fit_linear(X[tr], y[tr]),
              "gbm": fit_gbm(X[tr], y[tr])}
    rows = [{"model": k, "rmse": rmse(y[te], m.predict(X[te])),
             "mae": mae(y[te], m.predict(X[te])),
             "metric": competition_metric(y[te], m.predict(X[te]))} for k, m in models.items()]
    perf = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    perf.to_csv(OUT / "tables" / "model_comparison.csv", index=False)
    best_name = perf.iloc[0]["model"]
    best = models[best_name]
    log(f"baselines done; best = {best_name} (rmse={perf.iloc[0]['rmse']:.4f})")

    # --- Stage 4: stability-selected causal graph -----------------------------
    if temporal:
        edges = stability_scores(df[base_vars], lambda d: discover_temporal(d, max_lag=2),
                                 n_boot=args.nboot, block=60, random_state=1)
    else:
        edges = stability_scores(df[base_vars], discover_tabular, n_boot=args.nboot, random_state=1)
    stable = select_stable(edges, min_stability=0.6)
    stable.to_csv(OUT / "tables" / "causal_edges.csv", index=False)
    # conditional (MCI-style) pruning -> keep only DIRECT edges for the causal graph
    if temporal and len(stable):
        direct = prune_indirect_edges(df[base_vars], stable, max_lag=2, alpha=0.05)
        direct.to_csv(OUT / "tables" / "causal_edges_direct.csv", index=False)
        log(f"causal graph: {len(stable)} stable candidates -> {len(direct)} direct "
            f"after conditional pruning")
        graph_edges = direct if len(direct) else stable
    else:
        graph_edges = stable
        log(f"causal graph: {len(stable)} stable edges (of {len(edges)} candidates)")
    if len(graph_edges):
        to_graphml(graph_edges, OUT / "causal_graph" / "graph.graphml")

    # --- Stage 6: uncertainty (weighted conformal, covariate-shift robust) -----
    # Primary: weighted split-conformal (Tibshirani 2019) — reweights calibration
    # residuals by an estimated density ratio, the principled form of OOD widening.
    # Also report plain split-conformal as a marginal-coverage baseline.
    rows_c = []
    plain = ConformalModel(model=best, alpha=0.1).calibrate(X[va], y[va])
    _, lo_p, hi_p = plain.predict_interval(X[te])
    rows_c.append({"method": "split_conformal", "target_coverage": 0.90,
                   "empirical_coverage": round(coverage(y[te], lo_p, hi_p), 4),
                   "mean_width": round(mean_width(lo_p, hi_p), 4),
                   "interval_score": round(interval_score(y[te], lo_p, hi_p), 4)})
    wc = WeightedConformalModel(model=best, alpha=0.1, random_state=0).calibrate(X[va], y[va])
    point, lo, hi = wc.predict_interval(X[te])
    cov = coverage(y[te], lo, hi)
    rows_c.append({"method": "weighted_conformal", "target_coverage": 0.90,
                   "empirical_coverage": round(cov, 4),
                   "mean_width": round(mean_width(lo, hi), 4),
                   "interval_score": round(interval_score(y[te], lo, hi), 4)})
    pd.DataFrame(rows_c).to_csv(OUT / "tables" / "calibration.csv", index=False)
    log(f"uncertainty: weighted-conformal 90% coverage = {cov:.3f} "
        f"(plain = {rows_c[0]['empirical_coverage']})")

    # --- write submission predictions -----------------------------------------
    sub = pd.DataFrame({"row": np.arange(len(point)), "prediction": point,
                        "lower_90": lo, "upper_90": hi})
    sub.to_csv(OUT / "predictions" / "submission.csv", index=False)
    log(f"wrote {len(sub)} predictions -> outputs/predictions/submission.csv")

    # --- figures --------------------------------------------------------------
    _make_figures(verdict, perf, stable, y[te], point, lo, hi)
    log("figures written -> outputs/figures/")
    print("\n=== SUMMARY ===")
    print(perf.to_string(index=False))
    print(f"\nbest model: {best_name} | stable edges: {len(stable)} | coverage@90: {cov:.3f}")
    print(f"task: {verdict.primary} | recipe: {recipe.get('causal','-')}")


def _make_figures(verdict, perf, stable, y_te, point, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # model comparison
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(perf["model"], perf["rmse"], color="steelblue")
    ax.set_xlabel("RMSE"); ax.set_title("Model comparison"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / "figures" / "model_comparison.png", dpi=130); plt.close(fig)

    # forecast w/ intervals (first 150 test points)
    k = min(150, len(point))
    fig, ax = plt.subplots(figsize=(8, 3))
    xs = np.arange(k)
    ax.fill_between(xs, lo[:k], hi[:k], alpha=0.25, color="orange", label="90% interval")
    ax.plot(xs, y_te[:k], lw=1, color="black", label="actual")
    ax.plot(xs, point[:k], lw=1, color="orange", label="forecast")
    ax.set_title("Forecast with calibrated intervals"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "forecast_intervals.png", dpi=130); plt.close(fig)

    # causal graph
    if len(stable):
        import networkx as nx
        g = nx.DiGraph()
        for _, r in stable.iterrows():
            g.add_edge(r["cause"], r["effect"], w=r["stability"])
        fig, ax = plt.subplots(figsize=(5, 4))
        pos = nx.spring_layout(g, seed=1)
        widths = [3 * g[u][v]["w"] for u, v in g.edges]
        nx.draw(g, pos, ax=ax, with_labels=True, node_color="lightblue",
                node_size=900, width=widths, arrowsize=18, font_size=10)
        ax.set_title("Stability-selected causal graph")
        fig.tight_layout(); fig.savefig(OUT / "figures" / "causal_graph.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to challenge data (csv/parquet)")
    ap.add_argument("--target", default="y", help="target column name")
    ap.add_argument("--demo", action="store_true", help="run on synthetic mini-challenge")
    ap.add_argument("--nboot", type=int, default=30, help="bootstrap rounds for stability")
    main(ap.parse_args())
