"""Intervention forecasting — the part most teams get wrong.

Three levels of increasing causal fidelity:
  L1 naive feature replacement   — set do(X=x); predict. Fast, but biased if X has
                                    downstream mediators held fixed.
  L2 graph-constrained           — use the recovered graph to classify variables;
                                    do NOT hold post-intervention mediators fixed.
  L3 structural roll-forward     — fit a per-variable SCM and simulate forward.

The bad-control guard is the core safety check: it refuses to condition on
descendants of the intervention as fixed covariates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------- #
# Variable role classification from a recovered (directed) edge table
# --------------------------------------------------------------------------- #
@dataclass
class VariableRoles:
    intervention: list
    target: str
    ancestors: set = field(default_factory=set)     # upstream of intervention (safe to fix)
    descendants: set = field(default_factory=set)    # downstream of intervention (mediators/bad-controls)
    other: set = field(default_factory=set)

    def safe_covariates(self) -> list:
        """Variables OK to hold fixed: not the target, not downstream of intervention."""
        bad = set(self.intervention) | self.descendants | {self.target}
        return sorted(self.ancestors | self.other - bad)

    def bad_controls(self) -> list:
        """Descendants of the intervention — must NOT be conditioned on as fixed."""
        return sorted(self.descendants - {self.target})


def build_graph(edges_df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, r in edges_df.iterrows():
        g.add_edge(r["cause"], r["effect"],
                   lag=int(r.get("lag", 0)), sign=int(r.get("sign", 0)),
                   stability=float(r.get("stability", np.nan)))
    return g


def classify_roles(edges_df: pd.DataFrame, intervention, target: str) -> VariableRoles:
    """Partition variables relative to the intervention using the recovered graph."""
    intervention = [intervention] if isinstance(intervention, str) else list(intervention)
    g = build_graph(edges_df)
    desc, anc = set(), set()
    for iv in intervention:
        if iv in g:
            desc |= nx.descendants(g, iv)
            anc |= nx.ancestors(g, iv)
    desc -= set(intervention)
    anc -= set(intervention)
    allv = set(g.nodes) | set(intervention) | {target}
    other = allv - desc - anc - set(intervention) - {target}
    return VariableRoles(intervention=intervention, target=target,
                         ancestors=anc, descendants=desc, other=other)


# --------------------------------------------------------------------------- #
# L1 — naive feature replacement
# --------------------------------------------------------------------------- #
def intervene_naive(model, X_row: pd.DataFrame, intervention: dict):
    """do(X=x) by overwriting columns, holding everything else fixed. Baseline only."""
    X_new = X_row.copy()
    for k, v in intervention.items():
        X_new[k] = v
    return model.predict(X_new)


# --------------------------------------------------------------------------- #
# L2 — graph-constrained intervention
# --------------------------------------------------------------------------- #
def assert_no_bad_controls(feature_cols, roles: VariableRoles) -> list:
    """Return the bad-control columns present in the feature set (empty = clean)."""
    bad = set(roles.bad_controls())
    return [c for c in feature_cols if _base_var(c) in bad]


def _base_var(col: str) -> str:
    """Strip lag/rolling suffixes so 'X2_lag1' maps back to variable 'X2'."""
    for sep in ("_lag", "_roll", "_diff"):
        if sep in col:
            return col.split(sep)[0]
    return col


def intervene_graph_constrained(target_model, mediator_models: dict, X_row: pd.DataFrame,
                                intervention: dict, roles: VariableRoles):
    """Set intervention, PROPAGATE to downstream mediators via their models, then
    predict the target. Mediators are recomputed (not held fixed) — that is the
    whole point: holding them fixed would block the intervention's effect.

    mediator_models: {mediator_name: fitted model taking the same design row}.
    """
    X_new = X_row.copy()
    for k, v in intervention.items():
        X_new[k] = v
    # recompute each downstream mediator we have a model for
    for m in roles.bad_controls():
        for col in X_new.columns:
            if _base_var(col) == m and m in mediator_models:
                X_new[col] = mediator_models[m].predict(X_new)
    return target_model.predict(X_new)


# --------------------------------------------------------------------------- #
# L3 — structural roll-forward simulation
# --------------------------------------------------------------------------- #
def structural_rollforward(scm: dict, init_state: pd.DataFrame, steps: int,
                           intervention: dict | None = None, order=None):
    """Simulate a fitted SCM forward `steps`, applying a persistent intervention.

    scm: {var: callable(history_df) -> next value}.  init_state: recent history.
    Returns a DataFrame of the simulated trajectory. Errors compound — report
    widening uncertainty with horizon (handled by the uncertainty module).
    """
    order = order or list(scm.keys())
    hist = init_state.copy()
    traj = []
    for _ in range(steps):
        nxt = {}
        for v in order:
            nxt[v] = (intervention[v] if intervention and v in intervention
                      else float(scm[v](hist)))
        row = pd.DataFrame([nxt])
        hist = pd.concat([hist, row], ignore_index=True)
        traj.append(nxt)
    return pd.DataFrame(traj)


# --------------------------------------------------------------------------- #
# Sanity-check table (Stage 5 deliverable)
# --------------------------------------------------------------------------- #
def intervention_sanity_row(name, baseline_pred, intervened_preds_bootstrap, support_score):
    """One row of the counterfactual sanity-check table.

    intervened_preds_bootstrap: array of intervened predictions across bootstrap
    models -> gives mean effect, direction stability, and a 90% interval.
    """
    arr = np.asarray(intervened_preds_bootstrap, float)
    effect = arr - np.asarray(baseline_pred, float).mean()
    mean_eff = float(effect.mean())
    dir_stab = float(np.mean(np.sign(effect) == np.sign(mean_eff))) if mean_eff != 0 else 0.0
    lo, hi = np.quantile(effect, [0.05, 0.95])
    return {"intervention": name, "mean_effect": round(mean_eff, 4),
            "direction_stability": round(dir_stab, 3),
            "ci90_low": round(float(lo), 4), "ci90_high": round(float(hi), 4),
            "support": "high" if support_score < 1.0 else ("medium" if support_score < 2.0 else "low"),
            "effect_vs_noise": "signal" if abs(mean_eff) > (hi - lo) / 4 else "noise-like"}
