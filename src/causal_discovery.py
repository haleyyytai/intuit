"""Causal structure recovery.

Two branches, chosen after the Stage-1 data-type classification:
  * temporal  -> lagged dependencies via Granger / VAR coefficients
  * tabular   -> conditional independence via partial correlation

The unifying idea (our differentiator) is STABILITY SELECTION: never trust one
recovered graph. `stability_scores` re-runs any discovery function over bootstrap
resamples and keeps the edge frequency as a confidence score.

Output contract (consumed by report + outputs/causal_graph/):
    edges_df[cause, effect, lag, strength, stability, sign, evidence, kind]
    kind in {"observational", "plausible_causal"};  stability in [0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Tabular: partial correlation (conditional independence)
# --------------------------------------------------------------------------- #
def partial_corr_edges(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Edges where |partial correlation| (controlling for all other vars) > threshold.

    Partial correlation = negative off-diagonal of the inverse covariance (precision)
    matrix, normalized. Undirected association controlling for the rest — a
    conservative, fast skeleton. Direction is left to time order / domain.
    """
    cols = list(df.columns)
    X = np.asarray(df, float)
    # regularized precision for numerical stability
    cov = np.cov(X, rowvar=False)
    prec = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    d = np.sqrt(np.diag(prec))
    pcorr = -prec / np.outer(d, d)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = pcorr[i, j]
            if abs(r) > threshold:
                rows.append({"cause": cols[i], "effect": cols[j], "lag": 0,
                             "strength": abs(float(r)), "sign": int(np.sign(r)),
                             "evidence": "partial_corr", "kind": "observational"})
    return pd.DataFrame(rows)


def discover_tabular(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    return partial_corr_edges(df, threshold=threshold)


# --------------------------------------------------------------------------- #
# Temporal: pairwise Granger-style lagged dependence
# --------------------------------------------------------------------------- #
def granger_edges(df: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """Directed lagged edges X->Y where past X improves prediction of Y (Granger).

    Pairwise (not conditional) — a candidate generator, refined by stability
    selection and, ideally, a conditional method (VAR/PCMCI) when libs are present.
    """
    import contextlib
    import io
    import warnings

    from statsmodels.tsa.stattools import grangercausalitytests

    cols = list(df.columns)
    rows = []
    for tgt in cols:
        for src in cols:
            if src == tgt:
                continue
            pair = df[[tgt, src]].dropna()  # order: [effect, cause] per statsmodels
            if len(pair) <= max_lag + 2:
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(pair, maxlag=max_lag)
            except Exception:
                continue
            # best lag by smallest ssr_ftest p-value
            best_lag, best_p = min(
                ((lag, res[lag][0]["ssr_ftest"][1]) for lag in res),
                key=lambda t: t[1],
            )
            if best_p < alpha:
                # sign proxy: correlation of cause(t-lag) with effect(t)
                s = pair[src].shift(best_lag)
                m = s.notna() & pair[tgt].notna()
                sign = int(np.sign(np.corrcoef(s[m], pair[tgt][m])[0, 1])) if m.sum() > 2 else 0
                rows.append({"cause": src, "effect": tgt, "lag": int(best_lag),
                             "strength": float(1 - best_p), "sign": sign,
                             "evidence": "granger", "kind": "plausible_causal"})
    return pd.DataFrame(rows)


def discover_temporal(df: pd.DataFrame, max_lag: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    return granger_edges(df, max_lag=max_lag, alpha=alpha)


# --------------------------------------------------------------------------- #
# Conditional Granger  ==  the essence of PCMCI+'s MCI test.
# Pairwise Granger reports X->Y even when the path is X->M->Y (indirect). The fix
# is to test whether X's past still predicts Y AFTER conditioning on the histories
# of all other variables (incl. the mediator). If conditioning removes the signal,
# the edge is indirect and should be dropped. This is full (conditional) Granger
# causality (Geweke), the linear analogue of momentary conditional independence.
# --------------------------------------------------------------------------- #
def _lagged(df: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    return pd.concat({f"{col}_l{L}": df[col].shift(L) for L in range(1, max_lag + 1)}, axis=1)


def conditional_granger_pvalue(df: pd.DataFrame, cause: str, effect: str,
                               conditioning, max_lag: int = 3) -> float:
    """F-test p-value for cause->effect, conditioning on effect's own past AND the
    pasts of every variable in `conditioning`. Low p => DIRECT effect survives."""
    from scipy import stats

    blocks = [_lagged(df, effect, max_lag)]
    for c in conditioning:
        if c not in (cause, effect):
            blocks.append(_lagged(df, c, max_lag))
    Zr = pd.concat(blocks, axis=1)
    Xc = _lagged(df, cause, max_lag)
    y = df[effect]
    full = pd.concat([y.rename("y"), Zr, Xc], axis=1).dropna()
    if len(full) < (Zr.shape[1] + Xc.shape[1] + 5):
        return 1.0  # not enough data -> cannot claim a direct edge
    yv = full["y"].to_numpy(float)
    R = np.column_stack([np.ones(len(full)), full[Zr.columns].to_numpy(float)])
    F = np.column_stack([R, full[Xc.columns].to_numpy(float)])

    def _rss(X, y):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r)

    rss_r, rss_f = _rss(R, yv), _rss(F, yv)
    df_num, df_den = Xc.shape[1], len(full) - F.shape[1]
    if df_den <= 0 or rss_f <= 0:
        return 1.0
    Fstat = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    return float(stats.f.sf(Fstat, df_num, df_den))


def prune_indirect_edges(df: pd.DataFrame, edges_df: pd.DataFrame, max_lag: int = 3,
                         alpha: float = 0.05) -> pd.DataFrame:
    """Keep only edges that survive the conditional (MCI-style) test. Adds a
    `direct_pvalue` column and upgrades `evidence` to 'granger+conditional'."""
    if not len(edges_df):
        return edges_df
    allvars = list(df.columns)
    out = edges_df.copy()
    pvals, keep = [], []
    for _, r in out.iterrows():
        p = conditional_granger_pvalue(df, r["cause"], r["effect"],
                                       conditioning=allvars, max_lag=max_lag)
        pvals.append(p)
        keep.append(p < alpha)
    out["direct_pvalue"] = pvals
    out["evidence"] = np.where(keep, "granger+conditional", "indirect(pruned)")
    return out[np.array(keep)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stability selection — the core differentiator
# --------------------------------------------------------------------------- #
def _edge_key(row) -> tuple:
    return (row["cause"], row["effect"], int(row.get("lag", 0)))


def stability_scores(df: pd.DataFrame, discover_fn, n_boot: int = 100,
                     frac: float = 0.8, random_state: int = 0,
                     block: int | None = None) -> pd.DataFrame:
    """Bootstrap `discover_fn` and return edges with a stability (frequency) score.

    For temporal data pass `block` (moving-block bootstrap length) to preserve
    autocorrelation; otherwise rows are resampled i.i.d.
    """
    rng = np.random.RandomState(random_state)
    n = len(df)
    counts: dict[tuple, int] = {}
    agg: dict[tuple, dict] = {}
    for _ in range(n_boot):
        if block:  # moving-block bootstrap for time series
            n_blocks = int(np.ceil(n / block))
            starts = rng.randint(0, max(1, n - block + 1), n_blocks)
            idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
            sample = df.iloc[idx].reset_index(drop=True)
        else:
            idx = rng.choice(n, size=int(frac * n), replace=True)
            sample = df.iloc[idx]
        try:
            edges = discover_fn(sample)
        except Exception:
            continue
        for _, row in edges.iterrows():
            k = _edge_key(row)
            counts[k] = counts.get(k, 0) + 1
            a = agg.setdefault(k, {"strength": [], "sign": []})
            a["strength"].append(row.get("strength", np.nan))
            a["sign"].append(row.get("sign", 0))
    rows = []
    for k, c in counts.items():
        cause, effect, lag = k
        signs = agg[k]["sign"]
        mean_sign = int(np.sign(np.nanmean(signs))) if signs else 0
        rows.append({
            "cause": cause, "effect": effect, "lag": lag,
            "stability": c / n_boot,
            "strength": float(np.nanmean(agg[k]["strength"])),
            "sign": mean_sign,
            "sign_consistency": float(np.mean(np.sign(signs) == mean_sign)) if signs else 0.0,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("stability", ascending=False).reset_index(drop=True) if len(out) else out


def select_stable(edges_df: pd.DataFrame, min_stability: float = 0.6) -> pd.DataFrame:
    """Keep only edges that recur in >= min_stability of bootstrap runs."""
    if not len(edges_df):
        return edges_df
    return edges_df[edges_df["stability"] >= min_stability].reset_index(drop=True)


def to_graphml(edges_df: pd.DataFrame, path) -> None:
    """Persist the recovered directed graph for plotting / submission."""
    import networkx as nx

    g = nx.DiGraph()
    for _, r in edges_df.iterrows():
        g.add_edge(r["cause"], r["effect"], lag=int(r.get("lag", 0)),
                   stability=float(r.get("stability", np.nan)),
                   sign=int(r.get("sign", 0)))
    nx.write_graphml(g, str(path))
