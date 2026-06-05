"""Automatic task-type inference — the first-30-minutes decision that routes
everything else (forecaster, validation scheme, causal branch, intervention logic).

Returns a structured verdict, not just a label, so the Stage-1 report can show WHY.
Classes: 'timeseries' | 'panel' | 'treatment' | 'tabular' | 'anonymized'
(these are not mutually exclusive in spirit; we return primary + flags).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import numpy as np
import pandas as pd

_TIME_HINTS = ("time", "date", "timestamp", "ds", "datetime", "period", "t", "step", "week", "month", "day", "hour")
_ENTITY_HINTS = ("id", "entity", "user", "store", "region", "company", "product", "sku", "group", "unit", "series")
_TREATMENT_HINTS = ("treat", "treatment", "intervention", "do_", "assigned", "policy", "arm", "group_assign")


@dataclass
class TaskVerdict:
    primary: str
    flags: list = field(default_factory=list)
    time_col: str | None = None
    entity_col: str | None = None
    treatment_cols: list = field(default_factory=list)
    anonymized: bool = False
    n_rows: int = 0
    n_cols: int = 0
    evidence: dict = field(default_factory=dict)

    def __str__(self):
        lines = [f"TASK TYPE: {self.primary.upper()}  (flags: {', '.join(self.flags) or 'none'})",
                 f"  rows={self.n_rows} cols={self.n_cols} anonymized={self.anonymized}",
                 f"  time_col={self.time_col} entity_col={self.entity_col} treatment={self.treatment_cols}"]
        for k, v in self.evidence.items():
            lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        try:
            parsed = pd.to_datetime(s.dropna().iloc[:200], errors="coerce")
            return parsed.notna().mean() > 0.9
        except Exception:
            return False
    return False


def _is_monotonic_index_like(s: pd.Series) -> bool:
    """Integer/float column that increases regularly -> a time step."""
    x = s.dropna().values
    if len(x) < 5 or not np.issubdtype(x.dtype, np.number):
        return False
    d = np.diff(np.sort(np.unique(x)))
    return len(d) > 0 and np.allclose(d, d[0], rtol=0.05) and s.is_monotonic_increasing


def _find_col(cols, hints) -> str | None:
    low = {c: c.lower() for c in cols}
    # exact-ish first
    for c, lc in low.items():
        if lc in hints:
            return c
    for c, lc in low.items():
        if any(h == lc or re.fullmatch(rf"{h}[_\d]*", lc) or lc.endswith("_" + h) for h in hints):
            return c
    for c, lc in low.items():
        if any(h in lc for h in hints):
            return c
    return None


def _is_anonymized(cols) -> bool:
    """Columns like x1..x50, v1.., feature_3 with no semantic content."""
    pat = re.compile(r"^(x|v|var|feat|feature|col|c|f)[_]?\d+$", re.I)
    hits = sum(bool(pat.match(str(c))) for c in cols)
    return hits / max(1, len(cols)) > 0.6


def infer_task_type(df: pd.DataFrame, target: str | None = None) -> TaskVerdict:
    cols = list(df.columns)
    ev = {}

    # --- time column detection ---
    time_col = None
    for c in cols:
        if _looks_datetime(df[c]):
            time_col, ev["time_evidence"] = c, f"{c} parses as datetime"
            break
    if time_col is None:
        for c in cols:
            if str(c).lower() in _TIME_HINTS and _is_monotonic_index_like(df[c]):
                time_col, ev["time_evidence"] = c, f"{c} is a regular monotonic step index"
                break
    if time_col is None:
        cand = _find_col(cols, _TIME_HINTS)
        if cand is not None and _is_monotonic_index_like(df[cand]):
            time_col, ev["time_evidence"] = cand, f"{cand} (named like time) is monotonic"

    # --- entity column detection (panel) ---
    entity_col = None
    ecand = _find_col([c for c in cols if c != time_col], _ENTITY_HINTS)
    if ecand is not None:
        nun = df[ecand].nunique()
        if 1 < nun < len(df) * 0.9:  # repeats -> grouping variable
            entity_col, ev["entity_evidence"] = ecand, f"{ecand} has {nun} repeating groups"

    # --- treatment column detection ---
    treatment_cols = []
    for c in cols:
        lc = str(c).lower()
        if any(h in lc for h in _TREATMENT_HINTS):
            treatment_cols.append(c)
        elif c != target and df[c].dropna().nunique() == 2 and any(
            h in lc for h in ("treat", "do", "policy", "arm")):
            treatment_cols.append(c)
    if treatment_cols:
        ev["treatment_evidence"] = f"binary/assigned cols: {treatment_cols}"

    anonymized = _is_anonymized(cols)
    if anonymized:
        ev["anonymized"] = "column names are non-semantic (xN/vN pattern)"

    # --- decide primary ---
    flags = []
    if time_col and entity_col:
        primary = "panel"
    elif time_col:
        primary = "timeseries"
    elif treatment_cols:
        primary = "treatment"
    else:
        primary = "tabular"
    if treatment_cols and primary != "treatment":
        flags.append("has_treatment")
    if anonymized:
        flags.append("anonymized")

    return TaskVerdict(primary=primary, flags=flags, time_col=time_col,
                       entity_col=entity_col, treatment_cols=treatment_cols,
                       anonymized=anonymized, n_rows=len(df), n_cols=len(cols),
                       evidence=ev)


def recommended_recipe(verdict: TaskVerdict) -> dict:
    """Map a verdict to the concrete pipeline choices (the §11 decision tree)."""
    base = {"forecaster": "HistGBM + ElasticNet ensemble",
            "uncertainty": "split-conformal + OOD support inflation"}
    if verdict.primary == "timeseries":
        base.update(validation="temporal/blocked split",
                    features="lag + rolling + time features",
                    causal="Granger -> stability-selected lagged graph (PCMCI optional)",
                    intervention="graph-constrained + structural roll-forward")
    elif verdict.primary == "panel":
        base.update(validation="grouped-by-entity + temporal split",
                    features="per-entity lags + entity/time fixed effects",
                    causal="within-entity Granger + stability selection",
                    intervention="graph-constrained; causal-forest/DoubleML if treatment")
    elif verdict.primary == "treatment":
        base.update(validation="stratified by treatment",
                    features="covariate set (exclude post-treatment)",
                    causal="S/T/X-learner, DoubleML, propensity diagnostics",
                    intervention="estimated CATE; conformal/bootstrap effect intervals")
    else:  # tabular
        base.update(validation="stratified k-fold",
                    features="raw + interactions (guard leakage)",
                    causal="partial-corr skeleton + DAG learning (NOTEARS optional), stability",
                    intervention="feature-replacement + bad-control analysis")
    if verdict.anonymized:
        base["note"] = "anonymized: rely on statistical structure only; no domain overclaiming"
    return base
