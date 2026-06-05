# Strategy — Plan of Record

**Thesis: "Accurate prediction under causal humility."**

> We build an intervention-aware forecasting system that separates predictive
> accuracy from causal interpretability. A gradient-boosted ensemble provides
> robust forecasts, while a stability-selected causal graph constrains
> intervention reasoning. Uncertainty is calibrated using conformal residuals
> and widened for interventions outside the empirical support of the training
> data.

## Committed stack

| Component | Choice |
|---|---|
| Point forecast | LightGBM (+ ridge/elastic-net) ensemble; horizon-specific if temporal |
| Causal graph | **Stability-selected** lagged graph (bootstrap × windows × subsets × lags); keep stable edges only |
| Intervention logic | **Graph-constrained** simulation: set intervention vars, propagate through graph, never hold post-intervention mediators/bad-controls fixed |
| Uncertainty | Split-conformal residuals + bootstrap ensemble + **OOD inflation** under extrapolative interventions |
| Explainability | Permutation importance + SHAP, framed as *association*, reconciled against the causal edge table |

## Non-negotiables (these are what differentiate us)

1. **Feature eligibility taxonomy** computed before modeling — every column tagged:
   `safe-covariate | intervention | mediator | bad-control | leakage`.
   Two feature sets fall out of this:
   - *Predictive set* — maximizes validation accuracy (may include mediators).
   - *Intervention set* — causally valid; excludes bad-controls & post-intervention vars.
2. **No naive feature replacement** for interventions. Setting X and reading Y off
   the predictive model is wrong when X moves downstream vars. Propagate through
   the graph instead; do not condition on post-intervention colliders.
3. **Stability over single-shot discovery.** Report edge stability under resampling;
   never present one recovered graph as truth. Edge table columns:
   `cause, effect, lag, stability, sign, evidence (Granger/VAR/SHAP/bootstrap)`.
4. **Calibration is reported honestly** — empirical coverage vs. nominal AND mean
   width, per fold and per horizon. Narrow ≠ good.
5. **OOD-aware intervals** — intervention support score (Mahalanobis / density /
   isolation-forest / percentile range). Widen intervals for low-support interventions.
6. **Counterfactual sanity checks** per intervention — sign stability across
   bootstrap models, monotonicity where expected, effect-vs-noise, realistic range.

## Sharpening notes (my additions to the brief)

- **Honesty guardrail on "propagate through the graph":** structural propagation is
  only trustworthy if we actually recover the mediator mechanisms. If discovery is
  weak/unstable, a fully propagated SCM can be *worse* than a careful reduced-form
  model. Plan: implement both (a) graph-propagated intervention and (b) a
  reduced-form "intervention set only" predictive model, and **report where they
  agree/disagree** rather than forcing one. Disagreement is itself an honest finding.
- **Let the data pick the branch.** Temporal → PCMCI/Granger/VAR + lagged graph.
  Pure tabular → conditional-independence / DAG learning (NOTEARS as a *candidate
  generator*, not oracle) + stability selection. Panel → separate entity/time/lag
  effects. Decide AFTER Stage 1 classification; don't pre-commit to lags if i.i.d.
- **Match effort to the official metric.** If the metric scores intervals (pinball,
  CRPS, interval score) we optimize quantiles directly; if it scores points (RMSE/MAE)
  uncertainty is a methodology differentiator, not a leaderboard lever. Confirm first.
- **Leakage audit is gate-zero.** Before any modeling, prove no test info / no
  target-derived columns / correct temporal ordering. A leak invalidates everything.

## Rehearsal results — validated on synthetic ground truth (pre-data)

The full pipeline was rehearsed on a known 5-variable causal SCM
(`src/synthetic.py`; tests in `tests/`). Verified, reproducible findings:

- **Causal recovery:** stability selection recovered **all 4 true directed edges**
  (x1→x2, x4→x3, x2→y, x3→y) at stability 1.0; flagged the indirect x1→y path —
  honestly reported as a pairwise-Granger artifact needing conditional refinement.
- **Intervention (the headline):** for persistent `do(x1 += 1)`, true effect on y
  = **+15.98**. Naive L1 feature-replacement gave **+0.06** (250× under — it is
  blind to the multi-step x1→x2→y path); **L3 structural roll-forward gave +14.0**
  (within 25%). This is the quantified case for *not* doing naive replacement.
- **Uncertainty:** raw split-conformal hit **92.8%** empirical coverage at 90%
  nominal; OOD inflation widened intervals and lifted coverage to 95.2% under
  extrapolation — a deliberate safety trade, not miscalibration.
- **Failure-mode learned:** bagged-tree disagreement does **not** detect OOD
  (trees saturate); OOD must come from the support scorer. Documented, not hidden.

### Literature-driven upgrades (validated)

- **Weighted conformal** (Tibshirani et al. 2019; Lei & Candès 2021) now the
  PRIMARY uncertainty method, replacing the heuristic OOD inflation with the
  principled form: reweight calibration residuals by an estimated density ratio
  `w(x)=dP_test/dP_calib` (cross-fitted gradient-boosted classifier), then take a
  *weighted* quantile. Under an induced covariate shift, plain split-conformal
  **under-covered at 0.845**; weighted-conformal **restored 0.895** by adaptively
  widening (2.24→2.70) exactly in the shifted tail. *Bug found & fixed during
  rehearsal:* a linear density-ratio classifier missed the (tail/variance) shift
  and collapsed weights to ~1 — must use a nonlinear classifier.
- **Conditional Granger = PCMCI+'s MCI logic** now prunes indirect edges. The
  spurious `x1→y` (from the `x1→x2→y` chain) has `p(x1→y | rest)=0.45` and is
  **dropped**, while direct `x2→y`/`x3→y` have `p≈0` and are kept. Pipeline now
  emits `causal_edges.csv` (candidates) AND `causal_edges_direct.csv` (post-MCI).
  Honest residual: one reverse edge `x2→x1` survives the linear conditional test
  (full PCMCI+ would likely remove it) — reported, not hidden.
- **Calibration reported with a baseline:** both plain and weighted conformal in
  `calibration.csv`, with interval score — never a single number.

## Deliverable figures (target 4–6)
causal graph image · edge-stability heatmap · per-target feature importance ·
intervention response curves · calibration plot (coverage vs nominal) ·
residuals-by-horizon · interval-coverage table · model-comparison table.
