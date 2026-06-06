# SMB Underwriting Challenge — Full Technical Reference

Definitive record of methodology, algorithms (with exact hyperparameters), and every result.
Companion to `RESULTS.md` (the 15-check experiment log). Repo: github.com/haleyyy2001/intuit.

---

## 0. Where everything lives

| Artifact | Path |
|---|---|
| Pipeline (produces A/B/C) | `project/src/challenge_pipeline.py` |
| Economics (NPV/LGD) | `project/src/economics.py` |
| Survival model | `project/src/survival.py` (+ inline `WeeklyHazard` in pipeline) |
| Calibration | `project/src/calibration.py` |
| Decision policy | `project/src/decision.py` |
| Data audit (Gate 1) | `project/src/audit.py` |
| Backbone bake-off | `project/src/bakeoff.py` |
| C semi-synthetic validation | `project/src/c_validation.py` |
| DML cross-check | `project/src/dml_crosscheck.py` |
| Diagnostics (per-age/coverage/lambda) | `project/src/diagnostics.py` |
| Scoring harness + rolling CV | `project/src/harness.py` (`--cv`) |
| **All result numbers (15 checks)** | `project/report/RESULTS.md` |
| Writeup (source) | `project/report/deliverable_D.md` |
| **Submission files** | `challenge/.../submission/submission_{A,B,C}.csv`, `submission_D_writeup.pdf` |
| Audit tables/figures | `challenge/.../audit/` (bakeoff.md, dml_crosscheck.md, C_validation.png, task0_selection_check.md, *.csv) |
| Locked baseline | git tag `submission-baseline` + `challenge/.../submission/_baseline/` |

Run end-to-end: `python src/challenge_pipeline.py` (~45s) then
`python validate_submission.py submission` → must print `PASS`.

---

## 1. Problem & data

Small-business lender. Decide whom to fund to maximize portfolio NPV; forecast default timing;
answer causal what-ifs. **Score: S = 0.30 S_P&L + 0.25 S_traj + 0.20 S_cal + 0.10 S_C + 0.15 S_write.**

| split | rows | cols | labels |
|---|---|---|---|
| train | 85,340 | 44 | 51,722 (prior-approved + matured); 33,618 declined are blank |
| validation | 4,489 | 44 | ~2,551 labeled |
| test | 8,817 | 44 | outcomes withheld |

Decision set = validation + test = **13,306**. Loan terms (fixed): 60-day term, **35% APR, 3% fee**,
daily ACH draws. **Default** = 3 consecutive missed draws OR 6 cumulative OR balance>0 at day 90
(90-day window = 13 weeks). 8 column groups; bank-feed null when no feed (~36%).

**Data usage:** train models on `train` (e(x) on all rows; survival/LGD/C-model on the 51,722
labeled) → calibrate on labeled `validation` → predict on `validation+test` → only Intuit scores
held-out `test`. Model *selection* used an internal time-split of train (not official validation).

---

## 2. Architecture (selection-aware causal survival policy)

One weekly survival backbone feeds A, B, C. Economics + an E[NPV] rule turn PD into approve/decline.
Isotonic calibration + empirical intervals handle uncertainty. Causal C is a disciplined plug-in.

```
features -> approval model e(x) -> IPW weights
         -> empirical LGD by default week
         -> weekly discrete-time hazard ensemble (A/B) ; C-model ensemble (excl prior-uw)
         -> isotonic calibration (labeled validation)
         -> A: PD + E[NPV]>0 decision + epistemic interval
         -> B: cohort x age CDR over approved set, EB-shrunk, monotone
         -> C: plug-in do(f=v), descendants recomputed, overlap-scaled interval
```

---

## 3. Algorithms & exact hyperparameters

### 3.1 Feature engineering (`build_features`)
- **log1p** on money fields: stated_annual_revenue, requested_amount, existing_debt_obligations,
  observed_monthly_revenue_avg_3mo, prior_loans_amount_total.
- **signed-log** (`sign(x)·log1p(|x|)`) on observed_cash_balance_p10 (can be negative).
- **missingness indicators** (`<col>__isna`) for every column with >5% nulls (8 of them).
- **NaN kept** (HistGradientBoosting handles natively — no imputation).
- **Excluded from features:** ids (business_id, applicant_id), all 6 outcome columns,
  the selection columns prior_decision + prior_approved_amount, and application_timestamp
  (used only to assign cohort week). 33 base features + 8 missingness flags.

### 3.2 Approval-propensity model e(x) (selection)
- Algorithm: `HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.08,
  l2_regularization=1.0, random_state=0)`.
- Target: `prior_decision == 1`. Trained on all 85,340 rows. Output clipped to [1e-3, 1-1e-3].
- **IPW weights** on labeled rows: `w = clip(P(A=1)/e(x), 0.1, 10.0)` (stabilized + clipped).
  Purpose: calibration-reweighting + overlap flag, NOT de-biasing (positivity is the real limit).

### 3.3 Survival backbone — weekly discrete-time hazard (`WeeklyHazard`)
- **Person-period expansion:** each labeled loan -> one row per week it is at risk (1..default week,
  or 1..13 if it never defaults); binary target = 1 in the default week, else 0. `age_week` is a
  feature. Default week = `ceil(days_to_default/7)` clipped to [1,13]. ~614k person-period rows.
- **Learner:** `HistGradientBoostingClassifier(max_depth=3, max_iter=350, learning_rate=0.07,
  l2_regularization=1.0)`, `sample_weight = IPW`.
- **Ensemble:** 5 bootstrap resamples of the person-period rows (for epistemic uncertainty).
- **Outputs:** hazard h_a per week; survival `S(t) = prod_{k<=t}(1-h_k)`; **PD90 = 1 - S(13 weeks)**;
  cumulative default curve `CDR_a = 1 - S_a`.
- **C-model:** identical recipe but feature set **excludes prior_underwriter_score** (it is a
  descendant of the application features); 3-bootstrap ensemble.

### 3.4 Economics & loss-given-default (`economics.py`)
- Per the printed brief: `D = R·(1 + 0.35·60/365)/60 = 0.017625·R`; `REPAID_RATIO = 0.03 + 0.35·60/365 = 0.0875`.
- Realized NPV ratio for a default on day t: `0.03 + collected/R + rec/R - 1`, with
  `collected = D·clip(t-1, 0, 59)` (caps draws at min(t,60)-1 so post-term defaults don't over-collect).
- **Empirical LGD by default week** (`fit_lgd_by_week`): mean realized NPV ratio over training
  defaulters in each week 1..13 (missing weeks interpolated). Uses actual `days_to_default` and
  `final_recovered_amount`. Result: week-1 -0.69R, week-6 -0.21R, week-13 +0.09R.
- `E[NPV] = S(90)·0.0875·R + sum_a P(default week a)·LGD_a·R`.

### 3.5 Decision policy (`decision.py`)
- **Approve iff E[NPV] > 0** (lambda=0 chosen by sweep; see RESULTS #7). No PD threshold.

### 3.6 Calibration (`calibration.py`)
- **Isotonic regression** (`IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)`) fit on
  labeled validation: raw PD -> calibrated PD. The monotone map is also applied as a multiplicative
  `factor = pd_cal/pd_raw` to the hazard curve and CDR so A/B stay consistent.
- ECE measured by 10-bin expected calibration error.

### 3.7 Uncertainty intervals
- **A (PD):** `half = 1.645·sd_ensemble + 0.01` -> epistemic CI on the PD estimate (tight; S_cal
  penalizes width). Clipped [0,1], lower<=pd<=upper enforced.
- **B (cohort fraction):** `half = 1.645·sqrt(sd_ensemble^2 + p(1-p)/n_w)` (epistemic + binomial over
  the n_w approved loans), cumulative-max applied to point AND both bounds (monotone in age).
- **C:** `half = 1.645·sqrt(sd_ensemble^2 + (0.15·(1-overlap))^2)`; overlap = `clip(1 - dist/3, 0, 1)`,
  dist = |value - feature_median| / feature_IQR. Tight in-support, wide off-support.

### 3.8 Deliverable B — empirical-Bayes cohort shrinkage
- Pooled approved-book curve `g_a = mean over all approved loans of CDR_a`.
- `CDR_{w,a} = (n_w/(n_w+50))·p_cohort + (50/(n_w+50))·g_a` (kappa=50). Thin cohorts borrow strength.
- Then cumulative-max for monotonicity.

### 3.9 Deliverable C — disciplined causal plug-in
1. Set `do(feature = value)` on the applicant row.
2. **Recompute deterministic descendants**: `requested_amount_to_observed_revenue =
   requested_amount / observed_monthly_revenue_avg_3mo` (else the do-estimate keeps a stale ratio).
3. Predict PD with the **C-model** (excludes prior_underwriter_score, cross-fitted).
4. Calibrate via the shared isotonic map; overlap-scaled interval.
- **Identification:** under ignorability + support, conditioning on X_{-q} blocks backdoor paths, so
  the identified estimator is the outcome-model plug-in (NOT DML/causal forest, which estimate a
  population-average effect — used only as cross-checks).

### 3.10 DML cross-check (`dml_crosscheck.py`)
- Per intervened feature, cross-fitted **partial-linear DML**: residualize Y and f on X_{-f} with
  HistGBM regressors, theta = weighted regression of Y-residual on f-residual (IPW-weighted, 2-fold
  cross-fit, prior-underwriter excluded). Compared in sign/magnitude to the plug-in's mean slope.

### 3.11 Model selection (`bakeoff.py`) and evaluation (`harness.py`)
- **Bake-off:** internal train time-split (earlier fit / mid calibrate / later select). 6 learners,
  fixed robust defaults (LightGBM num_leaves 47, lr 0.04, min_child 80, subsample/colsample 0.8,
  reg_lambda 1; CatBoost depth 6, iter 400, lr 0.04). Primary metric = calibrated Brier.
- **Harness:** trains on early train, calibrates on mid, evaluates on late; reports S_P&L (realized
  NPV), trajectory MAE, AUC, ECE, interval coverage. `--cv` = 3 rolling-origin (time-blocked) folds.

### 3.12 Tuned-scalar budget (4)
Isotonic calibration map · decision lambda=0 · EB kappa=50 · off-support scale 0.15. Plus fixed
floors (interval floor 0.01). No grid search on official validation.

---

## 4. All results (see RESULTS.md for the full log)

### Headline (what shipped)
- **Validator: PASS, 0 errors, 0 warnings.**
- **A:** PD AUC ~0.77 (0.771 ± 0.008 over rolling folds); ECE 0.018; interval coverage 0.90–0.93;
  **E[NPV] policy: +340% vs approve-all, within 1.0% of the oracle threshold** (realized validation).
- **B:** 169-row monotone grid; per-age calibration gap <= 0.007; week-13 CDR ~14%.
- **C:** plug-in recovers a known synthetic effect within 6% (slope +0.084 vs truth +0.089),
  **7.2x closer to truth than naive** (MAE 0.018 vs 0.130); 100% DML sign agreement on material features.

### Selection audit (Task 0)
- e(x) AUC **1.000 with** prior_underwriter_score, **0.720 without** -> the 1.000 is tautological.
- Risk AUC with/without prior-underwriter group: **0.773 vs 0.774** (redundant).
- Approve by prior-score decile: **38% (lowest) -> 91% (highest)**; mean PD 0.44 -> 0.14.

### Bake-off (cal-Brier, lower better)
logistic 0.1329 | catboost-hazard 0.1340 | rf 0.1340 | **histgbm-hazard 0.1344 (kept)** |
lightgbm-hazard 0.1376 | lightgbm-binary 0.1398. No challenger > 2% better -> incumbent frozen.

### Economics / decision
- LGD by week: w1 -0.69R, w6 -0.21R, w13 +0.09R; repaid +0.0875R.
- Realized validation profit: our E[NPV] $2.68M vs approve-all $0.61M (+340%); oracle $2.70M.
- Lambda sweep: lambda=0 best ($2.678M); 0.005R +0.06% (noise); 0.01R/0.02R worse -> keep 0.

### Calibration / intervals
- ECE 0.0258 -> 0.0182 (isotonic). A width 0.222 -> 0.134 -> 0.114. B width 0.031. C width 0.50 -> 0.166.
- Per-age trajectory gap <= 0.007 across all 13 weeks (already calibrated).

### Rolling-origin CV (3 folds) — robustness
- Baseline: AUC 0.771±0.008, ECE 0.018±0.007, traj_MAE 0.017±0.006, coverage 0.93±0.05, approve 0.82±0.02.
- E6 credibility features: +0.00% / -2.30% / +2.97% -> **+0.23% ± 2.16%** (single-split -1.82% was noise) -> reject.

### Experiment tally (15 checks)
6 adopted · 5 rejected/no-change · 2 audits · 1 model-selection · 1 deferred.
Adopted: economics fix, isotonic calibration, A/B/C interval methods, EB shrinkage, overlap-C.
Rejected/no-change: lambda margin (E2), per-age recalibration (E3, already good), credibility
features (E6), daily timing (E1 deferred), backbone swap (bake-off).

---

## 5. Honest limitations
- **Positivity** in the ~43% never-approved-score tail (conditioning on the prior score): no
  counterfactual evidence there; handled by conservative decline; ignorability is an assumption.
- **LGD** uses week-average recovery (collected draws not directly observed; cure mechanism approximated).
- **C plug-in is near-flat on low-importance features** (tree gradient ~0); material drivers are captured.
- **Reported metrics are on the supported (approved) region**; the true test S_P&L is scored by Intuit
  on the withheld test set.
- **S_cal / S_C exact metrics are unpublished**; interval sizing and C handling are reasoned bets.
