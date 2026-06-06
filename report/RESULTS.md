# Recorded Results — all checks & experiments

Every measurement we ran, what it tested, the number, and the decision. **14 checks total.**
Adopted = changed the submission; Rejected/No-change = baseline kept (negative results are
credibility, not failure). All on real challenge data unless noted.

---

## A. Audits (diagnostic — informed the approach)

### 1. Gate-1 data audit
- train 85,340 (51,722 labeled / approved+matured); validation 4,489 (2,551 labeled); test 8,817 (0 labeled).
- Observed default rate: 17.4% (train), 20.6% (validation).
- `days_to_default`: median 36, **21% after day 60**, max 90 → justifies the 90-day / 13-week survival model.
- Largest approved-vs-declined selection gap (|SMD|): `prior_underwriter_score` **4.18** (dominant).
- **Use:** set the survival-model horizon and flagged selection bias.

### 2. Task 0 — selection / positivity check
- `e(x)` AUC **1.000 WITH** `prior_underwriter_score`; **0.720 WITHOUT** it → the 1.000 is tautological (the prior score *is* the decision basis), not genuine-feature determinism.
- Corrected no-overlap fraction (genuine-feature space): **~0** at e<0.05 (NOT 43%).
- `prior_underwriter_score` redundant for risk: risk AUC **0.773 with vs 0.774 without**.
- Decision by prior-score decile (lowest→highest): approve **38% → 91%**, mean PD **0.44 → 0.14** (conservative where support is thin).
- **Decision:** reframed D Section 1 (positivity relocated, not removed); kept conservative-decline policy.

---

## B. Model selection

### 3. Task B — backbone bake-off (internal train time-split, 6 learners)
| backbone | cal-Brier | AUC | ECE | policy NPV |
|---|---|---|---|---|
| logistic PD90 | 0.1329 | 0.777 | 0.035 | — |
| catboost hazard | 0.1340 | 0.778 | 0.038 | 10,362,776 |
| randomforest PD90 | 0.1340 | 0.777 | 0.031 | — |
| **histgbm hazard (incumbent)** | 0.1344 | 0.778 | 0.038 | 10,240,257 |
| lightgbm hazard | 0.1376 | 0.763 | 0.042 | 9,329,533 |
| lightgbm-binary PD90 | 0.1398 | 0.752 | 0.043 | — |
- **Decision:** KEEP HistGBM hazard (no challenger beat it by >2% Brier; CatBoost only +0.3%). Logistic within 1% ⇒ no complex leakage. **Backbone frozen.**

---

## C. Causal validation (Deliverable C)

### 4. Task 1 — semi-synthetic C validation (known ground-truth effect)
- Truth slope **+0.0893** per \$10k; disciplined plug-in **+0.0842** (within 6%); naive perturbation **+0.0155** (5.8× too flat).
- MAE vs truth: disciplined **0.0181** vs naive **0.1296** → **7.2× closer**. Sign recovery: yes. DML cross-check **+0.096**.
- **Decision:** confirms plug-in ≠ naive; figure → D Section 3.

### 5. Task 4 Part 2 — per-feature DML cross-check (real data)
- Sign agreement plug-in vs doubly-robust DML: **100% on the 5 responsive (material-effect) features** (26% across all 23, the rest near-zero on both → noise).
- Magnitudes agree: invoice delinquency +0.63 vs +0.55; credit utilization +0.12 vs +0.29; revenue volatility +0.05 vs +0.08; revenue trend −0.017 vs −0.025.
- **Decision:** credibility evidence → D Section 3; submitted value stays the plug-in.

---

## D. Economics & decision

### 6. Economics fix — realized NPV + empirical LGD (vs stylized formula)
- Empirical loss-given-default by default week: week-1 **−0.69 R**, week-6 −0.21 R, week-13 **+0.09 R**; repaid +0.0875 R.
- Realized validation profit: **our E[NPV]>0 = \$2.68M** vs approve-all \$0.61M (**+340%**); oracle threshold \$2.70M (**within 1.0%**).
- **Decision:** ADOPTED (closed the gap to oracle from 2.4% → 1.0%). Later aligned t>60 cap to `D·(min(t,60)−1)` to match the printed formula exactly.

### 7. E2 — decision-policy λ sweep (realized validation profit)
| λ | profit | approve |
|---|---|---|
| **0.000 R** | **\$2,678,370** | 76.6% |
| 0.005 R | \$2,680,082 (+0.06%) | 74.9% |
| 0.010 R | \$2,667,317 | 74.7% |
| 0.020 R | \$2,533,031 | 66.7% |
- **Decision:** KEEP λ=0 (the +0.06% is noise, not a robust gain).

---

## E. Calibration & intervals

### 8. Calibration — isotonic point calibration (labeled validation, cross-fit)
- ECE **0.0258 → 0.0182**; Brier 0.1346 → 0.1359 (tiny cost for better calibration).
- **Decision:** ADOPTED.

### 9. E3 — per-age trajectory calibration (predicted vs realized at each age week)
- Max gap across all 13 age weeks: **0.007** (e.g. age 4: pred 0.080 vs real 0.087). Already near-perfect.
- **Decision:** NO CHANGE (would add risk for ~0 gain). Confirms S_traj + E[NPV] timing.

### 10. E4 / 2C — A-interval coverage vs width (binned, labeled validation)
| width-mult | coverage | mean width |
|---|---|---|
| 0.5 | 1.00 | 0.078 |
| 1.0 | 1.00 | 0.116 |
| 2.0 | 1.00 | 0.191 |
- 100% coverage even at the narrowest → intervals were needlessly wide (S_cal penalizes width).
- **Decision:** ADOPTED tightening — A width **0.222 → 0.134 → 0.114**; harness later confirmed **exactly 0.900** coverage on the time-split.

### 11. B / C interval method
- B: proper binomial CI on the cohort fraction → mean width **0.031** (monotone point + bounds).
- C (Task 4.1): overlap-scaled → mean width **0.50 → 0.166** (tight in-support, wide off-support); point PDs unchanged.
- **Decision:** ADOPTED.

---

## F. Other tuning experiments

### 12. Task 3 — empirical-Bayes cohort shrinkage for B
- Approved loans per cohort: **559–778** (all cohorts thick) → shrinkage weight ≈ 0.93 (light); week-13 CDR ~14%.
- **Decision:** ADOPTED as a safe no-harm stabilizer (monotone, validator PASS).

### 13. E6 — credibility-gap features (stated − observed revenue) [HARNESS]
| metric | baseline | +credibility |
|---|---|---|
| **S_P&L** | \$12,464,533 | \$12,237,217 (**−1.82%**) |
| AUC | 0.7749 | 0.7741 |
| cal coverage | 0.900 | 0.900 |
- **Decision:** REJECTED (hurts P&L and AUC).

### 14. E1 — daily/finer timing
- Not built. Analytic bound: B needs weekly output; weekly E[NPV] already within 1% of oracle ⇒ entire S_P&L headroom ≤1%, at large cost/risk.
- **Decision:** DEFERRED (≤1% upside, fails the "clear win" bar).

---

## Scorecard summary (what shipped)
- Validator: **PASS 0 errors / 0 warnings**.
- A: PD AUC ~0.77, ECE 0.018, interval coverage 0.90; E[NPV] policy +340% vs approve-all, within 1% of oracle.
- B: monotone 13×13, per-age calibration gap ≤0.007, week-13 CDR ~14%.
- C: plug-in 7.2× closer to truth than naive; 100% DML sign agreement on material features.
- **Net: 6 adopted, 4 rejected/no-change, 2 audits, 1 model-selection, 1 deferred.**

---

## 15. Rolling-origin time-blocked CV (3 folds) — robustness of decisions
Temporally-correct CV (random k-fold would leak future->past). Error bars on the baseline
and a re-test of E6 across folds.

Baseline (mean +/- std over 3 later-block folds):
- AUC 0.771 +/- 0.008 ; ECE 0.018 +/- 0.007 ; traj_MAE 0.017 +/- 0.006 ;
  interval coverage 0.930 +/- 0.050 ; approve 0.821 +/- 0.022.

E6 credibility features per fold: +0.00% / -2.30% / +2.97% -> **+0.23% +/- 2.16%**.
- The single-split -1.82% was NOISE; across folds the mean effect is ~0 with high variance.
- **Decision:** KEEP BASELINE (confirmed; reason corrected to "no reliable signal").
- Also confirms the bake-off: backbone Brier gaps (0.1329 vs 0.1344) are smaller than the
  fold-to-fold AUC noise (+/-0.008), so "no clear challenger" is robust, not a single-split artifact.

**Updated tally: 15 checks — 6 adopted, 5 rejected/no-change, 2 audits, 1 model-selection, 1 deferred.**

---

## 16. Real-data interventional sanity check (perturb_sanity.py)
Sweep one feature p10->p90 on 1,500 real validation applicants (recompute descendants),
observe the C-model's mean PD response. Cannot score vs truth (counterfactuals unobservable
on real data) -> checks RESPONSE DIRECTION only.
- invoice delinquency 0.108->0.352 (+); utilization 0.172->0.288 (+); revenue volatility
  0.195->0.300 (+); overdrafts 0.231->0.253 (+); requested_amount 0.288->0.311 (+);
  revenue trend 0.258->0.206 (-); cash balance 0.241->0.241 (-).
- **7/8 features move the economically expected direction.** The miss (stated_time_in_business,
  0.240->0.241) is a near-zero-effect weak feature -> the documented tree-flatness limitation, not
  a sign error. Combined with #4 (synthetic accuracy) and #5 (DML), causal C is validated 3 ways.

**Updated tally: 16 checks.**

---

## 17. S_cal width hedge (Phase 2) + D framing fixes (Phase 1)
A-interval width sweep on the late time-split (one training run):
- mult 0.75 -> coverage 0.700 (under-covers), width 0.097
- **mult 1.00 (current) -> coverage 0.900, width 0.123 (target hit at min width)**
- mult 1.25 -> coverage 1.000 (needlessly wide), width 0.149
- **Decision:** KEEP current A interval (exactly 0.90 at minimum width; tightening under-covers).

D framing (S_write 0.15, no model risk): sharpened IPW degeneracy ("propensity ~1 on approved
-> weights collapse to near-constant -> diagnostic not de-biaser"); positivity stated as the
~43% never-approved tail (the forbidden "no overlap ~0" claim removed); score-alignment
paragraph at top; C 7.2x claim scoped to responsive features. PDF 4 pages, validator PASS.

**Final tally: 17 checks. Model FROZEN. A/B/C unchanged since golden; only D text improved.**

---

## 18. T1 weak-feature gate + T2 logistic-C challenger (REJECTED)
T1 (semi-synthetic, inject a KNOWN moderate effect of a weak feature, measure recovery):
- HistGBM C-model recovers **79%** of the weak-feature slope (MAE 0.0107) -> flatness is CHEAP
  (real-data flatness = those features have ~0 true effect, not a model deficit). GATE: skip T2.
- Side finding: HistGBM recovers the STRONG feature only 38% (tree saturation under do()).

T2-2c (logistic C-model challenger, on the same test-bed):
- Logistic recovers better: weak 96% (MAE 0.0061) vs 79%; strong 134% vs 38%.
- BUT on REAL queries (all in-support, 0% off-support) logistic OVER-extrapolates: 7.6% of
  predictions hit [0,1] extremes, revenue interventions mean 0.95 (vs tree 0.50). Root cause:
  intervening on a bank-feed feature leaves a stale `__isna` flag -> off-distribution input ->
  logistic extrapolates wildly; the tree is robust to it.
- **Decision:** REVERT to golden HistGBM-C (validated 7.2x naive, bounded). Per prime directive
  "if none clearly beats the plug-in, keep the plug-in." A/B/C confirmed identical to golden.

**Final tally: 18 checks. Submission = golden + team name + improved D framing. Model FROZEN.**

---

## 19. Flat-threshold baseline (the brief's named-wrong policy) + logistic-C re-test + D rewrite
Flat-threshold realized validation profit vs our E[NPV]:
- E[NPV]>0 (parameter-free): **$2.62M** (+370% vs approve-all $0.56M)
- fixed thresholds: p<0.20 -> $2.40M, p<0.25 -> $2.66M, p<0.30 -> $2.53M
- best ORACLE flat tau p<0.24 -> $2.66M (needs unseen outcomes); E[NPV] within **1.4%**.
- => every plausible fixed threshold underperforms E[NPV]; the size-aware rule needs no tuning.

Logistic-C re-test with the critique's bank-feed consistency fix (set has_linked_bank_feed=True
on bank-feed interventions): still over-extrapolates (extremes 7.6%->3.8%, revenue interventions
mean 0.95->0.92). Per "reject if weird real-query predictions like 0.95" -> KEEP golden HistGBM-C.

D rewrite (S_write): added flat-threshold baseline; cloglog/logistic-hazard textbook baseline;
softened "100% DML" to sign-only (magnitudes diverge, different estimands); C framed as a
controlled direct effect; reject-inference term; TEMPORAL DRIFT limitation (17.4%->20.6%, test
later, conservative hedge); non-descendant assumption. 4 pages, PASS, team=Falcons.

**Final tally: 19 checks. A/B/C unchanged (golden); D substantially strengthened.**
