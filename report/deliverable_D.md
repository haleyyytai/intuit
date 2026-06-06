---
geometry: margin=0.75in
fontsize: 11pt
---

# Deliverable D - Technical Writeup

**Team:** the-prior-distribution-1

## 1. Problem framing & assumptions violated

We are a small-business lender choosing whom to fund to maximize portfolio NPV, not to
maximize classification accuracy. Three properties of the data break the standard
i.i.d. supervised-learning assumption and shape every design choice.

**Selective labels (the central problem).** Repayment outcomes are observed only for loans
the prior underwriter approved and that matured: of 85,340 training rows only 51,722 are
labeled, and the 33,618 declined applications have no outcome. We must decide on the full
validation+test population (13,306 applicants), so the estimand `P(default | x)` is not the
labeled distribution `P(default | x, approved)`. Critically, selection is on observables. The
approval propensity `e(x)=P(approved | x)` is predicted with AUC 1.000 when the prior
underwriter's own score is included, and 0.720 from genuine applicant features alone. We read
the 1.000 as near-tautological: the prior score is the decision's basis, not evidence that
approval is deterministic in applicant features.

Conditioning on the recorded `prior_underwriter_score` therefore identifies the outcome model
where overlap exists, under an ignorability assumption made explicit below. But conditioning on
the score relocates rather than removes the positivity problem. Approval is a near-deterministic
threshold on that score, so in the feature space the A/B model conditions on, about 43% of the
decision set lies in a thin-support prior-score region with little labeled approved analog
(`e(x)<0.05`). No reweighting fixes this. We do not decline the whole region by construction:
the E[NPV] rule approves where supported features justify positive value and declines the rest,
so the book approves about two-thirds of applicants overall. Dropping the score would restore
positivity but reintroduce it as omitted confounding, since the underwriter's signal would then
be unobserved. We therefore state ignorability-given-the-score as an assumption (the underwriter
may also have used private signal), stated explicitly in §3, not as a fact proven by the
propensity AUC.

**Right-censoring / timing.** Default is defined over a 90-day window (3 consecutive missed ACH
draws, 6 cumulative misses, or positive balance at day 90) even though the term is 60 days.
Median `days_to_default` is 36 and about 21% of defaults occur after day 60, so a binary day-90
classifier discards financially decisive timing information. This forces a survival formulation
(§2) shared by Deliverables A and B.

**Informative missingness.** Bank-feed fields are null exactly when no feed is linked (about 36%
of applicants); this is signal, not noise, so we encode missingness indicators rather than
impute it away.

## 2. Methodology

![**Figure 1.** One IPW-weighted weekly survival backbone, built around the five scored components: the E[NPV] rule sets funding (S_P&L, 0.30); the same hazard forecasts the cohort trajectory (S_traj, 0.25); isotonic calibration sets the 90% intervals (S_cal, 0.20); a prior-underwriter-free do(f=v) plug-in answers the counterfactuals (S_C, 0.10).](../audit/pipeline.png)

**One survival backbone for A and B.** We fit a weekly discrete-time hazard
`h_a(x)=P(default in week a | survived, x)` over 13 weeks (the 90-day window) on person-period
expansions of the labeled loans, using a gradient-boosted classifier with `age_week` as a
feature. Survival is `S(t|x)=prod_{k<=t}(1-h_k)`, default probability is `PD = 1-S(13 weeks)`,
and the cumulative-default curve feeds B directly. We chose this backbone because it serves A,
B, and C coherently. A pre-registered bake-off on an internal train time-split (earlier weeks fit, later weeks
select; official validation reserved for calibration) compared six learners. The incumbent
histogram-GBM hazard was retained (calibrated Brier 0.1344; the best challenger, CatBoost, was
only 0.3% better, below our 2% swap threshold). A calibrated binary logistic PD90 scored within
about 1% Brier of the backbone and serves as our simple-model leakage check: it is not
anomalously stronger, so the signal is largely additive. We also ran the textbook credit-risk
baseline, a pooled logit / complementary-log-log discrete-time hazard on the same person-period
expansion. It is marginally better calibrated (cloglog cal-Brier 0.13288 vs the GBM's 0.13437),
but by about 1% relative, below our 2% swap bar. We kept the HistGBM hazard for its native
missingness handling, and retained the GLM hazard as a calibrated reference. Features: log1p
money fields, missingness indicators, GBM-native NaN handling; `prior_decision` and
`prior_approved_amount` never enter the risk model.

**Selection correction.** We fit `e(x)` on all 85,340 rows and form stabilized, clipped
inverse-propensity weights for the labeled sample. Because approval is nearly deterministic in
the prior score, the estimated propensity is about 1 on every labeled (approved) loan, so the
weights collapse to a near-constant: IPW acts as a support/overlap diagnostic, not a de-biaser.
The selection correction rests on conditioning, and is limited in the never-approved tail. We
verified the risk model is not merely cloning prior policy: dropping the entire prior-underwriter
group costs about 0 AUC (0.773 vs 0.774).

**Economics and decision.** We do not use a stylized accrual formula, because the cure mechanism
means collected draws differ from (t-1) draws. Instead we compute realized NPV from the outcome
columns and learn loss-given-default empirically by default week: a week-1 default realizes about
-0.69 R, a near-day-90 default about +0.09 R, and full repayment about +0.0875 R. Then
`E[NPV_i] = S(90)*0.0875*R_i + sum_a P(week a)*LGD_a*R_i`, and we approve iff E[NPV_i] > 0.
On realized validation profit this parameter-free policy earns \$2.62M and beats approve-all
(\$0.56M) by +370%. It also beats every plausible fixed PD threshold, the policy the brief names
as wrong (p<0.20 gives \$2.40M, p<0.30 gives \$2.53M), landing within 1.4% of the best
oracle-tuned threshold (p<0.24, \$2.66M) that a flat policy could only find with the unseen
outcomes. That gap is the brief's intended insight: low PD is not profitability, and a size-aware
NPV rule needs no tuning. All figures are on the labeled (supported) region, the honest caveat.
In the thin-support low-score region the rule funds only where E[NPV]>0 and declines the rest (a
conservative NPV floor of 0 beats an unsupported approval); this is the standard reject-inference
posture for selective labels.

**Deliverable B.** For each origination cohort week we average `1-S(7a|x)` over our approved
cohort-`w` loans, stabilized by empirical-Bayes shrinkage toward the pooled approved-book curve
(strength kappa=50) so thin cohorts borrow strength rather than chase noise. Here the
conservative book still leaves every cohort well-populated (>550 approved loans each), so the
shrinkage is light. We enforce non-decreasing cumulative rates by a cumulative-max pass on the
point and both interval bounds; intervals combine model uncertainty with cohort-size sampling
variance. The week-13 cohort default rate is about 14%, consistent with the approved book being
lower-risk than the marginal population.

## 3. Causal reasoning & counterfactual methodology

Deliverable C asks `P(Y | do(f=v), X_{-q}=x)`, an individual estimand that holds all other
observed covariates at their values. As written this is a per-unit controlled direct effect, not
a total effect: any true mediating path from `f` through another observed covariate is held
fixed. Under conditional ignorability and overlap, conditioning on `X_{-q}` blocks the backdoor
paths, so the identified estimator is the cross-fitted outcome-model plug-in with deterministic
descendants recomputed, evaluating the PD model at the intervened covariate vector. This is why
we submit the plug-in rather than DML or a causal forest, which target a population-average
effect and serve only as cross-checks.

The plug-in is not naive feature perturbation, though the two share the same final arithmetic.
Our estimator (i) **recomputes deterministic descendants** of the intervened feature (e.g.
`requested_amount_to_observed_revenue` moves when `requested_amount` is set; leaving it stale
discards the effect that flows through it); (ii) **excludes `prior_underwriter_score` from the C
model**, because that score is a descendant of the application features (the underwriter scored
them), so freezing it under `do(f)` is the same class of error as not recomputing a ratio, and
the audit shows excluding it costs about 0 accuracy; (iii) **audits local overlap** and widens
intervals off-support, addressing positivity / thin support; and (iv) **states the
unobserved-confounding assumption explicitly** as the key untested assumption. That is a distinct
threat from positivity which an overlap band does not bound; a formal E-value sweep is the
natural next step, not run here.

We validated this on a semi-synthetic system with a known effect (default risk driven by the
leverage ratio, plus a small direct effect; labels made selective by an approval rule).
Estimating the dose-response of `do(requested_amount)`:

![**Figure 2.** Semi-synthetic dose-response under do(requested_amount): the disciplined plug-in (green) tracks the known truth (black) while naive perturbation (red) stays flat. Right panel: observed support. Plug-in MAE 0.018 vs naive 0.130.](../audit/C_validation.png)

The disciplined plug-in recovers the true slope within 6% (+0.084 vs +0.089 per \$10k) and is
7.2x closer to truth in mean absolute error than naive perturbation, which underestimates the
effect 5.8x because it leaves the stale ratio. This is the difference between observational
association and an interventional prediction. On the real data, grouping the 900 queries by
feature, a cross-fitted doubly-robust average-effect estimator (partial-linear DML,
prior-underwriter excluded, approval IPW) agrees in sign with our plug-in on all 5 features the
model materially responds to. We claim sign agreement only: the magnitudes differ (e.g. credit
utilization +0.12 plug-in vs +0.29 DML), which is expected because DML targets a population-average
effect while the plug-in is an individual conditional one. They are different estimands, so we do
not read DML as validating our magnitude. The submitted per-query value is the conditional
plug-in by design.

**Defending drivers to a regulator.** Our stated drivers (leverage, credit utilization,
delinquency, cash-balance signals; see the selection and risk audits) are conditioned-on
observables with stable, signed, monotone relationships to default (e.g. default rate falls
monotonically across prior-score deciles, 0.44 to 0.14). We exclude the prior decision itself to
avoid laundering historical policy, report risk with and without the prior-underwriter group, and
attach explicit overlap and confounding caveats rather than claiming identified effects
off-support.

## 4. Calibration & uncertainty quantification

Point PDs are calibrated by isotonic regression fit on the labeled validation set, improving
cross-fit ECE from 0.026 to 0.018; the same monotone map rescales the survival curve so A and B
remain consistent. Because the scored target is the latent PD (a number), A's 90% interval is a
tight epistemic confidence interval on the estimate: bootstrap-ensemble spread plus a small floor
(mean width about 0.13). We do not inflate it with binary-outcome variance, which would make it
needlessly wide. On the late time-split this width achieved 0.900 binned coverage at the minimum tested safe
setting; tightening (x0.75) under-covered (0.70), and widening (x1.25) only added width (1.00).
For B, the scored object is a cohort fraction, so its 90% interval is a proper combination of
ensemble spread and binomial sampling over the `n_w` approved loans in each cohort (mean width
about 0.03). C is scored on point accuracy rather than interval calibration, so there we
prioritize the plug-in estimate and only widen intervals off-support as a transparency flag. The
trade-off is to target about 90% coverage at minimum width where calibration is scored (A, B),
and to spend C's budget on point accuracy.

## 5. Limitations & what we'd do differently

**Temporal drift (first and sharpest).** The observed default rate rises from 17.4% (train) to
20.6% (the later validation block), and the test window is later still. We calibrate isotonic on
validation, the most recent labeled block, so the train-to-validation drift is absorbed; but if
the upward trend continues into test, our PDs will under-predict, which hurts P&L (we fund loans
riskier than we think) and calibration (intervals miss low) together. 5-fold CV (random and
time-blocked) shows the backbone is stable (AUC 0.77 +/- 0.01 both ways) but does not correct a
trend: the latest rolling folds are measurably harder (Brier about 0.11 to 0.13). We do not claim
the drift is removed; a conservative production hedge would nudge the global PD level toward the
most recent observed rate.

**Positivity / reject inference.** About 43% of the decision set is a thin-support prior-score
region with little labeled analog. The E[NPV] rule declines its weakest-support part (the book
approves about two-thirds overall), not as profit-maximizing but as declining where we cannot
evaluate. Two further caveats: loss-given-default is noisy (collected draws before default are
not directly observed, so our empirical LGD-by-week is an approximation), and C is a controlled
direct effect (we treat the remaining covariates as non-descendants of the intervened feature
except the deterministic ratio, so if true mediating paths exist, e.g. revenue moving overdrafts,
we estimate a direct effect through a partially-blocked path, not a total effect).

**Counterfactuals on weak features.** The tree plug-in is near-flat for low-importance features,
so counterfactuals on weak drivers are conservative; the DML cross-check confirms only the
material drivers carry effect. We tested whether this flatness costs accuracy. On a semi-synthetic
test-bed injecting a known weak-feature effect, the tree recovered 79% of it (so the flatness is
bounded). A smoother logistic C-model recovered more on the test-bed but over-extrapolated on the
real in-support queries (PD around 0.91 under revenue interventions, verified manifold
extrapolation rather than a missingness-flag artifact), further from plausible truth than the
bounded tree, so we kept the tree.

## References

(Optional; on page 5, excluded from the 4-page body limit.)

1. Prentice, R.L. & Gloeckler, L.A. (1978). Regression analysis of grouped survival data with application to breast cancer data. *Biometrics* 34(1). [discrete-time / cloglog hazard]
2. Singer, J.D. & Willett, J.B. (2003). *Applied Longitudinal Data Analysis*. Oxford University Press. [person-period discrete-time survival]
3. Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. *KDD*. [isotonic calibration]
4. Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal* 21(1). [partial-linear DML]
5. VanderWeele, T.J. & Ding, P. (2017). Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine* 167(4). [unobserved-confounding sensitivity]
6. Hand, D.J. & Henley, W.E. (1997). Statistical classification methods in consumer credit scoring. *JRSS-A* 160(3). [reject inference / credit scoring]
