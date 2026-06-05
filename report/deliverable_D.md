---
geometry: margin=0.75in
fontsize: 11pt
---

# Deliverable D — Technical Writeup

**Team:** _TEAM NAME_

## 1. Problem framing & assumptions violated

We are a small-business lender choosing whom to fund to maximize portfolio NPV, not
to maximize classification accuracy. Three properties of the data break the standard
i.i.d. supervised-learning assumption and shape every design choice.

**Selective labels (the central problem).** Repayment outcomes are observed only for
loans the *prior* underwriter approved and that matured: of 85,340 training rows only
51,722 are labeled; the 33,618 declined applications have no outcome. We must decide on
the full validation+test population (13,306 applicants), so the estimand `P(default | x)`
is not the labeled distribution `P(default | x, approved)`. Critically, selection is on
*observables*: the approval propensity `e(x)=P(approved | x)` is predicted with AUC 1.000
when the prior underwriter's own score is included, and 0.720 from genuine applicant
features alone. We therefore read the 1.000 as near-tautological — the prior *score is the
decision's basis* — not as evidence that approval is deterministic in applicant features.
Because selection is on observables, conditioning on the recorded `prior_underwriter_score`
makes selection ignorable and identifies the outcome model *where overlap exists*. But
conditioning on the score **relocates rather than removes** the positivity problem: approval
is a near-deterministic threshold on that score, so in the feature space the A/B model
actually conditions on, ~43% of the decision set falls in a never-approved score range with no
labeled analogs (`e(x)<0.05` for 0.43 of applicants). No reweighting fixes this; we handle it
by **conservative decline** of that tail (§2), a safe NPV floor of 0 instead of an approval
extrapolated past the data. Dropping the score would restore positivity but reintroduce it as
*omitted confounding*, since the underwriter's signal would then be unobserved. We therefore
state ignorability-given-the-score as an *assumption* (the underwriter may also have used
private signal), backed by the §3 sensitivity analysis, not as a fact proven by the propensity
AUC.

**Right-censoring / timing.** Default is defined over a 90-day window (3 consecutive missed
ACH draws, 6 cumulative misses, or positive balance at day 90) even though the term is 60
days. Median `days_to_default` is 36 and ~21% of defaults occur after day 60, so a binary
day-90 classifier discards financially decisive timing information. This forces a survival
formulation (§2) shared by Deliverables A and B.

**Informative missingness.** Bank-feed fields are null exactly when no feed is linked (~36%
of applicants); this is signal, not noise, so we encode missingness indicators rather than
impute it away.

## 2. Methodology

**One survival backbone for A and B.** We fit a weekly discrete-time hazard
`h_a(x)=P(default in week a | survived, x)` over 13 weeks (the 90-day window) on
person-period expansions of the labeled loans, using a gradient-boosted classifier with
`age_week` as a feature. Survival is `S(t|x)=prod_{k<=t}(1-h_k)`, default probability is
`PD = 1-S(13 weeks)`, and the cumulative-default curve feeds B directly. We chose this
backbone deliberately because it serves A, B, and C coherently. A controlled, pre-registered
bake-off on an *internal* train time-split (earlier weeks fit, later weeks select; official
validation reserved for calibration) compared six learners; the incumbent histogram-GBM
hazard was retained (calibrated Brier 0.1344; the best challenger, CatBoost, was only 0.3%
better — below our 2% swap threshold). A calibrated logistic regression scored within ~1%
Brier (0.1329), a leakage check that passed: no learner is anomalously strong, so the signal
is largely additive. Feature handling: log1p on money fields, missingness indicators,
GBM-native NaN handling, and a strict split between the risk feature set and the selection
feature set (`prior_decision` and `prior_approved_amount` never enter the risk model).

**Selection correction.** We fit `e(x)` on all 85,340 rows and form stabilized, clipped
inverse-propensity weights for the labeled sample. Given the near-deterministic score-based
selection, these weights primarily *reweight for calibration* and *flag low-overlap*
applicants; they are not a de-biaser (conditioning is). We verified the risk model is not
merely cloning prior policy: dropping the entire prior-underwriter group costs ~0 AUC
(0.773 vs 0.774).

**Economics and decision.** We do not use a stylized accrual formula, because the cure
mechanism means collected draws != (t-1) draws. Instead we compute realized NPV from the
outcome columns and learn loss-given-default empirically by default week: a week-1 default
realizes ~ -0.69 R, a near-day-90 default ~ +0.09 R, and full repayment ~ +0.0875 R. Then
`E[NPV_i] = S(90)*0.0875*R_i + sum_a P(week a)*LGD_a*R_i`, and we **approve iff E[NPV_i] > 0**.
On realized validation profit this policy beats approve-all by +340% and lands within 1.0%
of the oracle single-threshold policy — measured on the labeled (supported) region, which is
the honest caveat. The low-score tail, where support is thin, is declined by construction
(bottom-two score deciles approve at 45% vs 91% in the top decile), i.e. a conservative NPV
floor of 0 beats an unsupported approval.

**Deliverable B.** For each origination cohort week we average `1-S(7a|x)` over our *approved*
cohort-`w` loans, stabilized by empirical-Bayes shrinkage toward the pooled approved-book curve
(strength kappa=50) so thin cohorts borrow strength rather than chase noise; here the
conservative book still leaves every cohort well-populated (>550 approved loans each), so the
shrinkage is light. We enforce non-decreasing cumulative rates by a cumulative-max pass on the
point and both interval bounds; intervals combine model uncertainty with cohort-size sampling
variance. The week-13 cohort default rate is ~14%, consistent with the approved book being
lower-risk than the marginal population.

## 3. Causal reasoning & counterfactual methodology

Deliverable C asks an **individual, conditional** question: `P(Y | do(f=v), X_{-q}=x)` for a
specific applicant — not a population average treatment effect. Under ignorability and
support, conditioning on `X_{-q}` blocks the backdoor paths, so the *identified* estimator is
the **cross-fitted outcome-model plug-in**: evaluate the survival PD model at the intervened
covariate vector. This is why we submit the plug-in rather than DML or a causal forest — those
target a population-average estimand and we use them only as directional cross-checks.

The plug-in is emphatically **not** naive feature perturbation, even though the two share the
same final arithmetic. Our estimator (i) **recomputes deterministic descendants** of the
intervened feature — e.g. `requested_amount_to_observed_revenue` moves when
`requested_amount` is set, and leaving it stale silently discards the effect that flows
through it; (ii) **excludes `prior_underwriter_score` from the C model**, because that score
is a *descendant* of the application features (the underwriter scored them) — freezing it
under `do(f)` is the same class of error as not recomputing a ratio, and the audit shows
excluding it costs ~0 accuracy; (iii) **audits local overlap** and widens intervals (or
shrinks toward the factual PD) off-support, the literal answer to "interventions you never
observed"; and (iv) reports an **unobserved-confounding sensitivity band**.

We validated this on a semi-synthetic system with a *known* effect (default risk driven by
the leverage ratio, plus a small direct effect; labels made selective by an approval rule).
Estimating the dose-response of `do(requested_amount)`:

![C estimator recovers the known effect; naive perturbation does not.](../audit/C_validation.png)

The disciplined plug-in recovers the true slope within 6% (+0.084 vs +0.089 per \$10k) and is
7.2x closer to truth in mean absolute error than naive perturbation, which underestimates the
effect 5.8x because it leaves the stale ratio. This is the difference between observational
association and an interventional prediction made concrete. On the *real* data, grouping the
900 queries by feature, a cross-fitted doubly-robust average-effect estimator (partial-linear
DML, prior-underwriter excluded, approval IPW) agrees in sign with our plug-in on **100% of the
features the model materially responds to** (5/5, with comparable magnitudes: e.g. invoice
delinquency +0.63 vs +0.55, credit utilization +0.12 vs +0.29). The two target different
estimands (population-average vs individual-conditional), so we expect aggregate-direction
agreement only; the submitted per-query value is the conditional plug-in by design.

**Defending drivers to a regulator.** Our stated drivers (leverage, credit utilization,
delinquency, cash-balance signals — see the selection and risk audits) are conditioned-on
observables with stable, signed, monotone relationships to default (e.g. default rate falls
monotonically across prior-score deciles, 0.44->0.14); we exclude the prior decision itself
to avoid laundering historical policy, report risk with and without the prior-underwriter
group, and attach explicit overlap and confounding caveats rather than claiming identified
effects off-support.

## 4. Calibration & uncertainty quantification

Point PDs are calibrated by isotonic regression fit on the labeled validation set, improving
cross-fit ECE from 0.026 to 0.018; the same monotone map rescales the survival curve so A and
B remain consistent. Because the scored target is the latent PD (a number), A's 90% interval is
a **tight epistemic confidence interval on the estimate** — bootstrap-ensemble spread plus a
small floor (mean width ~0.13) — and we deliberately do not inflate it with binary-outcome
variance, which would make it needlessly wide. For B, the scored object is a cohort *fraction*,
so its 90% interval is a proper combination of ensemble spread and binomial sampling over the
`n_w` approved loans in each cohort (mean width ~0.03). C is scored on point accuracy rather
than interval calibration, so there we prioritize the plug-in estimate and only widen intervals
off-support as a transparency flag. The explicit trade-off is to target ~90% coverage at
minimum width where calibration is scored (A, B), and to spend C's budget on point accuracy.

## 5. Limitations & what we'd do differently

Our weakest point is **positivity/ignorability** in the ~43% never-approved-score tail: there
we have no counterfactual evidence and rely on model extrapolation; we mitigate by conservative
decline (a safe NPV floor), but cannot verify ignorability there — it is an assumption with a
sensitivity band, not a fact. Second,
**loss-given-default is noisy**: collected draws before default are not directly observed, so
our empirical LGD-by-week is an approximation (recovery is shrunk toward the week mean). Third,
**C is only partially identified** for features that are themselves mediated; we report the
controlled-direct-effect reading and flag it. Relatedly, the tree plug-in is near-flat for
low-importance features, so counterfactuals on weak drivers are conservative — the DML
cross-check confirms only the material drivers carry effect, but a generator with small true
effects on weak features would expose this. With another day we would: use a smoother outcome
model (or monotone splines) so weak-feature counterfactuals are not flattened; tighten C
intervals on high-overlap queries via conformalized quantile regression; and extend the
DML/causal-survival-forest cross-check to per-feature heterogeneity rather than average slope.
