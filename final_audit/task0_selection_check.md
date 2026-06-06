# Task 0 - Selection / positivity check (resolves the AUC=1.000 claim)

## Both AUCs
- e(x) WITH prior_underwriter_score   : AUC = 1.000
- e(x) WITHOUT prior_underwriter_score : AUC = 0.720

The 1.000 is NOT genuine-feature determinism. It is driven entirely by
prior_underwriter_score, which IS the basis of the prior decision (approve ~ 1[score>c]).
So in D write "approval is highly predictable from observed features (AUC 0.72) and
near-deterministic given the prior underwriter's own score (1.000, expected)", NOT
"deterministic on observables".

## Corrected positivity / no-overlap
Using the genuine-feature e(x) (AUC 0.72), the no-overlap fraction of the decision set is:
  e<0.02: 0.000   e<0.05: 0.000   e<0.10: 0.000
=> In applicant-feature space, overlap is ADEQUATE. The earlier "43% no-overlap" was an
artifact of using the score-driven e(x). Positivity is a bounded caveat, not the fatal limit.

## No-overlap decision policy (the score-tail the risk model conditions on)
The risk model conditions on prior_underwriter_score; approval is ~deterministic in it, so
the genuinely unsupported region is the LOW-SCORE tail. The pipeline handles it conservatively
by construction: low score -> high predicted PD -> E[NPV]<0 -> DECLINE. The bottom-2 score
deciles are approved at rate 0.451 (mean PD 0.395).
This is the safe NPV floor (0 beats an unsupported approval). Rationale: we do not approve
into a region with no comparable matured-loan evidence.

## Profit-number caveat (for Section 4)
The "+340% vs approve-all / within 1% of oracle" profit is computed on labeled (approved-only)
validation, which by construction is the SUPPORTED region. It measures the policy where we
have outcomes; it does not claim to cover applicants the prior lender never funded.
