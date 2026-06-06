# Task 1 - semi-synthetic C validation

Known SCM: default depends on leverage ratio (theta=0.45) + small direct effect.

- truth slope per $10k: +0.0893
- naive perturbation slope: +0.0155  (MAE vs truth 0.1296)
- disciplined plug-in slope: +0.0842  (MAE vs truth 0.0181; 7.2x closer)
- DML cross-check slope: +0.0964
- sign recovery (disciplined): True

Naive perturbation leaves the stale requested/revenue ratio and so misses the effect that flows through it; the disciplined plug-in recomputes the descendant and tracks truth. Figure: C_validation.png
