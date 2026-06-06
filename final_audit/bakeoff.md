# Task B - backbone bake-off (internal train time-split)

Selection on a held-out LATER time slice of train (fit=28964, cal=7241, late=15517). Official validation reserved for final calibration. Fixed robust defaults, no grid search.

| backbone | kind | cal-Brier | logloss | AUC | ECE | policyNPV |
|---|---|---|---|---|---|---|
| logistic PD90 | binary | 0.1329 | 0.4252 | 0.777 | 0.0346 | nan |
| catboost hazard | hazard | 0.1340 | 0.4276 | 0.778 | 0.0376 | 10,362,776 |
| randomforest PD90 | binary | 0.1340 | 0.4255 | 0.777 | 0.0310 | nan |
| histgbm hazard (incumbent) | hazard | 0.1344 | 0.4263 | 0.778 | 0.0379 | 10,240,257 |
| lightgbm hazard | hazard | 0.1376 | 0.4378 | 0.763 | 0.0420 | 9,329,533 |
| lightgbm-binary PD90 | binary | 0.1398 | 0.4527 | 0.752 | 0.0427 | nan |

**Chosen backbone:** histgbm hazard (incumbent)

**Rationale:** incumbent retained (no hazard challenger beat it by >2% Brier); C reuses it.

Backbone is now FROZEN. Bootstrap kept only for interval uncertainty, not performance.