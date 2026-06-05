# Explainable ML Prediction Challenge — Causal Forecasting & Uncertainty

Submission for the Intuit Explainable ML Prediction Challenge (NYC Tech Week).

Deliverables: (1) recovered causal structure, (2) intervention forecasts,
(3) calibrated uncertainty estimates, (4) a technical methodology writeup.

## Approach (balanced hybrid)

A strong predictive backbone (regularized linear + gradient-boosted trees) for
forecasts and intervention scenarios, paired with a defensible causal-discovery
module (Granger/VAR or PCMCI for temporal data; conditional-independence / DAG
learning for tabular data) for structure. Uncertainty via conformal prediction
and/or quantile regression, validated by empirical coverage.

## Layout

```
project/
  data/                  # raw + processed challenge data (gitignored if large)
  notebooks/             # exploratory analysis
  src/
    load_data.py         # I/O, schema parsing
    preprocess.py        # cleaning, missingness, stationarity, splits
    features.py          # feature/lag construction
    causal_discovery.py  # structure recovery + edge confidence
    forecasting.py       # baselines + final forecaster
    uncertainty.py       # conformal / quantile / bootstrap intervals
    evaluation.py        # metrics, calibration diagnostics
    submit.py            # writes submission files + report artifacts
  outputs/
    figures/ tables/ predictions/ causal_graph/
  report/
    methodology.md       # technical writeup
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# end-to-end demo on the synthetic mini-challenge (known ground truth):
python src/submit.py --demo

# once real data is in place:
python src/submit.py --data data/raw/challenge.csv --target <target_col>

# tests / rehearsal:
python tests/test_engine_synthetic.py        # 13 engine unit checks
python tests/dry_run_mini_challenge.py        # 14 full-pipeline checks
```

`submit.py` writes: `outputs/predictions/submission.csv`,
`outputs/tables/{model_comparison,causal_edges,calibration}.csv`,
`outputs/causal_graph/graph.graphml`, and 3 figures.

## Status

**Engine built & verified before data (zero fragile deps — runs on
numpy/pandas/sklearn/statsmodels/networkx).** All 12 `src/` modules import; both
test suites pass; the one-command demo produces all submission artifacts.
Validated on synthetic ground truth: causal recovery finds all true edges;
structural intervention recovers the true effect where naive replacement misses
it 250×; conformal coverage is calibrated. See `report/strategy.md` for the
full validated findings. Awaiting the real dataset to wire `load_data.py`.
