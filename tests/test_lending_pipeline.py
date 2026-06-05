"""Validate the lending pivot: NPV economics, survival model, E[NPV] decision rule.

Run:  python tests/test_lending_pipeline.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lending                                                    # noqa: E402
from lending import (npv_repaid, npv_default, expected_npv_from_hazard,  # noqa: E402
                     LoanTerms, FEATURES)
from survival import DiscreteTimeHazard                           # noqa: E402
from decision import expected_npv, evaluate_policies             # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
ok = []
def check(name, cond, detail=""):
    ok.append(cond); print(f"  [{PASS if cond else FAIL}] {name}{(' — ' + detail) if detail else ''}")

terms = LoanTerms()

# === (1) NPV economics identities =========================================== #
print("[1] NPV economics")
R = 10000.0
repaid = npv_repaid(R, terms)
interest = R * terms.apr * terms.term_days / 365.0
check("repaid NPV = fee + interest", np.isclose(repaid, terms.fee_rate * R + interest),
      f"repaid={repaid:.2f}")
# all-zero hazard => certain repayment => E[NPV] == npv_repaid
h0 = np.zeros(terms.term_days)
check("E[NPV] with zero hazard == repaid NPV",
      np.isclose(expected_npv_from_hazard(R, h0, terms=terms), repaid))
# hazard that forces default on day 1 => E[NPV] == npv_default(day=1)
h1 = np.zeros(terms.term_days); h1[0] = 1.0
check("E[NPV] with day-1 certain default == npv_default(1)",
      np.isclose(expected_npv_from_hazard(R, h1, recovery_rate=0.30, terms=terms),
                 npv_default(R, 1, 0.30, terms)))
# default later is more profitable than default early (more draws collected)
check("later default is less unprofitable than early default",
      npv_default(R, 55, terms=terms) > npv_default(R, 5, terms=terms),
      f"day5={npv_default(R,5,terms=terms):.0f} day55={npv_default(R,55,terms=terms):.0f}")

# === (2) Synthetic portfolio + survival model =============================== #
print("\n[2] discrete-time hazard model recovers timing & risk")
df = lending.generate_portfolio(n=6000, seed=0)
print(f"   portfolio: {len(df)} loans, default rate = {df['y'].mean():.3f}, "
      f"median default day = {int(df.loc[df.y==1,'t'].median())}")
tr, te = df.iloc[:4000].copy(), df.iloc[4000:].copy()
haz = DiscreteTimeHazard(term_days=terms.term_days, model="gbm").fit(tr, FEATURES)
p_def = haz.predict_default_prob(te)

# predicted default prob should rank-correlate with the true per-day hazard
from scipy.stats import spearmanr
rho = spearmanr(p_def, te["_true_hazard"]).correlation
check("predicted P(default) tracks true hazard (Spearman>0.6)", rho > 0.6, f"rho={rho:.3f}")
# CDR curve must be monotonically increasing by week
cdr = haz.predict_cdr(te, weeks=(1, 2, 4, 8)).mean()
print("   mean CDR by week:", {k: round(v, 3) for k, v in cdr.items()})
check("CDR is monotonic non-decreasing in week",
      all(cdr[f"cdr_w{a}"] <= cdr[f"cdr_w{b}"] + 1e-9 for a, b in [(1, 2), (2, 4), (4, 8)]))

# === (3) E[NPV] decision beats threshold policies =========================== #
print("\n[3] E[NPV] policy vs naive threshold policies (portfolio profit)")
H = haz.predict_hazard(te)
enpv = expected_npv(te["loan_amount"].to_numpy(), H)
table = evaluate_policies(te, enpv, p_def)
print(table.to_string(index=False))
best_threshold_profit = table[table["policy"].str.startswith("threshold")]["portfolio_profit"].max()
enpv_profit = float(table[table["policy"] == "E[NPV]>0"]["portfolio_profit"].iloc[0])
check("E[NPV] policy is the most profitable overall",
      table.iloc[0]["policy"] == "E[NPV]>0",
      f"E[NPV]={enpv_profit:.0f} bestThresh={best_threshold_profit:.0f}")
check("E[NPV] policy beats best threshold policy",
      enpv_profit >= best_threshold_profit)

print("\n" + "=" * 60)
print(f"  {sum(ok)}/{len(ok)} checks passed")
print("=" * 60)
sys.exit(0 if sum(ok) == len(ok) else 1)
