"""Reconcile a regime-continuation strategy across twelve US holdouts, 1954-2025.

Reproduces the US application in Tso (2026). For each non-overlapping 72-month
holdout, a three-state relative-return taxonomy (K-means on standardized
stock-minus-bond and bond-minus-cash spreads) is estimated on all earlier
months and frozen. The continuation forecaster holds the policy of last
month's state. Each holdout is then passed to ``reconcile``.

Data: the Welch-Goyal monthly predictor file (Amit Goyal's website). Export
the "Monthly" sheet to CSV and pass its path::

    python examples/us_regime_reconciliation.py PredictorData2025_Monthly.csv

Columns used: yyyymm, CRSP_SPvw (stock), ltr (bond), Rfree (cash).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from alpha_lens import reconcile

ASSETS = ["stock", "bond", "cash"]
POLICY = np.array([[0.70, 0.20, 0.10], [0.20, 0.50, 0.30], [0.55, 0.10, 0.35]])
STATE_NAMES = ["Stock Leadership", "Bond Defensive", "Bond Underperformance"]
# Standardized pre-2020 centers of the global taxonomy (Tso, 2026); used only to
# give US clusters the same names and policy rows.
REFERENCE_CENTERS = np.array([[0.429875, 0.683987], [-1.421696, 0.407572], [0.370442, -0.850111]])
HOLDOUTS = [(f"{y}-01", f"{y + 5}-12") for y in range(1954, 2021, 6)]


def load_welch_goyal(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, na_values=["NaN"])
    raw["month"] = pd.to_datetime(raw["yyyymm"].astype(str), format="%Y%m").dt.strftime("%Y-%m")
    df = raw.rename(columns={"CRSP_SPvw": "stock", "ltr": "bond", "Rfree": "cash"})
    df = df.loc[(df["month"] >= "1926-01") & (df["month"] <= "2025-12"), ["month", *ASSETS]]
    return df.set_index("month").astype(float).dropna()


def frozen_states(r: pd.DataFrame, estimation_end: int) -> np.ndarray:
    """Fit K=3 on rows [0, estimation_end) and label every row with the frozen centers."""
    x = np.column_stack([r["stock"] - r["bond"], r["bond"] - r["cash"]])
    mu, sd = x[:estimation_end].mean(0), x[:estimation_end].std(0)
    z = (x - mu) / sd
    km = KMeans(n_clusters=3, random_state=42, n_init=100, max_iter=1000).fit(z[:estimation_end])
    cost = ((km.cluster_centers_[:, None, :] - REFERENCE_CENTERS[None]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    mapping = np.empty(3, dtype=int)
    mapping[rows] = cols
    return mapping[km.predict(z)]


def main(path: str) -> pd.DataFrame:
    r = load_welch_goyal(path)
    rows = []
    for start, end in HOLDOUTS:
        a, b = r.index.get_loc(start), r.index.get_loc(end) + 1
        states = frozen_states(r, a)
        held = states[a - 1 : b - 1]  # continuation: last month's state, known at t-1
        hold = r.iloc[a:b]
        w = pd.DataFrame(POLICY[held], index=hold.index, columns=ASSETS)
        res = reconcile(
            w,
            hold,
            {"60/40": [0.6, 0.4, 0.0]},
            realized_states=pd.Series(states[a:b], index=hold.index),
            predicted_states=pd.Series(held, index=hold.index),
            periods_per_year=12,
            block_length=6,
            n_boot=10_000,
            seed=20260910,
        )
        bm = res.benchmarks[0]
        et = res.exposure_timing
        rows.append(
            {
                "holdout": f"{start[:4]}-{end[:4]}",
                "active": bm.active_mean,
                "active_lo": bm.ci_low,
                "active_hi": bm.ci_high,
                "timing": et.timing_mean,
                "timing_lo": et.ci_low,
                "timing_hi": et.ci_high,
                "transition_acc": res.transitions.transition_accuracy if res.transitions else None,
                "verdict": res.verdict.value,
            }
        )
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 140, "display.float_format", "{:+.4f}".format):
        print(out.to_string(index=False))
    print("\nVerdict counts:", out["verdict"].value_counts().to_dict())
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
