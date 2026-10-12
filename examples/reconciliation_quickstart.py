"""Reconciliation quickstart: exposure that looks like timing, and real timing.

Two monthly three-asset strategies are run through the same five-layer check.

* "Lucky tilt" never forecasts anything. It just holds more stock than 60/40
  in an equity bull market, so it beats 60/40 handsomely.
* "Real timing" uses a signal that is right about the coming month's
  relative-return state 50% of the time and random otherwise.

A plain benchmark comparison praises both. The protocol separates them.

Run from the repo root::

    python examples/reconciliation_quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lens import reconcile

ASSETS = ["stock", "bond", "cash"]
# Policy used in Tso (2026): one target per relative-performance state.
POLICY = np.array([[0.70, 0.20, 0.10], [0.20, 0.50, 0.30], [0.55, 0.10, 0.35]])
BENCHMARKS = {"60/40": [0.6, 0.4, 0.0], "Equal weight": [1 / 3, 1 / 3, 1 / 3]}


def simulate_market(n_months: int, seed: int) -> pd.DataFrame:
    """Monthly decimal returns for stock, bond and cash in a strong equity market."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2010-01", periods=n_months, freq="M").astype(str)
    data = np.column_stack(
        [
            rng.normal(0.012, 0.045, n_months),
            rng.normal(0.002, 0.018, n_months),
            rng.normal(0.002, 0.0005, n_months),
        ]
    )
    return pd.DataFrame(data, index=idx, columns=ASSETS)


def realized_state(r: pd.DataFrame) -> np.ndarray:
    """0 = stock leads, 1 = stock lags bonds, 2 = bonds lag cash."""
    s1 = (r["stock"] - r["bond"]).to_numpy()
    s2 = (r["bond"] - r["cash"]).to_numpy()
    return np.where(s1 < 0, 1, np.where(s2 >= 0, 0, 2))


def main() -> None:
    r = simulate_market(120, seed=11)
    states = realized_state(r)
    rng = np.random.default_rng(12)

    lucky = pd.DataFrame(np.tile([0.75, 0.15, 0.10], (len(r), 1)), index=r.index, columns=ASSETS)

    guess = rng.integers(0, 3, len(r))
    signal = np.where(rng.random(len(r)) < 0.5, states, guess)
    timing = pd.DataFrame(POLICY[signal], index=r.index, columns=ASSETS)

    for name, w, sig in [("Lucky tilt", lucky, None), ("Real timing", timing, signal)]:
        res = reconcile(
            w,
            r,
            BENCHMARKS,
            realized_states=pd.Series(states, index=r.index),
            predicted_states=None if sig is None else pd.Series(sig, index=r.index),
            periods_per_year=12,
            block_length=6,
            n_boot=10_000,
        )
        print("=" * 72)
        print(name)
        print("=" * 72)
        print(res.summary())
        print()


if __name__ == "__main__":
    main()
