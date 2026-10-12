"""Tests for the five-layer reconciliation protocol.

Each test builds data with a known answer: constant policies have exactly zero
timing, an uninformative signal must not be labeled as timing skill, a
perfect-foresight signal must be, and a continuation forecaster is right on
every persistent period and wrong on every transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alpha_lens import TimingVerdict, reconcile
from alpha_lens.analysis.reconciliation import (
    circular_block_indices,
    exposure_timing_series,
    minimum_detectable_ir,
    simulate_rebalanced_portfolio,
    transition_accuracy,
)

ASSETS = ["stock", "bond", "cash"]
POLICY = np.array([[0.70, 0.20, 0.10], [0.20, 0.50, 0.30], [0.55, 0.10, 0.35]])
BENCH = {"60/40": [0.6, 0.4, 0.0]}


def make_monthly_returns(n: int = 72, *, seed: int = 0, stock_mu: float = 0.008) -> pd.DataFrame:
    """Independent monthly returns: stock, bond, cash (decimal)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M").astype(str)
    data = np.column_stack(
        [
            rng.normal(stock_mu, 0.045, n),
            rng.normal(0.003, 0.015, n),
            rng.normal(0.002, 0.0005, n),
        ]
    )
    return pd.DataFrame(data, index=idx, columns=ASSETS)


def states_from_returns(r: pd.DataFrame) -> np.ndarray:
    """Realized state per period from relative returns (0=stock leads, 1=stock lags, 2=bond lags)."""
    s1 = r["stock"] - r["bond"]
    s2 = r["bond"] - r["cash"]
    return np.where(s1 < 0, 1, np.where(s2 >= 0, 0, 2))


def weights_from_states(states: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(POLICY[states], index=index, columns=ASSETS)


# ----------------------------------------------------------------------------
# Accounting
# ----------------------------------------------------------------------------


def test_turnover_and_cost_match_hand_calculation() -> None:
    idx = ["t0", "t1"]
    w = pd.DataFrame([[0.6, 0.4, 0.0], [0.6, 0.4, 0.0]], index=idx, columns=ASSETS)
    r = pd.DataFrame([[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]], index=idx, columns=ASSETS)
    net, to = simulate_rebalanced_portfolio(w, r, cost_bps=10)
    pre = np.array([0.66, 0.40, 0.0]) / 1.06  # drifted weights
    expected_to = 0.5 * np.abs(np.array([0.6, 0.4, 0.0]) - pre).sum()
    assert to.iloc[0] == 0.0
    assert to.iloc[1] == pytest.approx(expected_to, abs=1e-15)
    assert net.iloc[0] == pytest.approx(0.06, abs=1e-15)
    assert net.iloc[1] == pytest.approx(-0.001 * expected_to, abs=1e-15)


def test_constant_policy_has_exactly_zero_timing() -> None:
    r = make_monthly_returns()
    w = pd.DataFrame(np.tile([0.7, 0.2, 0.1], (len(r), 1)), index=r.index, columns=ASSETS)
    timing, mean_w, constant = exposure_timing_series(w, r)
    assert constant
    assert (timing == 0.0).all()
    assert mean_w["stock"] == pytest.approx(0.7)


def test_circular_block_indices_shape_and_contiguity() -> None:
    idx = circular_block_indices(72, block_length=6, n_boot=50, seed=1)
    assert idx.shape == (50, 72)
    assert idx.min() >= 0 and idx.max() < 72
    first_block = idx[:, :6]
    steps = np.diff(first_block, axis=1) % 72
    assert (steps == 1).all()


def test_minimum_detectable_ir_formula() -> None:
    expected = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(6)
    assert minimum_detectable_ir(6.0) == pytest.approx(expected)
    assert minimum_detectable_ir(6.0) == pytest.approx(1.144, abs=1e-3)


# ----------------------------------------------------------------------------
# Layer 1
# ----------------------------------------------------------------------------


def test_continuation_is_perfect_on_persistent_and_zero_on_transitions() -> None:
    r = make_monthly_returns(seed=3)
    y = pd.Series(states_from_returns(r), index=r.index)
    pred = y.shift(1).fillna(y.iloc[0]).astype(int)
    layer = transition_accuracy(y, pred)
    assert layer.transition_accuracy == 0.0
    assert layer.persistent_accuracy == 1.0
    assert layer.null_rate == pytest.approx(0.5)
    assert layer.passed is False


def test_perfect_forecaster_passes_transition_test() -> None:
    r = make_monthly_returns(seed=4)
    y = pd.Series(states_from_returns(r), index=r.index)
    layer = transition_accuracy(y, y)
    assert layer.transition_accuracy == 1.0
    assert layer.passed is True


# ----------------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------------


def test_exposure_trap_is_not_labeled_timing() -> None:
    """Equity-heavy constant policy in a strong equity market beats 60/40 by exposure only."""
    r = make_monthly_returns(n=240, seed=5, stock_mu=0.02)
    w = pd.DataFrame(np.tile([0.9, 0.1, 0.0], (len(r), 1)), index=r.index, columns=ASSETS)
    res = reconcile(w, r, BENCH, periods_per_year=12, block_length=6, n_boot=999)
    assert res.benchmarks[0].ci_low > 0
    assert res.exposure_timing.constant_policy
    assert res.verdict is TimingVerdict.EXPOSURE_NOT_TIMING


def test_perfect_foresight_is_labeled_timing() -> None:
    r = make_monthly_returns(n=144, seed=6)
    y = states_from_returns(r)
    w = weights_from_states(y, r.index)
    res = reconcile(
        w,
        r,
        BENCH,
        realized_states=pd.Series(y, index=r.index),
        predicted_states=pd.Series(y, index=r.index),
        periods_per_year=12,
        block_length=6,
        n_boot=999,
    )
    assert res.exposure_timing.ci_low > 0
    assert res.verdict is TimingVerdict.TIMING_SUPPORTED


def test_uninformative_signal_rarely_claims_timing() -> None:
    """Size check: a persistent signal independent of returns should seldom pass layer 3."""
    rng = np.random.default_rng(7)
    claims = 0
    n_sims = 60
    for i in range(n_sims):
        r = make_monthly_returns(seed=100 + i)
        s = np.empty(len(r), dtype=int)
        s[0] = rng.integers(3)
        for t in range(1, len(r)):
            s[t] = s[t - 1] if rng.random() < 0.5 else rng.integers(3)
        res = reconcile(
            weights_from_states(s, r.index),
            r,
            BENCH,
            periods_per_year=12,
            block_length=6,
            n_boot=299,
        )
        claims += res.exposure_timing.ci_low > 0
    assert claims / n_sims <= 0.12  # nominal one-sided 2.5%; generous bound for 60 draws


def test_reconcile_is_deterministic_given_seed() -> None:
    r = make_monthly_returns(seed=8)
    w = weights_from_states(states_from_returns(r.shift(1).fillna(0.0)), r.index)
    a = reconcile(w, r, BENCH, periods_per_year=12, block_length=6, n_boot=499, seed=1)
    b = reconcile(w, r, BENCH, periods_per_year=12, block_length=6, n_boot=499, seed=1)
    assert a.exposure_timing.ci_low == b.exposure_timing.ci_low
    assert a.verdict == b.verdict
    assert "Verdict:" in a.summary()


# ----------------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------------


def test_rejects_percent_encoded_returns() -> None:
    r = make_monthly_returns() * 100
    w = pd.DataFrame(np.tile([0.6, 0.4, 0.0], (len(r), 1)), index=r.index, columns=ASSETS)
    with pytest.raises(ValueError, match="percent"):
        reconcile(w, r, BENCH, periods_per_year=12)


def test_rejects_weights_not_summing_to_one() -> None:
    r = make_monthly_returns()
    w = pd.DataFrame(np.tile([0.6, 0.3, 0.0], (len(r), 1)), index=r.index, columns=ASSETS)
    with pytest.raises(ValueError, match="sum to 1"):
        reconcile(w, r, BENCH, periods_per_year=12)


def test_significant_underperformance_is_reported() -> None:
    """A bond-heavy constant policy in an equity bull market lags 60/40 significantly."""
    r = make_monthly_returns(n=240, seed=9, stock_mu=0.02)
    w = pd.DataFrame(np.tile([0.1, 0.6, 0.3], (len(r), 1)), index=r.index, columns=ASSETS)
    res = reconcile(w, r, BENCH, periods_per_year=12, block_length=6, n_boot=999)
    assert res.benchmarks[0].ci_high < 0
    assert res.verdict is TimingVerdict.INCONCLUSIVE
    assert any("Underperforms 60/40" in s for s in res.reasons)
