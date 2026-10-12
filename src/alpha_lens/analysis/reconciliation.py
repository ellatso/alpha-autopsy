"""Five-layer reconciliation protocol for forecast-driven allocation strategies.

A dynamic allocation that beats a benchmark has not necessarily timed anything.
It may simply hold a different *average* mix of assets. This module separates
the claims, following Tso (2026), "A Reconciliation Protocol for Tactical Asset
Allocation" (SSRN working paper):

    1. Transitions   - does forecast accuracy survive state changes?
    2. Benchmarks    - paired net comparison with each benchmark.
    3. Exposure      - timing = dynamic net return minus the net return of a
                       portfolio held at the strategy's own mean weights.
    4. Concentration - does timing survive removing the k best periods?
    5. Resolution    - what effect size could this sample detect?

All functions are pure: no I/O and no randomness without a seed.

Timing convention: ``weights`` row ``t`` is the target held *during* period
``t`` and must be formed from information available by the end of ``t - 1``.
``asset_returns`` row ``t`` holds the returns realized during period ``t``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lens.core.types import (
    BenchmarkComparison,
    ConcentrationLayer,
    ExposureTimingLayer,
    ReconciliationResult,
    ResolutionLayer,
    TimingVerdict,
    TransitionLayer,
)

_TOL = 1e-12


# ----------------------------------------------------------------------------
# Portfolio accounting
# ----------------------------------------------------------------------------


def simulate_rebalanced_portfolio(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series]:
    """Net returns of a portfolio reset to target weights every period.

    Turnover is one half of the L1 distance between the new target and the
    previous target after drift. Cost is ``cost_bps`` per unit of one-way
    turnover. The first period carries no cost (initial establishment is
    treated identically for every portfolio being compared).

    Args:
        weights: Target weights, index = periods, columns = assets. Rows sum to 1.
        asset_returns: Simple decimal returns with the same index and columns.
        cost_bps: Transaction cost in basis points per unit of turnover.

    Returns:
        (net_returns, turnover) as Series on the input index.
    """
    w = weights.to_numpy(dtype=float)
    r = asset_returns.to_numpy(dtype=float)
    n = len(w)
    turnover = np.zeros(n)
    for t in range(1, n):
        grown = w[t - 1] * (1.0 + r[t - 1])
        pretrade = grown / grown.sum()
        turnover[t] = 0.5 * np.abs(w[t] - pretrade).sum()
    net = (w * r).sum(axis=1) - cost_bps / 10_000.0 * turnover
    return (
        pd.Series(net, index=weights.index, name="net_return"),
        pd.Series(turnover, index=weights.index, name="turnover"),
    )


def _cagr(x: np.ndarray, periods_per_year: int) -> float:
    return float(np.prod(1.0 + x) ** (periods_per_year / len(x)) - 1.0)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------


def circular_block_indices(n: int, *, block_length: int, n_boot: int, seed: int) -> np.ndarray:
    """Index matrix for a circular block bootstrap (Politis and Romano, 1992).

    Each row concatenates blocks of ``block_length`` consecutive indices with
    wrap-around and keeps the first ``n``. Reusing one matrix for several
    series gives *paired* intervals.
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def bootstrap_mean_ci(
    x: np.ndarray, idx: np.ndarray, *, periods_per_year: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Annualized mean and percentile interval from a bootstrap index matrix."""
    x = np.asarray(x, dtype=float)
    boot = x[idx].mean(axis=1) * periods_per_year
    a = (1.0 - level) / 2.0
    return (
        float(x.mean() * periods_per_year),
        float(np.percentile(boot, 100 * a)),
        float(np.percentile(boot, 100 * (1 - a))),
    )


def minimum_detectable_ir(years: float, *, alpha: float = 0.05, power: float = 0.80) -> float:
    """True information ratio needed to reject zero with given power (iid).

    IR_min = (z_{1-alpha/2} + z_{power}) / sqrt(years).
    Six years at 5%/80% gives about 1.14.
    """
    if years <= 0:
        raise ValueError("years must be positive")
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z / np.sqrt(years))


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------


def transition_accuracy(
    realized_states: pd.Series, predicted_states: pd.Series, *, alpha: float = 0.05
) -> TransitionLayer:
    """Layer 1: accuracy on persistent versus transition periods.

    A transition is a period whose realized state differs from the previous
    period's realized state (so the first period is excluded). Conditional on
    a transition, a uniform guess among the K - 1 other states succeeds with
    probability 1/(K - 1); that is the null for the one-sided exact test.
    """
    y = realized_states.to_numpy()
    p = predicted_states.reindex(realized_states.index).to_numpy()
    correct = y == p
    trans = np.zeros(len(y), dtype=bool)
    trans[1:] = y[1:] != y[:-1]
    persistent = np.zeros(len(y), dtype=bool)
    persistent[1:] = ~trans[1:]
    k = len(pd.unique(realized_states))
    n_tr = int(trans.sum())
    k_tr = int(correct[trans].sum())
    null = 1.0 / (k - 1) if k >= 3 else None
    pval = (
        float(stats.binomtest(k_tr, n_tr, null, alternative="greater").pvalue)
        if (null is not None and n_tr > 0)
        else None
    )
    return TransitionLayer(
        n_periods=len(y),
        n_transitions=n_tr,
        overall_accuracy=float(correct.mean()),
        persistent_accuracy=float(correct[persistent].mean()) if persistent.any() else None,
        transition_accuracy=k_tr / n_tr if n_tr else None,
        null_rate=null,
        p_value=pval,
        passed=None if pval is None else bool(pval < alpha),
    )


def exposure_timing_series(
    weights: pd.DataFrame, asset_returns: pd.DataFrame, *, cost_bps: float = 10.0
) -> tuple[pd.Series, pd.Series, bool]:
    """Layer 3 series: dynamic net minus net of the strategy's mean weights.

    The comparator uses ex-post average weights, so it is an attribution
    device, not an investable portfolio. If the strategy never changes its
    target, timing is exactly zero.

    Returns:
        (timing_series, mean_weights, constant_policy)
    """
    mean_w = weights.mean(axis=0)
    constant = bool(np.allclose(weights.to_numpy(), weights.iloc[[0]].to_numpy(), atol=_TOL))
    dyn, _ = simulate_rebalanced_portfolio(weights, asset_returns, cost_bps=cost_bps)
    if constant:
        timing = pd.Series(0.0, index=weights.index, name="timing")
    else:
        static_w = pd.DataFrame(
            np.tile(mean_w.to_numpy(), (len(weights), 1)),
            index=weights.index,
            columns=weights.columns,
        )
        stat, _ = simulate_rebalanced_portfolio(static_w, asset_returns, cost_bps=cost_bps)
        timing = (dyn - stat).rename("timing")
    return timing, mean_w, constant


def _benchmark_weights(spec: Sequence[float] | pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    if isinstance(spec, pd.DataFrame):
        return spec.reindex(index=like.index, columns=like.columns)
    arr = np.asarray(spec, dtype=float)
    if arr.shape != (like.shape[1],):
        raise ValueError("fixed benchmark weights must have one entry per asset")
    return pd.DataFrame(np.tile(arr, (len(like), 1)), index=like.index, columns=like.columns)


def _check_inputs(weights: pd.DataFrame, asset_returns: pd.DataFrame) -> None:
    if not weights.index.equals(asset_returns.index):
        raise ValueError("weights and asset_returns must share the same index")
    if list(weights.columns) != list(asset_returns.columns):
        raise ValueError("weights and asset_returns must share the same columns, in order")
    if weights.isna().any().any() or asset_returns.isna().any().any():
        raise ValueError("inputs contain NaN; clean them before reconciling")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of weights must sum to 1")
    if (asset_returns.abs() > 1.0).any().any():
        raise ValueError("asset_returns look percent-encoded; pass decimals (0.01 = 1%)")


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def reconcile(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    benchmarks: Mapping[str, Sequence[float] | pd.DataFrame],
    *,
    realized_states: pd.Series | None = None,
    predicted_states: pd.Series | None = None,
    primary_benchmark: str | None = None,
    cost_bps: float = 10.0,
    periods_per_year: int = 252,
    block_length: int | None = None,
    n_boot: int = 10_000,
    seed: int = 20260910,
    concentration_k: int = 5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> ReconciliationResult:
    """Run the five-layer reconciliation protocol on a dynamic allocation.

    Args:
        weights: Target weights held in each period (formed with information
            through the previous period). Rows sum to 1.
        asset_returns: Simple decimal asset returns realized in each period.
        benchmarks: Name -> fixed weight vector or a weights DataFrame.
        realized_states, predicted_states: Optional state labels per period;
            enable layer 1 when both are given.
        primary_benchmark: Benchmark used for the active-return resolution
            figure. Defaults to the first benchmark.
        cost_bps: Transaction cost per unit of one-way turnover.
        periods_per_year: 252 for daily, 12 for monthly data.
        block_length: Bootstrap block length in periods. Default: round(n^(1/3)).
        n_boot: Bootstrap replications (one common, paired index matrix).
        seed: Seed for the bootstrap index matrix.
        concentration_k: Number of most favorable periods removed in layer 4.
        alpha, power: Test size and power for layers 1 and 5.

    Returns:
        A :class:`ReconciliationResult` with every layer and a verdict.
    """
    _check_inputs(weights, asset_returns)
    if not benchmarks:
        raise ValueError("provide at least one benchmark")
    n = len(weights)
    if n <= concentration_k + 2:
        raise ValueError("sample too short for the concentration layer")
    bl = block_length if block_length is not None else max(1, int(round(n ** (1 / 3))))
    idx = circular_block_indices(n, block_length=bl, n_boot=n_boot, seed=seed)
    ppy = periods_per_year
    level = 1 - alpha

    strat_net, _ = simulate_rebalanced_portfolio(weights, asset_returns, cost_bps=cost_bps)
    s = strat_net.to_numpy()

    # Layer 2
    comps: list[BenchmarkComparison] = []
    active_by_name: dict[str, np.ndarray] = {}
    for name, spec in benchmarks.items():
        bw = _benchmark_weights(spec, weights)
        _check_inputs(bw, asset_returns)
        b_net, _ = simulate_rebalanced_portfolio(bw, asset_returns, cost_bps=cost_bps)
        act = s - b_net.to_numpy()
        active_by_name[name] = act
        mean, lo, hi = bootstrap_mean_ci(act, idx, periods_per_year=ppy, level=level)
        te = float(act.std(ddof=1) * np.sqrt(ppy))
        comps.append(
            BenchmarkComparison(
                name=name,
                strategy_cagr=_cagr(s, ppy),
                benchmark_cagr=_cagr(b_net.to_numpy(), ppy),
                active_mean=mean,
                ci_low=lo,
                ci_high=hi,
                tracking_error=te,
                information_ratio=mean / te if te > _TOL else 0.0,
            )
        )

    # Layer 3
    timing, mean_w, constant = exposure_timing_series(weights, asset_returns, cost_bps=cost_bps)
    tv = timing.to_numpy()
    t_mean, t_lo, t_hi = bootstrap_mean_ci(tv, idx, periods_per_year=ppy, level=level)
    t_te = float(tv.std(ddof=1) * np.sqrt(ppy))
    t_ir = t_mean / t_te if t_te > _TOL else 0.0
    if constant:
        t_mean = t_lo = t_hi = 0.0
    expo = ExposureTimingLayer(
        mean_weights={str(k): float(v) for k, v in mean_w.items()},
        timing_mean=t_mean,
        ci_low=t_lo,
        ci_high=t_hi,
        timing_ir=t_ir,
        constant_policy=constant,
        timing_series=timing,
    )

    # Layer 4
    primary = primary_benchmark or next(iter(benchmarks))
    if primary not in active_by_name:
        raise ValueError(f"primary_benchmark {primary!r} is not in benchmarks")
    k = concentration_k
    order = np.argsort(tv)[::-1]
    top = sorted(order[:k])
    conc = ConcentrationLayer(
        k=k,
        timing_mean_excluding_top_k=float(np.sort(tv)[:-k].mean() * ppy),
        active_mean_excluding_top_k=float(np.sort(active_by_name[primary])[:-k].mean() * ppy),
        top_timing_periods=[str(timing.index[i]) for i in top],
    )

    # Layer 5
    years = n / ppy
    ir_min = minimum_detectable_ir(years, alpha=alpha, power=power)
    primary_te = next(c.tracking_error for c in comps if c.name == primary)
    reso = ResolutionLayer(
        years=years,
        ir_min=ir_min,
        mde_timing=ir_min * t_te,
        mde_active=ir_min * primary_te,
        observed_timing_ir=t_ir,
        # A constant allocation has no timing to detect, so "underpowered" does not apply.
        underpowered=bool(not constant and abs(t_ir) < ir_min),
    )

    # Layer 1
    trans = None
    if realized_states is not None and predicted_states is not None:
        trans = transition_accuracy(
            realized_states.reindex(weights.index), predicted_states, alpha=alpha
        )

    verdict, reasons = _verdict(comps, expo, conc, reso, trans)
    return ReconciliationResult(
        verdict=verdict,
        reasons=reasons,
        transitions=trans,
        benchmarks=comps,
        exposure_timing=expo,
        concentration=conc,
        resolution=reso,
        settings={
            "cost_bps": cost_bps,
            "periods_per_year": ppy,
            "block_length": bl,
            "n_boot": n_boot,
            "seed": seed,
            "concentration_k": k,
            "alpha": alpha,
            "power": power,
            "primary_benchmark": primary,
        },
    )


def _verdict(
    comps: list[BenchmarkComparison],
    expo: ExposureTimingLayer,
    conc: ConcentrationLayer,
    reso: ResolutionLayer,
    trans: TransitionLayer | None,
) -> tuple[TimingVerdict, list[str]]:
    reasons: list[str] = []
    beats = [c.name for c in comps if c.ci_low > 0]
    if expo.constant_policy:
        reasons.append("Strategy holds one constant allocation; timing is zero by construction.")

    if expo.ci_low > 0:
        failed = []
        if conc.timing_mean_excluding_top_k <= 0:
            failed.append(f"timing turns non-positive without the best {conc.k} periods")
        if trans is not None and trans.passed is False:
            failed.append("transition accuracy does not beat the conditional guess rate")
        if failed:
            reasons += [f"Timing interval is above zero, but {f}." for f in failed]
            return TimingVerdict.TIMING_UNCONFIRMED, reasons
        reasons.append("Timing interval is above zero and every applicable layer passes.")
        return TimingVerdict.TIMING_SUPPORTED, reasons

    if expo.ci_high < 0 and not expo.constant_policy:
        reasons.append(
            "Timing interval is below zero: weight changes lose value versus the average allocation."
        )
        return TimingVerdict.NEGATIVE_TIMING, reasons

    if beats:
        reasons.append(
            f"Beats {', '.join(beats)} with an interval above zero, but the timing interval "
            "includes zero: the outperformance is attributable to average exposure."
        )
        verdict = TimingVerdict.EXPOSURE_NOT_TIMING
    else:
        lags = [c.name for c in comps if c.ci_high < 0]
        if lags:
            reasons.append(
                f"Underperforms {', '.join(lags)} with an interval below zero; "
                "timing is not established."
            )
        else:
            reasons.append("No benchmark comparison or timing interval excludes zero.")
        verdict = TimingVerdict.INCONCLUSIVE
    if reso.underpowered:
        reasons.append(
            f"Sample resolves timing effects only above ~{reso.mde_timing:.2%}/yr at the stated "
            "power; absence of evidence here is not evidence of absence."
        )
    return verdict, reasons
