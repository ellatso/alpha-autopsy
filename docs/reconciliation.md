# Reconciliation protocol: is it timing, or just exposure?

`reconcile()` checks whether a dynamic allocation strategy adds value **beyond
its own average exposure**. It implements the five-layer protocol of
Tso (2026), *A Reconciliation Protocol for Tactical Asset Allocation:
Forecasting, Exposure, and Economic Value in Relative-Performance Regimes*
(SSRN working paper).

## Why

A strategy can beat 60/40 without forecasting anything. If its average weights
hold more stock than 60/40 during an equity bull market, it wins by exposure.
In simulation, an uninformative but equity-tilted signal beats 60/40 on
compound growth in 76% of 72-month paths and passes a block-bootstrap
significance test against 60/40 in 15.8% of them. The protocol's timing test
rejects in 2.1% to 3.4% of paths, close to its nominal 2.5%.

## The five layers

| Layer | Question | What `reconcile` reports |
|---|---|---|
| 1 Transitions | Does accuracy survive state changes? | Persistent vs transition accuracy; exact test vs the conditional guess rate 1/(K-1) |
| 2 Benchmarks | Does it beat each benchmark? | Paired net active return, block-bootstrap interval, IR |
| 3 Exposure | Is it timing or average exposure? | Dynamic net minus the net of a portfolio held at the strategy's own mean weights |
| 4 Concentration | Do a few periods carry it? | Timing mean after removing the k best periods |
| 5 Resolution | What could this sample detect? | Minimum detectable IR and effect size at the stated power |

## Usage

```python
import pandas as pd
from alpha_lens import reconcile

# weights: target weights held in each period, formed with information
#          available by the end of the previous period (rows sum to 1)
# returns: simple decimal asset returns realized in each period
result = reconcile(
    weights,
    returns,
    benchmarks={"60/40": [0.6, 0.4, 0.0], "Equal weight": [1/3, 1/3, 1/3]},
    realized_states=states,        # optional, enables layer 1
    predicted_states=forecasts,    # optional, enables layer 1
    periods_per_year=12,           # 252 for daily data
    block_length=6,                # bootstrap block length in periods
    cost_bps=10,
)
print(result.verdict)      # TimingVerdict.EXPOSURE_NOT_TIMING, ...
print(result.summary())    # every layer in plain text
```

## Verdicts

| Verdict | Meaning |
|---|---|
| `TIMING_SUPPORTED` | Timing interval above zero and every applicable layer passes |
| `TIMING_UNCONFIRMED` | Timing interval above zero, but transitions or concentration fail |
| `EXPOSURE_NOT_TIMING` | Beats a benchmark significantly; timing interval includes zero |
| `NEGATIVE_TIMING` | Timing interval below zero |
| `INCONCLUSIVE` | Nothing established either way (reasons list any significant underperformance and whether the sample is underpowered) |

## Accounting and inference

* Monthly (or per-period) reset to target weights; turnover is half the L1
  distance between the target and the drifted previous target; cost is
  `cost_bps` per unit of turnover; the first period carries no cost.
* All intervals use **one** circular block-bootstrap index matrix, so
  comparisons are paired. The default block length is `round(n ** (1/3))`.
* The mean-weight comparator uses ex-post average weights. It is an
  attribution device, not an investable portfolio.
* In 72-month samples the percentile block bootstrap is mildly liberal
  (simulated one-sided rate 3.5%–5.1% without costs), so a failure to reject
  is conservative but a marginal rejection deserves caution.

## Examples

* `examples/reconciliation_quickstart.py` – a lucky equity tilt is flagged as
  `EXPOSURE_NOT_TIMING`; a genuinely informed signal is `TIMING_SUPPORTED`.
* `examples/us_regime_reconciliation.py` – twelve US holdouts, 1954–2025, from
  the public Welch–Goyal data. Its output matches the paper to within 1e-16.
