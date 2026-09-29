# Strategy specification

[中文](strategy.zh-CN.md)

## Plain-language version

At each month end, the platform gives every eligible stock a health check. It prefers stocks with stable relative strength and reasonable risk, avoids difficult-to-trade names, and uses either a transparent weighted score or XGBoost to rank the candidates. The portfolio buys the best-ranked diversified group on the next trading day and repeats the process monthly.

## Research universe

The strategy's default research universe is the CSI 800, constructed explicitly as CSI 300
(`LARGE`) plus CSI 500 (`MID`). This covers liquid large- and mid-cap A-shares without admitting the
full tail of difficult-to-trade microcaps. A stock must be listed long enough to form the longest
factor window, have valid prices, be trading, not be risk flagged under the configured policy, and
rank above the bottom 20% of the universe by trailing 20-day median turnover amount.

The free research profile reconstructs point-in-time CSI 800 membership from BaoStock month-end
CSI 300 and CSI 500 snapshots. This removes the current-constituent backfill used by the early
prototype. It remains a monthly approximation rather than a licensed daily constituent database,
which matches the strategy's month-end decision frequency and is disclosed in the run manifest.

## Signal timing and target

- Information cutoff: T close.
- Order generation: after T close.
- Earliest execution: T+1 open.
- Holding horizon: 20 trading days.
- Target: stock return from T+1 open to T+20 close minus the same-date cross-sectional median return.
- Model task: rank stocks within each rebalance date.

## Models

### Linear factor benchmark

The baseline uses fixed, documented signs for standardized factors. It provides a falsifiable benchmark: a complex model must add value beyond a simple economic hypothesis.

### XGBoost ranker

`XGBRanker` learns nonlinear interactions, such as short-term reversal being useful only when medium-term trend and liquidity are healthy. Its score is a ranking value, not a calibrated probability of profit.

## Portfolio policy

- Select the highest-ranked candidates.
- Equal, inverse-volatility, or score-tilted weights subject to maximum stock weight.
- Enforce an absolute industry concentration limit only when point-in-time industry data is present.
- Retain existing holdings until they fall below the exit rank.
- Admit new holdings from the tighter entry rank.
- Apply ADV participation and lot-size constraints.
- Keep unallocated cash when orders cannot be filled safely.
- Preserve a configurable cash buffer and ignore uneconomic small rebalances.

The default schedule computes scores on the final trading session of each month and submits the
new target portfolio on the next available trading session. The rebalance ledger records both dates,
the number of entries and exits, target exposure, realized turnover, and fees. Before training starts,
the pipeline rejects universes that are smaller than `top_k`, cannot reach full target exposure, lack
enough industry capacity, or are too small for the configured exit-rank buffer to remove holdings.

Portfolio size is a deployment choice rather than a model constant. `--holdings 3` or
`--holdings 5` creates a concentrated personal portfolio, while the default 50-stock configuration
represents an institutional research portfolio. The override derives a compatible exit buffer and
equal-weight stock/industry limits, and `--initial-cash` changes capital without bypassing A-share lot
sizes, minimum commission, slippage, or liquidity constraints.

The model always scores a broad cross-section even when the final account holds only 3–5 names.
Formal runs require at least 200 tradable candidates per rebalance date by default; portfolio size is
therefore not allowed to masquerade as model-universe size.

When both size segments are present, no more than 60% of positions may come from `MID` and total
`MID` target weight is capped at 70%. A five-stock personal portfolio therefore keeps at least two
`LARGE` names; a three-stock portfolio keeps at least one. These constraints are skipped, with an
explicit data warning, when a custom historical membership file does not provide size labels.

Cash dividends and bonus shares are booked on their ex-dates. Rights entitlements are recorded but
not automatically subscribed, because exercising them is a separate capital-allocation decision.

The zero-cost profile intentionally leaves industry unknown instead of leaking today's classification
into the past. Factor industry demeaning and industry caps are therefore disabled for that profile;
size-segment controls remain active. Daily historical ST/suspension enrichment can also be skipped
to make a full-universe run practical, but that choice forces the evidence grade to `LIMITED_RESEARCH`.

## Research evidence hierarchy

1. Causal feature tests.
2. Single-factor RankIC and quantile monotonicity.
3. Linear baseline out-of-sample results.
4. XGBoost incremental value over the baseline.
5. Cost and capacity sensitivity.
6. Subperiod and market-regime stability.

No strategy claim should be made from synthetic data or in-sample results.

## Result classification

- `FORMAL`: all configured point-in-time data gates passed and the result may enter research review.
- `REPRODUCIBLE_FREE_DATA`: all required gates passed while the deliberately optional historical
  industry gate was disabled for the zero-cost profile.
- `LIMITED_RESEARCH`: the full universe and prices are available, but at least one historical-state
  gate is incomplete; the result may be inspected but not presented as unbiased performance.
- `ENGINEERING_ONLY`: the universe is truncated/incomplete or a core storage/benchmark gate failed.
- `--allow-non-research-ready`: explicitly permits the latter two grades while preserving the label.

The six benchmark comparisons answer different questions. CSI 800 is the primary investable
benchmark; CSI 300 and CSI 500 expose large/mid-cap style dependence; SSE Composite and SZSE
Component provide broad Shanghai/Shenzhen market context; beating universe equal weight is stronger
evidence that the ranking model added stock-selection value. XGBoost is accepted only when it adds
stable out-of-sample value over the transparent linear factor model after costs.
