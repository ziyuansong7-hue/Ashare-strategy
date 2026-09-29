# Architecture

[中文](architecture.zh-CN.md)

## Design principle

The platform separates three concerns that are often mixed inside one notebook:

1. Research truth: what was known at each historical timestamp?
2. Model truth: was the validation/test boundary respected?
3. Trading truth: could the proposed portfolio actually be executed?

## Components

| Component | Responsibility | Failure examples |
|---|---|---|
| Data loader | Preserve symbols and load CSV/Parquet | `000001` parsed as integer, duplicate bars |
| Validator | Enforce canonical types and OHLC invariants | negative volume, invalid high/low |
| Factor engine | Compute causal, explainable factors | rolling across symbol boundaries |
| Dataset builder | Align next-open labels and rebalance dates | same-close leakage |
| Splitter | Purge labels that cross time boundaries | training target uses validation prices |
| Walk-forward orchestrator | Retrain each fold and stitch test-only predictions | overlap, test leakage |
| Baseline model | Provide transparent factor benchmark | incorrect factor direction |
| XGBoost ranker | Learn nonlinear cross-factor relationships | test data used for early stopping |
| Portfolio constructor | Convert scores to diversified targets | industry concentration, churn |
| Order planner | Convert current holdings and cash into next-session instructions | stale prices, sub-lot orders |
| Execution simulator | Apply tradability and cost assumptions | impossible limit-up fill |
| Corporate-action ledger | Book dividends and share changes explicitly | phantom PnL around ex-dates |
| Ledger | Reconcile cash, holdings and NAV | fee sign or quantity error |
| Analytics | Report factor and portfolio evidence | accuracy reported without tradability |
| Data auditor | Re-open persisted Parquet and enforce storage quality gates | fund codes, duplicate bars, missing raw prices |
| Research gate | Prevent non-point-in-time data from being presented as research evidence | current constituents backfilled into history |

## Core records

```text
Bar(date, code, open, high, low, close, volume, amount)
FactorObservation(date, code, factor_name, value, as_of)
Signal(date, code, model_version, score, rank)
Order(date, code, side, requested_quantity, status)
Fill(date, code, side, quantity, price, fees)
Position(date, code, quantity, close, market_value)
PortfolioSnapshot(date, cash, market_value, nav)
BenchmarkSnapshot(date, benchmark_key, close, source_endpoint)
CorporateAction(ex_date, code, cash_per_share, bonus_ratio, rights_ratio)
```

## Benchmark hierarchy

The backtest reports six independent controls rather than combining unrelated indices:

1. Universe equal weight tests whether stock selection beats the available candidate pool.
2. CSI 800 is the primary investable benchmark for the selected large/mid-cap universe.
3. CSI 300 exposes large-cap relative performance.
4. CSI 500 exposes mid-cap relative performance.
5. SSE Composite tests performance relative to the broad Shanghai market.
6. SZSE Component tests performance relative to the Shenzhen market.

All benchmark series are aligned to strategy trading dates using current or earlier observations only.
Reported statistics include excess return, tracking error, information ratio, annualized alpha, beta,
correlation, and calendar-year returns.

## Why hybrid rather than fully event-driven

Millions of historical daily rows are efficiently handled with vectorized group operations. Orders, fills, cash and positions are stateful and are therefore processed as events. This gives the project an auditable trading chain without making factor research unnecessarily slow or complex.

## Production boundary

This repository is a research and simulation platform. A production brokerage deployment would additionally require authenticated market data, persistent event storage, entitlement controls, reconciliation, disaster recovery, monitoring, approval workflows, and regulatory reporting.
