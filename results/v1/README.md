# V1证据包 / Strategy V1 evidence pack

本目录保存从完整本地账本导出的、适合提交Git的汇总证据，不包含原始行情、逐股票预测或个人持仓。

This directory contains compact, Git-trackable summaries exported from the full local ledgers.
It contains no raw market data, stock-level predictions, or personal positions.

- Strategy version / 策略版本: `v1.0.0`
- Evidence grade / 证据等级: `LIMITED_RESEARCH`
- OOS period / 样本外区间: `2019-08-01` — `2026-09-04`
- Rebalance observations / 调仓期数: `85`
- Full local source / 完整本地来源: `artifacts/low_turnover`, `artifacts/strategy_validation`
- Rebuild / 重新导出: `python scripts/export_v1_results.py`

## 文件 / Files

- `headline_results.csv`：机构版与个人版核心绩效 / headline portfolio metrics
- `monthly_normalized_nav.csv`：月末归一化净值与主要基准 / monthly normalized NAV and benchmarks
- `annual_returns.csv`：逐年策略与基准收益 / calendar-year strategy and benchmark returns
- `parameter_stability_summary.csv`：参数网格稳健区间 / parameter-grid robustness ranges
- `cost_stress_summary.csv`：0–3倍交易成本结果 / 0–3× transaction-cost scenarios
- `factor_ablation_summary.csv`：五组因子消融组合结果 / five factor-group ablations
- `market_regime_attribution.csv`：牛市、震荡、熊市归因 / bull, sideways, and bear attribution

原始产物体积较大且可能包含数据源派生内容，因此继续由 `.gitignore` 排除。
Raw artifacts remain ignored because they are large and may include provider-derived data.
