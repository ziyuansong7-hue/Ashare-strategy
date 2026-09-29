# 正式研究前检查清单

只有 `data/csi800_pit_market/metadata/download_report.json` 中的 `research_ready=true`，结果才允许进入 `FORMAL` 级研究。工程跑通、模型能训练、净值为正都不能替代数据门禁。免费降级运行可在明确标记为 `LIMITED_RESEARCH` 后用于方法研究，但不得改称正式无偏业绩。

## 必须通过的数据门禁

1. `point_in_time_membership`：指数成分必须带 `effective_from/effective_to`，不能用当前成分回填历史。
2. `universe_completeness`：中证800至少覆盖270只 LARGE 和450只 MID，测试截断数据不得冒充完整股票池。
3. `historical_security_status`：每个交易日使用当时的 ST 与交易状态。
4. `point_in_time_industry`：若启用行业中性化/行业上限，行业分类必须按生效日匹配；免费版可将 `historical_industry_required=false` 并完全关闭相关约束，不允许使用今天行业回填。
5. `corporate_actions`：分红、送股和配股历史下载完整。
6. `raw_and_adjusted_prices`：复权研究价格与原始成交价格同时存在。
7. `required_benchmarks_downloaded`：中证800、沪深300、中证500、上证综指与深证成指均完整。
8. `storage_audit_passed`：代码类型、OHLC、重复行、基金混入和文件覆盖检查通过。

模型层另要求每个调仓日至少有 200 只可交易股票。个人账户最终只买 3–5 只，也仍然从这个宽股票池中排名。

## 推荐执行顺序

```powershell
ashare-quant sync-data --universe csi800 --start 2015-01-01 --end 2026-09-25 --data-dir data\csi800_pit_market --workers 12 --skip-historical-status
ashare-quant audit-data --data-dir data\csi800_pit_market --min-rows 120
ashare-quant run --data data\csi800_pit_market --config configs\default.toml --output artifacts\csi800_pit_walk_forward --walk-forward --allow-non-research-ready
```

若任一核心门禁失败，修复数据后重新同步。上述零成本命令会保留时点化成分，但因跳过每日历史 ST/停牌状态而标记为 `LIMITED_RESEARCH`。`--allow-non-research-ready` 只是允许运行这个已披露局限的回测，不会修改产物中的证据标签。

## V1 执行状态

V1 已在 `LIMITED_RESEARCH` 数据等级下完成带标签清洗的 expanding walk-forward、线性基准与 XGBoost 比较、低换手机构/个人组合、成本压力、参数稳定性、因子消融及风险收益归因。结果见[正式验证报告](formal_strategy_validation.md)。由于每日历史 ST/状态和时点行业仍不完整，数据门禁没有被伪装成 `FORMAL`；若未来取得授权数据，应保持冻结参数重新运行，作为真正的新样本验证。
