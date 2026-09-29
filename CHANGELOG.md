# Changelog / 版本记录

## v1.0.0 — 2026-09-27 — Strategy V1 freeze / 策略V1冻结版

V1 freezes the validated monthly, low-turnover A-share research workflow. Future alpha, label, or
portfolio-policy experiments must be developed as V2 and must not overwrite the V1 evidence pack.

V1冻结了经过验证的A股月频低换手研究流程。后续因子、标签或组合规则实验必须作为V2开发，
不得覆盖V1证据包。

### Included / 已包含

- Point-in-time CSI 300 + CSI 500 research universe / 历史时点中证800研究股票池
- Causal price-volume factors and XGBoost ranking / 因果价量因子与XGBoost排序
- Purged expanding walk-forward evaluation / 带净化的扩展窗口样本外验证
- Institutional 50-stock and personal 5-stock profiles / 机构50股与个人5股组合
- Deterministic order priority and execution-aware backtest / 确定性订单优先级与成交约束回测
- Parameter, cost, ablation, and attribution validation / 参数、成本、消融与归因验证

### Frozen headline results / 冻结核心结果

| Profile / 版本 | Initial / 本金 | Total return / 累计收益 | Sharpe | Max drawdown / 最大回撤 |
|---|---:|---:|---:|---:|
| Institution / 机构 | ¥5,000,000 | 53.68% | 0.448 | -21.84% |
| Personal / 个人 | ¥300,000 | 54.42% | 0.415 | -44.16% |

Evidence grade: `LIMITED_RESEARCH`. This release is research software, not investment advice.

证据等级：`LIMITED_RESEARCH`。本版本是研究软件，不构成投资建议。
