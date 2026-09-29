# Local data

批量真实数据推荐使用按月历史成分近似的中证800研究范围：

```powershell
ashare-quant sync-data --universe csi800 --start 2015-01-01 --data-dir data\csi800_pit_market --skip-historical-status
```

`data/csi800_pit_market/bars/` 保存逐股票 Parquet；`benchmarks.parquet` 保存中证800、沪深300、中证500、上证综指与深证成指；
`corporate_actions.parquet` 保存公司行为；`industry_history.parquet` 保存时点行业；
`metadata/download_report.json` 是每次同步的质量报告。所有真实行情均被 `.gitignore` 排除。

同步器优先使用 BaoStock 月末历史沪深300/中证500成分快照，而不是把今天成分倒填历史。
`--skip-historical-status` 是免费数据降级：跳过完整逐日历史 ST/状态补充，因此证据等级仍为
`LIMITED_RESEARCH`。如另有授权数据，可通过 `--membership-file` 提供
`code,effective_from,effective_to,size_bucket`，并通过 `--industry-file` 提供时点行业。

`--allow-current-universe-backfill` 只适合工程冒烟测试，不能用于正式策略结论。

Market data is intentionally excluded from Git. The canonical input schema is:

`date, code, open, high, low, close, volume, amount`

Optional fields include:

`security_name, exchange, board, instrument_type, industry, size_bucket, market_cap, turnover, is_trading, is_st, lot_size, raw_open, raw_high, raw_low, raw_close, limit_up, limit_down`

`code` must be treated as a six-character string. Research prices should be consistently adjusted.
When raw execution prices are available, place them in `raw_open` and `raw_close`.
