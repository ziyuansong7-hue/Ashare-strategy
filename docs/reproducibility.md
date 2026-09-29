# 复现手册 / Reproducibility Guide

本页先给出中文完整流程，后附 English quick guide。命令默认从仓库根目录执行。

## 中文

### 1. 运行环境

- Python `3.11` 或更高版本；
- Windows PowerShell 为主验证环境；Linux/macOS 同样可用；
- 工程演示只需基础依赖；下载真实数据需要 `data` 可选依赖；
- 正式验证会多次回测并重训消融模型，耗时和内存明显高于演示。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev]"
ashare-quant --version
```

Linux/macOS 激活命令为 `source .venv/bin/activate`。

### 2. 零外部数据的确定性演示

```powershell
python scripts/run_demo.py
```

默认固定 `seed=42`，生成 160 只合成股票、520 个交易日，输出到 `artifacts/demo_v1/`。若想核对确定性，可删除或改用另一个输出目录后重复运行并比较汇总指标：

```powershell
python scripts/run_demo.py --output artifacts/demo_repeat
```

演示证据等级为 `ENGINEERING_ONLY`；合成收益不能写入策略结论。

### 3. 下载免费真实数据

推荐研究范围为中证 800，即历史时点沪深 300 + 中证 500：

```powershell
ashare-quant sync-data --universe csi800 --start 2015-01-01 --data-dir data/csi800_pit_market --skip-historical-status
```

下载器会保存逐股票 Parquet、五类基准、成分快照、公司行动和 `metadata/download_report.json`。默认刷新模式是 `missing`，中断或接口暂时失败后可原命令重跑，不必从头下载。

`--skip-historical-status` 是免费数据降级：保留历史成分近似，但不声称已获得完整逐日 ST/停牌状态。若下载报告不满足正式门禁，后续命令必须显式加入 `--allow-non-research-ready`，最终证据等级不会被隐藏。

### 4. 先审计，再研究

```powershell
ashare-quant audit-data --data-dir data/csi800_pit_market
```

审计会重新读取落盘文件，检查：

- 股票代码为六位字符串，且为沪深 A 股股票而非基金代码；
- 日期、重复行、OHLC 关系、价格与成交量合法；
- 原始价/复权价一致性；
- 单股票历史长度、股票池规模与基准覆盖；
- 下载失败、缺失字段、历史成分和证据等级。

审计失败时不要直接回测。先查看 `data/csi800_pit_market/metadata/download_report.json`，重跑同步或修复输入。

### 5. 生成冻结的 Walk-Forward 预测

机构配置是研究主配置：

```powershell
ashare-quant run --data data/csi800_pit_market --benchmarks data/csi800_pit_market/benchmarks.parquet --config configs/low_turnover_institution.toml --output artifacts/csi800_pit_walk_forward --walk-forward --allow-non-research-ready
```

关键产物包括：

- `walk_forward_folds.csv`：每折训练/验证/测试边界；
- `walk_forward_predictions.csv`：仅样本外的逐股票分数；
- `walk_forward_*_nav.csv`：基线与 XGBoost 净值；
- `run_manifest.json` 或对应汇总：输入、配置、质量门禁和警告。

不要用测试期结果重新选择 XGBoost 超参数。V1 超参数与时间切分已经冻结。

### 6. 机构版与个人版低换手回测

两套组合共享同一份冻结的样本外分数，只改变账户和组合约束：

```powershell
python scripts/run_low_turnover_scenarios.py
```

输出位于 `artifacts/low_turnover/`。机构版初始资金 500 万元、目标 50 股；个人版 30 万元、目标 5 股。订单、拒单、成交、现金、持仓和 NAV 都会保存。

### 7. 正式稳健性验证

```powershell
python scripts/run_strategy_validation.py
```

这一步会运行参数网格、0–3 倍成本、因子组消融、年度/市场状态/规模/回撤/执行归因。它会重训消融模型，预计是整套复现中最耗时的阶段。输出位于 `artifacts/strategy_validation/`。

### 8. 导出适合 Git 的证据包和图表

```powershell
python scripts/export_v1_results.py
```

该命令只导出小型汇总 CSV 到 `results/v1/`，并生成 `docs/assets/*.svg`。不会提交原始行情、逐股票预测、订单、持仓或完整账本。

### 9. 工程质量检查

```powershell
ruff check src tests scripts
python -m compileall -q src scripts tests
pytest
```

CI 执行同一组检查。确定性回归还要求相同输入和配置在独立进程里产生相同的期末资产、收益、Sharpe 和最大回撤；订单并列时以股票代码作为稳定排序键。

### 10. 预期冻结结果

在 V1 本地数据快照上，`results/v1/headline_results.csv` 应显示：

- 机构版：期末 `7,684,181.67` 元，累计 `53.68%`，最大回撤 `-21.84%`；
- 个人版：期末 `463,266.77` 元，累计 `54.42%`，最大回撤 `-44.16%`。

免费接口返回内容可能随后修订，因此从今天重新下载的数据不保证逐分逐厘一致。可复现性分成两层：代码 + 冻结本地快照应严格确定；代码 + 重新下载免费数据应保证方法一致，但结果可能有数据版本漂移。

### 11. Git 数据政策

应提交：源代码、配置、测试、文档、SVG 图和 `results/v1` 汇总。不要提交：`data/` 下真实行情、`artifacts/` 下预测/模型/订单/持仓账本、账户持仓文件、任何数据供应商密钥。

## English quick guide

1. Create Python 3.11+ environment and install: `python -m pip install -e ".[data,dev]"`.
2. Run the deterministic software demo: `python scripts/run_demo.py`.
3. Sync free point-in-time CSI 800 approximation with `ashare-quant sync-data ...`.
4. Run `ashare-quant audit-data` before any research claim.
5. Produce purged walk-forward OOS predictions with `ashare-quant run ... --walk-forward`.
6. Run frozen institution/personal portfolios: `python scripts/run_low_turnover_scenarios.py`.
7. Run parameter, cost, ablation, and attribution tests: `python scripts/run_strategy_validation.py`.
8. Export Git-safe evidence: `python scripts/export_v1_results.py`.
9. Verify engineering quality with Ruff, compileall, and Pytest.

Free-source revisions can change newly downloaded observations. Exact numerical reproducibility therefore requires the same local data snapshot; methodological reproducibility requires the code, frozen configs, data manifest, and recorded evidence grade. Never interpret `LIMITED_RESEARCH` or `ENGINEERING_ONLY` as production-grade investment evidence.
