# V1 冻结清单 / V1 Freeze Manifest

## 中文

策略版本 `v1.0.0` 于 2026-09-27 冻结。冻结意味着：当前样本外区间不再用于选择更高收益的因子、模型超参数或组合参数；任何利用该测试期重新优化的方案必须作为新版本，并通过新增的未见数据验证。

### 冻结对象

| 对象 | 冻结值 |
|---|---|
| 包版本 | `1.0.0` |
| 样本外区间 | `2019-08-01` 至 `2026-09-04` |
| 决策频率 | 月末信号、下一交易日开盘执行 |
| 股票池 | 历史月末沪深 300 + 中证 500 成分近似 |
| 模型 | `XGBRanker`，`random_state=42` |
| Walk-Forward | expanding；36 月训练、12 月验证、12 月测试、12 月步长 |
| 机构组合 | 500 万元、50 股、退出排名 100、每月最多替换 10 股、±0.5% 调仓带 |
| 个人组合 | 30 万元、5 股、退出排名 12、每月最多替换 1 股、±3% 调仓带 |
| 证据等级 | `LIMITED_RESEARCH` |

### 冻结结果

| 账户 | 期末资产 | 累计收益 | 年化收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|---:|---:|
| 机构版 | 7,684,181.67 元 | 53.68% | 6.49% | 0.448 | -21.84% |
| 个人版 | 463,266.77 元 | 54.42% | 6.57% | 0.415 | -44.16% |

冻结结论同样包括负面证据：两套组合均未跑赢中证 500 与历史时点股票池等权；个人版未通过因子规格稳定性判断。

### 本地输入与配置指纹

以下大文件不进入 Git，但哈希记录了生成 V1 证据时使用的本地快照。重新下载免费数据时发生哈希变化是正常的数据版本漂移，必须生成新的研究运行清单，不能声称逐数值复现本快照。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `configs/low_turnover_institution.toml` | 1,082 | `75f534b72bc2a9a36c47e09a98b9897d49e60002c7bfc736e26a5903ec6867a3` |
| `configs/low_turnover_personal.toml` | 1,075 | `ff9fbd44b6df218eacce57954d5d6e5ffbf41144e36cdeae4f13a90cf98f05a5` |
| `artifacts/csi800_pit_walk_forward/walk_forward_predictions.csv` | 66,751,324 | `50f9eb45e77074ca29640d73ea64f9b3d75439bfe6d1d7b63a120be5e7c1d555` |
| `data/csi800_pit_market/metadata/download_report.json` | 519,240 | `38ef3302dc77c955051bdfdb4509db784bb7b561a1b38eea141cc79405ebecdb` |
| `artifacts/low_turnover/comparison.csv` | 1,023 | `bb303c81ae30602e8ca2d94edc20517f52ca4fee86df40fba3e0eb36108be55a` |
| `artifacts/strategy_validation/validation_manifest.json` | 4,522 | `2b9e52fc63613bcbeb20d479a9e04b13cf994b373318deb79c3df2dba1d68098` |

PowerShell 核对示例：

```powershell
Get-FileHash -Algorithm SHA256 configs/low_turnover_institution.toml
Get-FileHash -Algorithm SHA256 artifacts/csi800_pit_walk_forward/walk_forward_predictions.csv
```

### 发布步骤

仓库第一次提交后，可创建带说明的 Git 标签：

```powershell
git tag -a v1.0.0 -m "Freeze Strategy V1 research and evidence"
git push origin v1.0.0
```

标签只应指向通过 CI、且包含 `results/v1/` 证据包的提交。

## English

Strategy `v1.0.0` freezes the model, walk-forward boundaries, portfolio policies, test interval, evidence grade, and both positive and negative findings. The frozen test interval must not be reused to select better factors or parameters. Any such change is a new research version and requires genuinely unseen data.

Large local data and ledgers are excluded from Git; their hashes above identify the snapshot used for the published summaries. A fresh free-source download may legitimately drift, but must receive a new run manifest. After the initial repository commit passes CI, create the annotated `v1.0.0` Git tag shown above.
