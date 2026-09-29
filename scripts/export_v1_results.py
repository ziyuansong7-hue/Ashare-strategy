from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#475569"]


def _esc(value: object) -> str:
    return html.escape(str(value))


def _line_chart(
    frame: pd.DataFrame,
    series: list[tuple[str, str]],
    *,
    title: str,
    y_label: str,
    output: Path,
    x_labels: list[str] | None = None,
) -> None:
    width, height = 1000, 520
    left, right, top, bottom = 78, 24, 58, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column, _ in series])
    y_min, y_max = float(values.min()), float(values.max())
    padding = max((y_max - y_min) * 0.08, 1.0)
    y_min -= padding
    y_max += padding

    def x(index: int) -> float:
        return left + plot_width * index / max(len(frame) - 1, 1)

    def y(value: float) -> float:
        return top + plot_height * (y_max - value) / max(y_max - y_min, 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{_esc(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="30" font-family="Arial,sans-serif" font-size="22" '
            f'font-weight="600" fill="#0f172a">{_esc(title)}</text>'
        ),
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        py = y(value)
        parts.extend(
            [
                (
                    f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" '
                    'stroke="#e2e8f0" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" '
                    f'font-family="Arial,sans-serif" font-size="12" fill="#475569">{value:.0f}</text>'
                ),
            ]
        )
    if x_labels is None:
        date_values = pd.to_datetime(frame["date"])
        x_labels = [value.strftime("%Y-%m") for value in date_values]
    if len(x_labels) != len(frame):
        raise ValueError("x_labels must contain one label per row")
    tick_indexes = sorted(
        {0, len(frame) // 4, len(frame) // 2, 3 * len(frame) // 4, len(frame) - 1}
    )
    for index in tick_indexes:
        px = x(index)
        parts.append(
            f'<text x="{px:.1f}" y="{height - 28}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#475569">'
            f"{_esc(x_labels[index])}</text>"
        )
    for series_index, (column, label) in enumerate(series):
        points = " ".join(
            f"{x(index):.1f},{y(float(value)):.1f}"
            for index, value in enumerate(pd.to_numeric(frame[column], errors="coerce"))
            if pd.notna(value)
        )
        color = COLORS[series_index % len(COLORS)]
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_x = left + series_index * 190
        parts.extend(
            [
                (
                    f'<line x1="{legend_x}" y1="48" x2="{legend_x + 24}" y2="48" '
                    f'stroke="{color}" stroke-width="3"/>'
                ),
                (
                    f'<text x="{legend_x + 31}" y="52" font-family="Arial,sans-serif" '
                    f'font-size="12" fill="#334155">{_esc(label)}</text>'
                ),
            ]
        )
    parts.extend(
        [
            (
                f'<text x="18" y="{top + plot_height / 2:.1f}" transform="rotate(-90 18 '
                f'{top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial,sans-serif" '
                f'font-size="13" fill="#334155">{_esc(y_label)}</text>'
            ),
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")


def _ablation_chart(frame: pd.DataFrame, output: Path) -> None:
    scenarios = [
        "full_model",
        "remove_momentum",
        "remove_trend",
        "remove_reversal",
        "remove_risk",
        "remove_price_volume",
    ]
    labels = ["Full", "−Momentum", "−Trend", "−Reversal", "−Risk", "−Price/volume"]
    pivot = frame.pivot(index="scenario", columns="profile", values="total_return").reindex(
        scenarios
    )
    width, height = 1000, 520
    left, right, top, bottom = 150, 30, 62, 48
    plot_width = width - left - right
    row_height = (height - top - bottom) / len(scenarios)
    maximum = max(float(pivot.max().max()), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        "<title>Factor-group ablation — cumulative return</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="30" font-family="Arial,sans-serif" font-size="22" '
            'font-weight="600" fill="#0f172a">Factor-group ablation — cumulative return</text>'
        ),
    ]
    for tick in range(6):
        value = maximum * tick / 5
        px = left + plot_width * value / maximum
        parts.extend(
            [
                (
                    f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" '
                    'stroke="#e2e8f0"/>'
                ),
                (
                    f'<text x="{px:.1f}" y="{height - 20}" text-anchor="middle" '
                    f'font-family="Arial,sans-serif" font-size="12" fill="#475569">{value:.0%}</text>'
                ),
            ]
        )
    for index, (scenario, label) in enumerate(zip(scenarios, labels, strict=False)):
        center = top + row_height * (index + 0.5)
        parts.append(
            f'<text x="{left - 12}" y="{center + 4:.1f}" text-anchor="end" '
            f'font-family="Arial,sans-serif" font-size="12" fill="#334155">{_esc(label)}</text>'
        )
        for offset, profile in [(-10, "institution"), (10, "personal")]:
            value = float(pivot.loc[scenario, profile])
            bar_width = max(plot_width * value / maximum, 1)
            color = COLORS[0] if profile == "institution" else COLORS[1]
            parts.append(
                f'<rect x="{left}" y="{center + offset - 7:.1f}" width="{bar_width:.1f}" '
                f'height="14" fill="{color}" rx="2"/>'
            )
            parts.append(
                f'<text x="{left + bar_width + 7:.1f}" y="{center + offset + 4:.1f}" '
                f'font-family="Arial,sans-serif" font-size="11" fill="#334155">{value:.1%}</text>'
            )
    parts.extend(
        [
            f'<rect x="{left}" y="44" width="14" height="10" fill="{COLORS[0]}"/>',
            (
                f'<text x="{left + 20}" y="53" font-family="Arial,sans-serif" font-size="12" '
                'fill="#334155">Institution</text>'
            ),
            f'<rect x="{left + 110}" y="44" width="14" height="10" fill="{COLORS[1]}"/>',
            (
                f'<text x="{left + 130}" y="53" font-family="Arial,sans-serif" font-size="12" '
                'fill="#334155">Personal</text>'
            ),
            "</svg>",
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the compact, Git-trackable V1 evidence pack."
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/v1"))
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)

    low = args.artifacts / "low_turnover"
    validation = args.artifacts / "strategy_validation"
    comparison = pd.read_csv(low / "comparison.csv")
    comparison.to_csv(args.output / "headline_results.csv", index=False)

    annual = pd.read_csv(validation / "attribution_annual_returns.csv")
    annual.to_csv(args.output / "annual_returns.csv", index=False)
    parameter = pd.read_csv(validation / "parameter_stability_summary.csv")
    parameter.to_csv(args.output / "parameter_stability_summary.csv", index=False)

    cost = pd.read_csv(validation / "cost_stress_results.csv")
    cost_columns = [
        "profile",
        "cost_multiplier",
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "fees_as_initial_cash",
        "csi300_excess_total_return",
        "csi500_excess_total_return",
        "sse_composite_excess_total_return",
        "szse_component_excess_total_return",
    ]
    cost[cost_columns].to_csv(args.output / "cost_stress_summary.csv", index=False)

    ablation = pd.read_csv(validation / "factor_ablation_portfolio_results.csv")
    ablation_columns = [
        "profile",
        "scenario",
        "removed_group",
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
    ]
    ablation[ablation_columns].to_csv(args.output / "factor_ablation_summary.csv", index=False)
    pd.read_csv(validation / "attribution_market_regimes.csv").to_csv(
        args.output / "market_regime_attribution.csv", index=False
    )

    institution = pd.read_csv(low / "institution" / "xgboost_nav.csv")
    personal = pd.read_csv(low / "personal" / "xgboost_nav.csv")
    institution["date"] = pd.to_datetime(institution["date"])
    personal["date"] = pd.to_datetime(personal["date"])
    institution = institution.set_index("date").resample("ME").last().reset_index()
    personal = personal.set_index("date").resample("ME").last().reset_index()
    nav = institution[["date", "nav", "csi300_nav", "csi500_nav"]].rename(
        columns={"nav": "institution"}
    )
    nav = nav.merge(personal[["date", "nav"]].rename(columns={"nav": "personal"}), on="date")
    for column, initial in [
        ("institution", 5_000_000.0),
        ("personal", 300_000.0),
        ("csi300_nav", 5_000_000.0),
        ("csi500_nav", 5_000_000.0),
    ]:
        nav[column] = nav[column] / initial * 100.0
    nav.to_csv(args.output / "monthly_normalized_nav.csv", index=False, float_format="%.6f")

    _line_chart(
        nav,
        [
            ("institution", "Institution V1"),
            ("personal", "Personal V1"),
            ("csi300_nav", "CSI 300"),
            ("csi500_nav", "CSI 500"),
        ],
        title="Strategy V1 out-of-sample NAV",
        y_label="Normalized NAV (start = 100)",
        output=args.assets / "v1-nav-comparison.svg",
    )
    cost_plot = (
        cost[["profile", "cost_multiplier", "total_return"]]
        .pivot(index="cost_multiplier", columns="profile", values="total_return")
        .reset_index()
    )
    cost_plot["institution"] *= 100
    cost_plot["personal"] *= 100
    _line_chart(
        cost_plot,
        [("institution", "Institution"), ("personal", "Personal")],
        title="Transaction-cost stress test",
        y_label="Cumulative return (%)",
        output=args.assets / "v1-cost-stress.svg",
        x_labels=[f"{value:g}×" for value in cost_plot["cost_multiplier"]],
    )
    _ablation_chart(ablation, args.assets / "v1-factor-ablation.svg")

    manifest = """# V1证据包 / Strategy V1 evidence pack

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
"""
    (args.output / "README.md").write_text(manifest, encoding="utf-8")
    print(f"Exported V1 evidence to {args.output.resolve()}")
    print(f"Generated documentation charts under {args.assets.resolve()}")


if __name__ == "__main__":
    main()
