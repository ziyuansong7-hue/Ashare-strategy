from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.analytics.performance import attach_benchmark_nav, performance_metrics
from ashare_quant.config import CostConfig, PortfolioConfig


def rank_ic_summary(predictions: pd.DataFrame, score_column: str) -> dict[str, float]:
    """Summarize cross-sectional predictive quality without mixing dates."""
    values = []
    for _, group in predictions.groupby("date", sort=True):
        value = group[score_column].corr(group["target_excess_return"], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    series = pd.Series(values, dtype=float)
    deviation = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return {
        "mean_rank_ic": float(series.mean()) if not series.empty else np.nan,
        "rank_ic_ir": float(series.mean() / deviation) if deviation > 0 else np.nan,
        "positive_rank_ic_rate": float((series > 0).mean()) if not series.empty else np.nan,
        "rebalance_dates": len(series),
    }


def selected_return_summary(
    predictions: pd.DataFrame,
    score_column: str,
    *,
    top_k: int,
) -> dict[str, float]:
    """Evaluate the model's selected cross-section before execution effects."""
    selected_rows = []
    remainder_rows = []
    for _, group in predictions.groupby("date", sort=True):
        selected = group.nlargest(top_k, score_column)
        selected_rows.append(selected["target_excess_return"])
        remainder_rows.append(group.loc[~group.index.isin(selected.index), "target_excess_return"])
    selected_values = pd.concat(selected_rows, ignore_index=True)
    remainder_values = pd.concat(remainder_rows, ignore_index=True)
    return {
        "selected_mean_excess_return": float(selected_values.mean()),
        "remainder_mean_excess_return": float(remainder_values.mean()),
        "selection_spread": float(selected_values.mean() - remainder_values.mean()),
        "selected_positive_rate": float((selected_values > 0).mean()),
    }


def summarize_backtest(
    result: Any,
    *,
    portfolio: PortfolioConfig,
    benchmarks: pd.DataFrame | None,
) -> dict[str, Any]:
    """Return comparable performance and execution metrics for a validation scenario."""
    nav = attach_benchmark_nav(result.nav, benchmarks, initial_nav=portfolio.initial_cash)
    metrics = performance_metrics(nav, initial_nav=portfolio.initial_cash)
    orders = result.orders
    rebalances = result.rebalance_summary
    fees = float(orders["fees"].sum()) if not orders.empty else 0.0
    filled = int((orders["filled_quantity"] > 0).sum()) if not orders.empty else 0
    rejected = int((orders["filled_quantity"] == 0).sum()) if not orders.empty else 0
    cash_weight = pd.to_numeric(nav["cash"], errors="coerce") / pd.to_numeric(
        nav["nav"], errors="coerce"
    )
    metrics.update(
        {
            "total_fees": fees,
            "fees_as_initial_cash": fees / portfolio.initial_cash,
            "filled_orders": filled,
            "rejected_orders": rejected,
            "average_turnover": float(rebalances["turnover"].mean()),
            "median_turnover": float(rebalances["turnover"].median()),
            "average_entries_after_initial": float(rebalances["entered_count"].iloc[1:].mean()),
            "average_exits_after_initial": float(rebalances["exited_count"].iloc[1:].mean()),
            "average_cash_weight": float(cash_weight.mean()),
            "maximum_cash_weight": float(cash_weight.max()),
        }
    )
    return metrics


def flatten_backtest_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten the benchmark subset used by robustness tables."""
    row = {key: value for key, value in metrics.items() if key != "benchmarks"}
    for benchmark in [
        "csi300",
        "csi500",
        "csi800",
        "sse_composite",
        "szse_component",
        "universe_equal_weight",
    ]:
        values = metrics.get("benchmarks", {}).get(benchmark, {})
        for metric in ["total_return", "excess_total_return", "alpha_annualized", "beta"]:
            if metric in values:
                row[f"{benchmark}_{metric}"] = values[metric]
    return row


def scale_costs(costs: CostConfig, multiplier: float) -> CostConfig:
    if multiplier < 0:
        raise ValueError("cost multiplier must be non-negative")
    return CostConfig(
        **{key: float(value) * multiplier for key, value in asdict(costs).items()}
    )


def parameter_robustness_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Describe the distribution of outcomes instead of selecting the best row."""
    rows = []
    for profile, group in results.groupby("profile", sort=True):
        row: dict[str, Any] = {"profile": profile, "scenarios": len(group)}
        for metric in [
            "total_return",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "average_turnover",
            "fees_as_initial_cash",
        ]:
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_max"] = float(group[metric].max())
        row["positive_return_rate"] = float((group["total_return"] > 0).mean())
        for benchmark in ["csi300", "sse_composite", "szse_component", "csi500"]:
            column = f"{benchmark}_excess_total_return"
            row[f"beat_{benchmark}_rate"] = float((group[column] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def portfolio_regime_attribution(
    nav: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    initial_cash: float,
    lookback_days: int = 120,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Attribute realized portfolio returns to causal CSI800 market regimes."""
    index = benchmarks.loc[
        benchmarks["benchmark_key"].eq("csi800"), ["date", "close"]
    ].copy()
    index["date"] = pd.to_datetime(index["date"]).dt.normalize()
    index = index.sort_values("date").drop_duplicates("date", keep="last")
    index["trailing_return"] = index["close"] / index["close"].shift(lookback_days) - 1.0
    index["regime"] = np.select(
        [index["trailing_return"] > threshold, index["trailing_return"] < -threshold],
        ["BULL", "BEAR"],
        default="SIDEWAYS",
    )
    dated = nav[["date", "nav"]].copy()
    dated["date"] = pd.to_datetime(dated["date"]).dt.normalize()
    values = pd.to_numeric(dated["nav"], errors="coerce")
    dated["strategy_return"] = values.pct_change(fill_method=None)
    dated.loc[dated.index[0], "strategy_return"] = values.iloc[0] / initial_cash - 1.0
    dated = pd.merge_asof(
        dated.sort_values("date"),
        index[["date", "trailing_return", "regime"]],
        on="date",
        direction="backward",
    )
    total_log_return = float(np.log1p(dated["strategy_return"]).sum())
    rows = []
    for regime, group in dated.groupby("regime", sort=True):
        returns = group["strategy_return"].dropna()
        conditional_nav = (1.0 + returns).cumprod()
        drawdown = conditional_nav / conditional_nav.cummax() - 1.0
        regime_log_return = float(np.log1p(returns).sum())
        rows.append(
            {
                "regime": regime,
                "trading_days": len(returns),
                "mean_trailing_csi800_return": float(group["trailing_return"].mean()),
                "mean_daily_return": float(returns.mean()),
                "conditional_annualized_return": float((1.0 + returns.mean()) ** 252 - 1.0),
                "positive_day_rate": float((returns > 0).mean()),
                "compounded_return_on_regime_days": float(conditional_nav.iloc[-1] - 1.0),
                "log_return_contribution": regime_log_return,
                "share_of_total_log_return": regime_log_return / total_log_return
                if abs(total_log_return) > 1e-12
                else np.nan,
                "conditional_max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def size_exposure_attribution(result: Any) -> pd.DataFrame:
    """Measure realized LARGE/MID exposure from the actual position ledger."""
    if result.positions.empty:
        return pd.DataFrame()
    positions = result.positions.copy()
    nav = result.nav[["date", "nav"]].copy()
    positions["date"] = pd.to_datetime(positions["date"]).dt.normalize()
    nav["date"] = pd.to_datetime(nav["date"]).dt.normalize()
    daily = positions.groupby(["date", "size_bucket"], as_index=False)["market_value"].sum()
    daily = daily.merge(nav, on="date", how="left")
    daily["portfolio_weight"] = daily["market_value"] / daily["nav"]
    rows = []
    for bucket, group in daily.groupby("size_bucket", sort=True):
        rows.append(
            {
                "size_bucket": str(bucket),
                "average_weight": float(group["portfolio_weight"].mean()),
                "minimum_weight": float(group["portfolio_weight"].min()),
                "maximum_weight": float(group["portfolio_weight"].max()),
                "average_market_value": float(group["market_value"].mean()),
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows)


def maximum_drawdown_episode(nav: pd.DataFrame, *, initial_cash: float) -> dict[str, Any]:
    """Locate the peak, trough, and recovery of the worst realized drawdown."""
    dated = nav[["date", "nav"]].copy()
    dated["date"] = pd.to_datetime(dated["date"]).dt.normalize()
    initial = pd.DataFrame(
        {"date": [dated["date"].iloc[0] - pd.Timedelta(days=1)], "nav": [initial_cash]}
    )
    series = pd.concat([initial, dated], ignore_index=True)
    running_peak = series["nav"].cummax()
    drawdown = series["nav"] / running_peak - 1.0
    trough_index = int(drawdown.idxmin())
    peak_value = float(running_peak.iloc[trough_index])
    peak_index = int(series.loc[:trough_index, "nav"].idxmax())
    after = series.loc[trough_index + 1 :]
    recovered = after.loc[after["nav"] >= peak_value]
    recovery_date = recovered["date"].iloc[0] if not recovered.empty else pd.NaT
    return {
        "peak_date": series.loc[peak_index, "date"],
        "peak_nav": float(series.loc[peak_index, "nav"]),
        "trough_date": series.loc[trough_index, "date"],
        "trough_nav": float(series.loc[trough_index, "nav"]),
        "max_drawdown": float(drawdown.iloc[trough_index]),
        "recovery_date": recovery_date,
        "recovered": bool(not recovered.empty),
    }
