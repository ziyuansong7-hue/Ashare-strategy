from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.config import MODEL_FEATURES

from .performance import annual_return_table, attach_benchmark_nav


def _daily_rank_ic(frame: pd.DataFrame, score_column: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        value = group[score_column].corr(group["target_excess_return"], method="spearman")
        if pd.notna(value):
            values[pd.Timestamp(date)] = float(value)
    return pd.Series(values, dtype=float)


def factor_year_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report out-of-sample factor RankIC by calendar year."""
    dated = predictions.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    rows: list[dict[str, Any]] = []
    for year, year_frame in dated.groupby(dated["date"].dt.year, sort=True):
        for feature in MODEL_FEATURES:
            if feature not in year_frame:
                continue
            daily = _daily_rank_ic(
                year_frame.rename(columns={feature: "_factor_score"}), "_factor_score"
            )
            rows.append(
                {
                    "year": int(year),
                    "feature": feature.removesuffix("_z"),
                    "mean_rank_ic": float(daily.mean()) if not daily.empty else np.nan,
                    "positive_ic_rate": float((daily > 0).mean()) if not daily.empty else np.nan,
                    "rebalance_dates": len(daily),
                    "observations": int(year_frame[feature].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def selection_report(
    predictions: pd.DataFrame,
    score_columns: list[str],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Compare the model-selected group with all non-selected eligible stocks."""
    dated = predictions.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    frames: list[tuple[str, pd.DataFrame]] = [("ALL", dated)]
    frames.extend((str(year), group) for year, group in dated.groupby(dated["date"].dt.year))
    rows: list[dict[str, Any]] = []
    for period, period_frame in frames:
        for score_column in score_columns:
            selected_index: set[int] = set()
            for _, group in period_frame.groupby("date", sort=True):
                selected_index.update(group.nlargest(top_k, score_column).index.tolist())
            selected = period_frame.loc[period_frame.index.isin(selected_index)]
            remainder = period_frame.loc[~period_frame.index.isin(selected_index)]
            for label, group in [("selected", selected), ("not_selected", remainder)]:
                rows.append(
                    {
                        "period": period,
                        "model": score_column.removesuffix("_score"),
                        "group": label,
                        "mean_forward_return": float(group["forward_return"].mean()),
                        "mean_excess_return": float(group["target_excess_return"].mean()),
                        "positive_excess_rate": float((group["target_excess_return"] > 0).mean()),
                        "observations": len(group),
                        "rebalance_dates": int(group["date"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def segment_signal_attribution(
    predictions: pd.DataFrame,
    score_columns: list[str],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Attribute selected-group forward returns to LARGE/MID signal exposure."""
    if "size_bucket" not in predictions:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        daily_rows: list[dict[str, Any]] = []
        for date, group in predictions.groupby("date", sort=True):
            selected = group.nlargest(top_k, score_column)
            count = max(len(selected), 1)
            for bucket, segment in selected.groupby("size_bucket", sort=True):
                share = len(segment) / count
                daily_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "size_bucket": str(bucket),
                        "selected_count": len(segment),
                        "selected_share": share,
                        "mean_forward_return": float(segment["forward_return"].mean()),
                        "mean_excess_return": float(segment["target_excess_return"].mean()),
                        "forward_return_contribution": share * float(segment["forward_return"].mean()),
                        "excess_return_contribution": share
                        * float(segment["target_excess_return"].mean()),
                    }
                )
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            continue
        periods: list[tuple[str, pd.DataFrame]] = [("ALL", daily)]
        periods.extend(
            (str(year), group) for year, group in daily.groupby(daily["date"].dt.year)
        )
        for period, period_frame in periods:
            for bucket, segment in period_frame.groupby("size_bucket", sort=True):
                rows.append(
                    {
                        "period": period,
                        "model": score_column.removesuffix("_score"),
                        "size_bucket": bucket,
                        "average_selected_count": float(segment["selected_count"].mean()),
                        "average_selected_share": float(segment["selected_share"].mean()),
                        "mean_forward_return": float(segment["mean_forward_return"].mean()),
                        "mean_excess_return": float(segment["mean_excess_return"].mean()),
                        "average_forward_return_contribution": float(
                            segment["forward_return_contribution"].mean()
                        ),
                        "average_excess_return_contribution": float(
                            segment["excess_return_contribution"].mean()
                        ),
                        "rebalance_dates": int(segment["date"].nunique()),
                    }
                )
    return pd.DataFrame(rows)


def market_regime_report(
    predictions: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    score_columns: list[str],
    *,
    top_k: int,
    lookback_days: int = 120,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Measure model quality after causal bull/sideways/bear classification."""
    if benchmarks is None or benchmarks.empty:
        return pd.DataFrame()
    index = benchmarks.loc[benchmarks["benchmark_key"].eq("csi800"), ["date", "close"]].copy()
    if index.empty:
        return pd.DataFrame()
    index["date"] = pd.to_datetime(index["date"]).astype("datetime64[ns]")
    index = index.sort_values("date").drop_duplicates("date", keep="last")
    index["trailing_return"] = index["close"] / index["close"].shift(lookback_days) - 1.0
    index["regime"] = np.select(
        [index["trailing_return"] > threshold, index["trailing_return"] < -threshold],
        ["BULL", "BEAR"],
        default="SIDEWAYS",
    )
    dated = predictions.copy()
    dated["date"] = pd.to_datetime(dated["date"]).astype("datetime64[ns]")
    dated = pd.merge_asof(
        dated.sort_values("date"),
        index[["date", "trailing_return", "regime"]],
        on="date",
        direction="backward",
    )
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        for regime, regime_frame in dated.groupby("regime", sort=True):
            daily_ic = _daily_rank_ic(regime_frame, score_column)
            selected_returns: list[float] = []
            for _, group in regime_frame.groupby("date", sort=True):
                selected_returns.append(
                    float(group.nlargest(top_k, score_column)["target_excess_return"].mean())
                )
            selected_series = pd.Series(selected_returns, dtype=float)
            rows.append(
                {
                    "model": score_column.removesuffix("_score"),
                    "regime": regime,
                    "mean_trailing_csi800_return": float(regime_frame["trailing_return"].mean()),
                    "mean_rank_ic": float(daily_ic.mean()) if not daily_ic.empty else np.nan,
                    "positive_ic_rate": float((daily_ic > 0).mean()) if not daily_ic.empty else np.nan,
                    "selected_mean_excess_return": float(selected_series.mean()),
                    "selected_positive_rate": float((selected_series > 0).mean()),
                    "rebalance_dates": int(regime_frame["date"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def replacement_report(
    predictions: pd.DataFrame,
    score_columns: list[str],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Test whether newly selected names subsequently beat displaced names."""
    rows: list[dict[str, Any]] = []
    for score_column in score_columns:
        previous: set[str] | None = None
        for date, group in predictions.groupby("date", sort=True):
            selected = group.nlargest(top_k, score_column)
            current = set(selected["code"].astype(str))
            if previous is not None:
                entrants = current.difference(previous)
                exits = previous.difference(current)
                entrant_rows = group.loc[group["code"].astype(str).isin(entrants)]
                exit_rows = group.loc[group["code"].astype(str).isin(exits)]
                entrant_return = float(entrant_rows["target_excess_return"].mean())
                exit_return = float(exit_rows["target_excess_return"].mean())
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "model": score_column.removesuffix("_score"),
                        "entrants": len(entrants),
                        "observable_exits": len(exit_rows),
                        "entrant_mean_excess_return": entrant_return,
                        "exit_mean_excess_return": exit_return,
                        "replacement_spread": entrant_return - exit_return,
                    }
                )
            previous = current
    return pd.DataFrame(rows)


def write_research_diagnostics(
    predictions: pd.DataFrame,
    model_results: dict[str, Any],
    output_dir: str | Path,
    *,
    benchmarks: pd.DataFrame | None,
    top_k: int,
    initial_cash: float,
    prefix: str,
) -> dict[str, Any]:
    """Write a reusable diagnosis pack for model, portfolio, and execution failures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    score_columns = [column for column in ["baseline_score", "xgboost_score"] if column in predictions]

    factor_year = factor_year_report(predictions)
    selection = selection_report(predictions, score_columns, top_k=top_k)
    segments = segment_signal_attribution(predictions, score_columns, top_k=top_k)
    regimes = market_regime_report(
        predictions, benchmarks, score_columns, top_k=top_k
    )
    replacements = replacement_report(predictions, score_columns, top_k=top_k)

    factor_year.to_csv(output / f"{prefix}_factor_year_report.csv", index=False)
    selection.to_csv(output / f"{prefix}_selection_report.csv", index=False)
    segments.to_csv(output / f"{prefix}_segment_attribution.csv", index=False)
    regimes.to_csv(output / f"{prefix}_market_regime_report.csv", index=False)
    replacements.to_csv(output / f"{prefix}_replacement_report.csv", index=False)

    execution_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for model_name, result in model_results.items():
        nav = attach_benchmark_nav(result.nav, benchmarks, initial_nav=initial_cash)
        nav_values = pd.to_numeric(nav["nav"], errors="coerce")
        cash_weight = pd.to_numeric(nav["cash"], errors="coerce") / nav_values
        orders = result.orders
        rebalances = result.rebalance_summary
        fees = float(orders["fees"].sum()) if not orders.empty else 0.0
        rejected = int((orders["filled_quantity"] == 0).sum()) if not orders.empty else 0
        execution = {
            "model": model_name,
            "total_fees": fees,
            "fees_as_initial_cash": fees / initial_cash,
            "filled_orders": int((orders["filled_quantity"] > 0).sum()) if not orders.empty else 0,
            "rejected_orders": rejected,
            "average_cash_weight": float(cash_weight.mean()),
            "maximum_cash_weight": float(cash_weight.max()),
            "average_turnover": float(rebalances["turnover"].mean())
            if not rebalances.empty
            else np.nan,
            "average_entries": float(rebalances["entered_count"].mean())
            if not rebalances.empty
            else np.nan,
            "average_exits": float(rebalances["exited_count"].mean())
            if not rebalances.empty
            else np.nan,
        }
        execution_rows.append(execution)
        if not orders.empty:
            rejected_orders = orders.loc[orders["filled_quantity"] == 0]
            for reason, count in rejected_orders["reason"].fillna("UNKNOWN").value_counts().items():
                rejection_rows.append(
                    {"model": model_name, "reason": str(reason), "rejected_orders": int(count)}
                )

        annual = annual_return_table(nav, initial_nav=initial_cash)
        for row in annual.to_dict("records"):
            strategy_return = float(row["nav"])
            for benchmark, benchmark_return in row.items():
                if benchmark in {"year", "nav"} or pd.isna(benchmark_return):
                    continue
                annual_rows.append(
                    {
                        "model": model_name,
                        "year": int(row["year"]),
                        "benchmark": benchmark,
                        "strategy_return": strategy_return,
                        "benchmark_return": float(benchmark_return),
                        "excess_return": strategy_return - float(benchmark_return),
                    }
                )

        model_selection = selection.loc[
            (selection["period"] == "ALL")
            & (selection["model"] == model_name)
        ]
        selected_mean = model_selection.loc[
            model_selection["group"] == "selected", "mean_excess_return"
        ]
        model_replacements = replacements.loc[replacements["model"] == model_name]
        summary[model_name] = {
            **{key: value for key, value in execution.items() if key != "model"},
            "selected_mean_excess_return": float(selected_mean.iloc[0])
            if not selected_mean.empty
            else np.nan,
            "mean_replacement_spread": float(model_replacements["replacement_spread"].mean())
            if not model_replacements.empty
            else np.nan,
            "positive_replacement_rate": float(
                (model_replacements["replacement_spread"] > 0).mean()
            )
            if not model_replacements.empty
            else np.nan,
        }

    pd.DataFrame(execution_rows).to_csv(
        output / f"{prefix}_execution_diagnostics.csv", index=False
    )
    pd.DataFrame(rejection_rows).to_csv(
        output / f"{prefix}_rejection_report.csv", index=False
    )
    pd.DataFrame(annual_rows).to_csv(
        output / f"{prefix}_annual_excess_returns.csv", index=False
    )
    (output / f"{prefix}_diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
