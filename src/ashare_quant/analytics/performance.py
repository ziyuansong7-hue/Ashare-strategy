from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _daily_returns(values: pd.Series, starting_value: float) -> pd.Series:
    numeric = values.astype(float).reset_index(drop=True)
    return pd.concat(
        [pd.Series([numeric.iloc[0] / starting_value - 1.0]), numeric.pct_change().iloc[1:]],
        ignore_index=True,
    )


def _maximum_drawdown(values: pd.Series, starting_value: float) -> float:
    with_start = pd.concat(
        [pd.Series([starting_value]), values.astype(float).reset_index(drop=True)],
        ignore_index=True,
    )
    return float((with_start / with_start.cummax() - 1.0).min())


def attach_benchmark_nav(
    nav: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    *,
    initial_nav: float,
) -> pd.DataFrame:
    """Align external index closes to strategy dates without using future observations."""
    result = nav.copy()
    if benchmarks is None or benchmarks.empty:
        return result
    required = {"date", "benchmark_key", "close"}
    missing = required.difference(benchmarks.columns)
    if missing:
        raise ValueError(f"Benchmark data is missing columns: {sorted(missing)}")

    strategy_dates = pd.DatetimeIndex(pd.to_datetime(result["date"]).dt.normalize())
    for key, group in benchmarks.groupby("benchmark_key", sort=True):
        safe_key = re.sub(r"[^a-z0-9_]+", "_", str(key).lower()).strip("_")
        if not safe_key:
            raise ValueError(f"Invalid benchmark key: {key!r}")
        closes = group.copy()
        closes["date"] = pd.to_datetime(closes["date"]).dt.normalize()
        closes["close"] = pd.to_numeric(closes["close"], errors="coerce")
        closes = closes.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
        close_series = closes.set_index("date")["close"].sort_index()
        aligned = close_series.reindex(close_series.index.union(strategy_dates)).ffill().reindex(strategy_dates)
        first_valid = aligned.first_valid_index()
        if first_valid is None:
            continue
        prior = close_series.loc[close_series.index < strategy_dates.min()]
        base = float(prior.iloc[-1]) if not prior.empty else float(aligned.loc[first_valid])
        result[f"{safe_key}_nav"] = (aligned / base * initial_nav).to_numpy()
    return result


def performance_metrics(
    nav: pd.DataFrame,
    *,
    initial_nav: float | None = None,
) -> dict[str, object]:
    if nav.empty:
        raise ValueError("NAV series is empty")
    values = nav["nav"].astype(float).reset_index(drop=True)
    starting_value = float(initial_nav if initial_nav is not None else values.iloc[0])
    returns = _daily_returns(values, starting_value)
    periods = max(len(values), 1)
    total_return = values.iloc[-1] / starting_value - 1.0
    annualized_return = (values.iloc[-1] / starting_value) ** (252.0 / periods) - 1.0
    annualized_volatility = returns.std(ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    maximum_drawdown = _maximum_drawdown(values, starting_value)
    calmar = annualized_return / abs(maximum_drawdown) if maximum_drawdown < 0 else 0.0
    metrics: dict[str, object] = {
        "start_nav": starting_value,
        "end_nav": float(values.iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": maximum_drawdown,
        "calmar_ratio": float(calmar),
        "positive_day_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "trading_days": len(values),
    }

    benchmark_metrics: dict[str, dict[str, float | int | str]] = {}
    nav_columns = sorted(name for name in nav.columns if name.endswith("_nav") and name != "nav")
    for column in nav_columns:
        benchmark_name = column.removesuffix("_nav")
        benchmark_values = pd.to_numeric(nav[column], errors="coerce").reset_index(drop=True)
        valid_values = benchmark_values.dropna()
        if valid_values.empty:
            continue
        first_position = int(valid_values.index[0])
        strategy_slice = values.iloc[first_position:].reset_index(drop=True)
        benchmark_slice = benchmark_values.iloc[first_position:].ffill().reset_index(drop=True)
        aligned = pd.DataFrame(
            {"strategy": strategy_slice, "benchmark": benchmark_slice}
        ).dropna()
        if aligned.empty:
            continue
        strategy_start = starting_value if first_position == 0 else float(values.iloc[first_position - 1])
        benchmark_start = starting_value
        strategy_returns = _daily_returns(aligned["strategy"], strategy_start)
        benchmark_returns = _daily_returns(aligned["benchmark"], benchmark_start)
        active_returns = strategy_returns - benchmark_returns
        benchmark_total = float(aligned["benchmark"].iloc[-1] / benchmark_start - 1.0)
        benchmark_annualized = float(
            (1.0 + benchmark_total) ** (252.0 / max(len(aligned), 1)) - 1.0
        )
        tracking_error = float(active_returns.std(ddof=1) * np.sqrt(252))
        benchmark_variance = float(benchmark_returns.var(ddof=1))
        beta = (
            float(strategy_returns.cov(benchmark_returns) / benchmark_variance)
            if benchmark_variance > 1e-12
            else 0.0
        )
        alpha = float((strategy_returns.mean() - beta * benchmark_returns.mean()) * 252.0)
        benchmark_metrics[benchmark_name] = {
            "start_date": str(pd.Timestamp(nav["date"].iloc[first_position]).date()),
            "observations": len(aligned),
            "total_return": benchmark_total,
            "annualized_return": benchmark_annualized,
            "max_drawdown": _maximum_drawdown(aligned["benchmark"], benchmark_start),
            "excess_total_return": float(
                aligned["strategy"].iloc[-1] / strategy_start - 1.0 - benchmark_total
            ),
            "tracking_error": tracking_error,
            "information_ratio": float(
                active_returns.mean() / active_returns.std(ddof=1) * np.sqrt(252)
                if len(active_returns) > 1 and active_returns.std(ddof=1) > 0
                else 0.0
            ),
            "alpha_annualized": alpha,
            "beta": beta,
            "correlation": float(strategy_returns.corr(benchmark_returns))
            if len(aligned) > 1
            else 0.0,
        }
    metrics["benchmarks"] = benchmark_metrics
    return metrics


def annual_return_table(nav: pd.DataFrame, *, initial_nav: float) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame()
    dated = nav.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    benchmark_columns = sorted(name for name in dated if name.endswith("_nav") and name != "nav")
    rows: list[dict[str, object]] = []
    for column in ["nav", *benchmark_columns]:
        values = pd.to_numeric(dated[column], errors="coerce")
        returns = values.pct_change(fill_method=None)
        first_valid = values.first_valid_index()
        if first_valid is not None:
            returns.loc[first_valid] = values.loc[first_valid] / initial_nav - 1.0
        grouped = (
            pd.DataFrame({"year": dated["date"].dt.year, "return": returns})
            .dropna()
            .groupby("year")["return"]
            .apply(lambda series: (1.0 + series).prod() - 1.0)
        )
        for year, value in grouped.items():
            rows.append(
                {"year": int(year), "series": column.removesuffix("_nav"), "return": value}
            )
    return pd.DataFrame(rows).pivot(index="year", columns="series", values="return").reset_index()
