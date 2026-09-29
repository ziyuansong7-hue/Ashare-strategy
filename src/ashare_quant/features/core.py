from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.config import RAW_FEATURES


def _trend_t_stat(values: np.ndarray) -> float:
    if len(values) < 3 or np.any(values <= 0) or not np.isfinite(values).all():
        return np.nan
    y = np.log(values)
    x = np.arange(len(y), dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    ssx = float(np.dot(x_centered, x_centered))
    if ssx == 0:
        return np.nan
    slope = float(np.dot(x_centered, y_centered) / ssx)
    residual = y_centered - slope * x_centered
    degrees_of_freedom = len(y) - 2
    residual_variance = float(np.dot(residual, residual) / degrees_of_freedom)
    standard_error = np.sqrt(residual_variance / ssx)
    if standard_error == 0:
        return 0.0
    return slope / standard_error


def _downside_deviation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    downside = np.minimum(finite, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def compute_raw_factors(
    bars: pd.DataFrame,
    *,
    min_history_days: int = 120,
    min_liquidity_percentile: float = 0.20,
) -> pd.DataFrame:
    """Compute causal factors using data available on or before each row's date."""
    if not 0 <= min_liquidity_percentile < 1:
        raise ValueError("min_liquidity_percentile must be in [0, 1)")
    frame = bars.sort_values(["code", "date"], kind="stable").copy()
    by_code = frame.groupby("code", sort=False, group_keys=False)

    frame["return_1"] = by_code["close"].pct_change(fill_method=None)
    frame["market_return"] = frame.groupby("date")["return_1"].transform("median")
    frame["residual_return"] = frame["return_1"] - frame["market_return"]

    frame["momentum_20_5"] = by_code["close"].shift(5) / by_code["close"].shift(20) - 1.0
    frame["momentum_60_5"] = by_code["close"].shift(5) / by_code["close"].shift(60) - 1.0
    frame["trend_quality_60"] = by_code["close"].transform(
        lambda values: values.rolling(60, min_periods=60).apply(_trend_t_stat, raw=True)
    )
    frame["reversal_5"] = frame.groupby("code", sort=False)["residual_return"].transform(
        lambda values: -values.rolling(5, min_periods=5).sum()
    )

    rolling_high = by_code["close"].transform(
        lambda values: values.rolling(120, min_periods=120).max()
    )
    frame["high_position_120"] = frame["close"] / rolling_high - 1.0

    frame["idio_vol_60"] = frame.groupby("code", sort=False)["residual_return"].transform(
        lambda values: values.rolling(60, min_periods=40).std()
    )
    frame["downside_vol_60"] = frame.groupby("code", sort=False)["residual_return"].transform(
        lambda values: values.rolling(60, min_periods=40).apply(_downside_deviation, raw=True)
    )

    activity_source = "turnover" if "turnover" in frame else "volume"
    frame["_log_activity"] = np.log1p(frame[activity_source].clip(lower=0))
    activity_group = frame.groupby("code", sort=False)["_log_activity"]
    activity_mean = activity_group.transform(lambda values: values.rolling(20, min_periods=15).mean())
    activity_std = activity_group.transform(lambda values: values.rolling(20, min_periods=15).std())
    frame["volume_z20"] = (frame["_log_activity"] - activity_mean) / activity_std.replace(0, np.nan)
    signed_activity = np.sign(frame["residual_return"]) * frame["volume_z20"]
    frame["price_volume_confirmation_5"] = signed_activity.groupby(frame["code"]).transform(
        lambda values: values.rolling(5, min_periods=3).mean()
    )

    daily_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["_clv"] = (2.0 * frame["close"] - frame["high"] - frame["low"]) / daily_range
    frame["clv_5"] = frame.groupby("code", sort=False)["_clv"].transform(
        lambda values: values.rolling(5, min_periods=3).mean()
    )

    frame["median_amount_20"] = by_code["amount"].transform(
        lambda values: values.rolling(20, min_periods=15).median()
    )
    frame["liquidity_percentile"] = frame.groupby("date")["median_amount_20"].rank(
        pct=True, method="average"
    )
    frame["_amihud_daily"] = (
        frame["return_1"].abs() / frame["amount"].replace(0, np.nan) * 1_000_000.0
    )
    frame["amihud_20"] = frame.groupby("code", sort=False)["_amihud_daily"].transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )

    frame["listing_days"] = by_code.cumcount() + 1
    universe_member = (
        frame["is_universe_member"].fillna(False).astype(bool)
        if "is_universe_member" in frame
        else pd.Series(True, index=frame.index)
    )
    frame["tradable"] = (
        frame["is_trading"]
        & ~frame["is_st"]
        & universe_member
        & (frame["listing_days"] >= min_history_days)
        & (frame["median_amount_20"] > 0)
        & (frame["liquidity_percentile"] > min_liquidity_percentile)
    )
    frame[RAW_FEATURES] = frame[RAW_FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame.drop(columns=["_log_activity", "_clv", "_amihud_daily"])


def _robust_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad == 0:
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=values.index)
        return (values - values.mean()) / std
    return (values - median) / (1.4826 * mad)


def _neutralize_market_cap(group: pd.DataFrame, column: str) -> pd.Series:
    y = group[column]
    if "market_cap" not in group or group["market_cap"].notna().sum() < 10:
        return y
    valid = y.notna() & group["market_cap"].gt(0)
    if valid.sum() < 10:
        return y
    x = np.log(group.loc[valid, "market_cap"].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y.loc[valid].to_numpy(dtype=float), rcond=None)
    result = y.copy()
    result.loc[valid] = y.loc[valid] - design @ coefficients
    return result


def cross_sectional_preprocess(
    observations: pd.DataFrame,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.DataFrame:
    """Winsorize, industry-demean, size-neutralize, and robustly standardize factors."""
    frame = observations.copy()
    for feature in RAW_FEATURES:
        lower = frame.groupby("date")[feature].transform(lambda values: values.quantile(lower_quantile))
        upper = frame.groupby("date")[feature].transform(lambda values: values.quantile(upper_quantile))
        neutral = frame[feature].clip(lower=lower, upper=upper)

        point_in_time_industry = (
            "industry_is_point_in_time" not in frame
            or bool(frame["industry_is_point_in_time"].fillna(False).all())
        )
        meaningful_industry = "industry" in frame and not frame["industry"].eq("UNKNOWN").all()
        if meaningful_industry and point_in_time_industry:
            neutral = neutral - neutral.groupby([frame["date"], frame["industry"]]).transform("median")

        temporary = (
            frame[["date", "market_cap"]].copy()
            if "market_cap" in frame
            else frame[["date"]].copy()
        )
        temporary["value"] = neutral
        adjusted = pd.Series(np.nan, index=frame.index, dtype=float)
        for indexes in temporary.groupby("date", sort=False).groups.values():
            group = temporary.loc[indexes]
            adjusted.loc[indexes] = _neutralize_market_cap(group, "value")
        frame[f"{feature}_z"] = adjusted.groupby(frame["date"]).transform(_robust_zscore).clip(-8, 8)

    return frame
