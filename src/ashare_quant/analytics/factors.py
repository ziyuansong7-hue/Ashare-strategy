from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.config import MODEL_FEATURES


def factor_report(dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for feature in MODEL_FEATURES:
        daily_ic = dataset.groupby("date").apply(
            lambda group, feature=feature: group[feature].corr(
                group["target_excess_return"], method="spearman"
            ),
            include_groups=False,
        ).dropna()
        mean_ic = float(daily_ic.mean()) if not daily_ic.empty else np.nan
        std_ic = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else np.nan
        rows.append(
            {
                "feature": feature,
                "mean_rank_ic": mean_ic,
                "rank_ic_std": std_ic,
                "rank_ic_ir_annualized": mean_ic / std_ic * np.sqrt(12)
                if std_ic and np.isfinite(std_ic)
                else np.nan,
                "positive_ic_rate": float((daily_ic > 0).mean()) if not daily_ic.empty else np.nan,
                "observations": int(dataset[feature].notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rank_ic", ascending=False)


def segment_model_report(dataset: pd.DataFrame, score_columns: list[str]) -> pd.DataFrame:
    """Compare model ranking quality inside LARGE and MID sub-universes."""
    if "size_bucket" not in dataset:
        return pd.DataFrame()
    rows: list[dict[str, float | int | str]] = []
    for bucket, segment in dataset.groupby("size_bucket", sort=True):
        bucket_name = str(bucket).upper()
        if bucket_name not in {"LARGE", "MID"}:
            continue
        for score_column in score_columns:
            if score_column not in segment:
                continue
            daily_ic: list[float] = []
            daily_top_returns: list[float] = []
            for _, group in segment.groupby("date", sort=True):
                value = group[score_column].corr(
                    group["target_excess_return"], method="spearman"
                )
                if pd.notna(value):
                    daily_ic.append(float(value))
                cutoff = group[score_column].quantile(0.8)
                top = group.loc[group[score_column] >= cutoff, "target_excess_return"]
                if not top.empty:
                    daily_top_returns.append(float(top.mean()))
            ic = pd.Series(daily_ic, dtype=float)
            top_returns = pd.Series(daily_top_returns, dtype=float)
            standard_deviation = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
            rows.append(
                {
                    "size_bucket": bucket_name,
                    "model": score_column.removesuffix("_score"),
                    "mean_rank_ic": float(ic.mean()) if not ic.empty else np.nan,
                    "rank_ic_ir_annualized": (
                        float(ic.mean() / standard_deviation * np.sqrt(12))
                        if np.isfinite(standard_deviation) and standard_deviation > 0
                        else np.nan
                    ),
                    "positive_ic_rate": float((ic > 0).mean()) if not ic.empty else np.nan,
                    "mean_top_quintile_excess_return": (
                        float(top_returns.mean()) if not top_returns.empty else np.nan
                    ),
                    "top_quintile_positive_rate": (
                        float((top_returns > 0).mean()) if not top_returns.empty else np.nan
                    ),
                    "rebalance_dates": int(segment["date"].nunique()),
                    "observations": len(segment),
                }
            )
    return pd.DataFrame(rows)
