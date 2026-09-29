from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashare_quant.config import MODEL_FEATURES
from ashare_quant.features import cross_sectional_preprocess


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    validation_start: pd.Timestamp
    test_start: pd.Timestamp


def add_forward_target(frame: pd.DataFrame, *, horizon_days: int) -> pd.DataFrame:
    data = frame.sort_values(["code", "date"], kind="stable").copy()
    by_code = data.groupby("code", sort=False)
    data["entry_open"] = by_code["open"].shift(-1)
    data["exit_close"] = by_code["close"].shift(-horizon_days)
    data["label_end_date"] = by_code["date"].shift(-horizon_days)
    data["forward_return"] = data["exit_close"] / data["entry_open"] - 1.0
    data["benchmark_forward_return"] = data.groupby("date")["forward_return"].transform("median")
    data["target_excess_return"] = data["forward_return"] - data["benchmark_forward_return"]
    data["target_rank"] = data.groupby("date")["target_excess_return"].rank(pct=True, method="average")
    data["relevance"] = np.floor(data["target_rank"] * 5).clip(0, 4)
    return data


def month_end_dates(dates: pd.Series) -> pd.DatetimeIndex:
    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    return pd.DatetimeIndex(unique.groupby(unique.dt.to_period("M")).max().to_numpy())


def build_research_dataset(
    factor_frame: pd.DataFrame,
    *,
    horizon_days: int,
    winsor_lower: float,
    winsor_upper: float,
) -> pd.DataFrame:
    labeled = add_forward_target(factor_frame, horizon_days=horizon_days)
    rebalance_dates = month_end_dates(labeled["date"])
    dataset = labeled.loc[labeled["date"].isin(rebalance_dates) & labeled["tradable"]].copy()
    dataset = cross_sectional_preprocess(
        dataset,
        lower_quantile=winsor_lower,
        upper_quantile=winsor_upper,
    )
    dataset = dataset.dropna(subset=MODEL_FEATURES + ["target_rank", "label_end_date"])
    return dataset.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def split_dataset(
    dataset: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> DatasetSplit:
    dates = pd.DatetimeIndex(sorted(dataset["date"].unique()))
    if len(dates) < 12:
        raise ValueError("At least 12 rebalance dates are required for train/validation/test splitting")

    validation_index = max(1, int(len(dates) * train_fraction))
    test_index = max(validation_index + 1, int(len(dates) * (train_fraction + validation_fraction)))
    if test_index >= len(dates):
        test_index = len(dates) - 1

    validation_start = dates[validation_index]
    test_start = dates[test_index]

    train = dataset.loc[
        (dataset["date"] < validation_start) & (dataset["label_end_date"] < validation_start)
    ].copy()
    validation = dataset.loc[
        (dataset["date"] >= validation_start)
        & (dataset["date"] < test_start)
        & (dataset["label_end_date"] < test_start)
    ].copy()
    test = dataset.loc[dataset["date"] >= test_start].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Purged split produced an empty train, validation, or test partition")

    return DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        validation_start=pd.Timestamp(validation_start),
        test_start=pd.Timestamp(test_start),
    )

