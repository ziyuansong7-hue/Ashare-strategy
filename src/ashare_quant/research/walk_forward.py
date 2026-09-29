from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    dataset: pd.DataFrame,
    *,
    train_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
    window: str = "expanding",
) -> list[WalkForwardFold]:
    """Build purged monthly walk-forward folds with non-overlapping test windows."""
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be expanding or rolling")
    for name, value in {
        "train_months": train_months,
        "validation_months": validation_months,
        "test_months": test_months,
        "step_months": step_months,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if step_months != test_months:
        raise ValueError(
            "step_months must equal test_months so test windows are contiguous and non-overlapping"
        )
    required = {"date", "label_end_date"}
    missing = required.difference(dataset.columns)
    if missing:
        raise ValueError(f"Walk-forward dataset is missing columns: {sorted(missing)}")

    data = dataset.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["label_end_date"] = pd.to_datetime(data["label_end_date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    minimum_dates = train_months + validation_months + test_months
    if len(dates) < minimum_dates:
        raise ValueError(
            f"Walk-forward requires at least {minimum_dates} rebalance dates, got {len(dates)}"
        )

    folds: list[WalkForwardFold] = []
    first_test_index = train_months + validation_months
    for fold_id, test_start_index in enumerate(
        range(first_test_index, len(dates), step_months), start=1
    ):
        test_end_index = min(test_start_index + test_months, len(dates))
        validation_start_index = test_start_index - validation_months
        train_start_index = 0 if window == "expanding" else validation_start_index - train_months
        train_dates = dates[train_start_index:validation_start_index]
        validation_dates = dates[validation_start_index:test_start_index]
        test_dates = dates[test_start_index:test_end_index]
        if len(test_dates) == 0:
            continue

        validation_start = pd.Timestamp(validation_dates[0])
        test_start = pd.Timestamp(test_dates[0])
        train = data.loc[
            data["date"].isin(train_dates) & (data["label_end_date"] < validation_start)
        ].copy()
        validation = data.loc[
            data["date"].isin(validation_dates) & (data["label_end_date"] < test_start)
        ].copy()
        test = data.loc[data["date"].isin(test_dates)].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError(f"Purging produced an empty partition in walk-forward fold {fold_id}")

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train=train,
                validation=validation,
                test=test,
                train_start=pd.Timestamp(train_dates[0]),
                train_end=pd.Timestamp(train_dates[-1]),
                validation_start=validation_start,
                validation_end=pd.Timestamp(validation_dates[-1]),
                test_start=test_start,
                test_end=pd.Timestamp(test_dates[-1]),
            )
        )
    return folds
