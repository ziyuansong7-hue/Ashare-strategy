import pandas as pd
import pytest

from ashare_quant.research import build_walk_forward_splits


def _monthly_dataset(months: int = 72, symbols: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2018-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        [
            {
                "date": date,
                "label_end_date": date + pd.Timedelta(days=20),
                "code": f"{symbol:06d}",
            }
            for date in dates
            for symbol in range(symbols)
        ]
    )


def test_expanding_walk_forward_is_purged_contiguous_and_non_overlapping():
    folds = build_walk_forward_splits(
        _monthly_dataset(),
        train_months=36,
        validation_months=12,
        test_months=12,
        step_months=12,
        window="expanding",
    )

    assert len(folds) == 2
    assert folds[0].test_end < folds[1].test_start
    assert len(folds[1].train) > len(folds[0].train)
    for fold in folds:
        assert fold.train["label_end_date"].max() < fold.validation_start
        assert fold.validation["label_end_date"].max() < fold.test_start
    stitched = pd.concat([fold.test.assign(fold_id=fold.fold_id) for fold in folds])
    assert not stitched.duplicated(["date", "code"]).any()
    assert stitched["date"].nunique() == 24


def test_rolling_walk_forward_keeps_fixed_training_window():
    folds = build_walk_forward_splits(
        _monthly_dataset(),
        train_months=36,
        validation_months=12,
        test_months=12,
        step_months=12,
        window="rolling",
    )
    assert folds[0].train["date"].nunique() == folds[1].train["date"].nunique()
    assert folds[1].train_start > folds[0].train_start


def test_walk_forward_rejects_insufficient_history_and_overlapping_tests():
    with pytest.raises(ValueError, match="at least 60 rebalance dates"):
        build_walk_forward_splits(
            _monthly_dataset(months=59),
            train_months=36,
            validation_months=12,
            test_months=12,
            step_months=12,
        )
    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        build_walk_forward_splits(
            _monthly_dataset(),
            train_months=36,
            validation_months=12,
            test_months=12,
            step_months=6,
        )
