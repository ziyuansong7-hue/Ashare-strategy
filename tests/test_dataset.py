import numpy as np
import pandas as pd

from ashare_quant.research.dataset import add_forward_target, split_dataset


def test_forward_target_starts_at_next_open():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "code": ["000001"] * 6,
            "open": [10, 11, 12, 13, 14, 15],
            "close": [10, 12, 13, 14, 15, 16],
        }
    )
    labeled = add_forward_target(frame, horizon_days=2)
    assert np.isclose(labeled.loc[0, "forward_return"], 13 / 11 - 1)
    assert labeled.loc[0, "label_end_date"] == pd.Timestamp("2024-01-03")


def test_purged_split_has_no_label_overlap():
    dates = pd.date_range("2020-01-31", periods=20, freq="ME")
    rows = []
    for date_index, date in enumerate(dates[:-1]):
        for code in ["000001", "000002"]:
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "label_end_date": dates[date_index + 1] - pd.Timedelta(days=1),
                    "target_rank": 0.5,
                }
            )
    dataset = pd.DataFrame(rows)
    split = split_dataset(dataset, train_fraction=0.6, validation_fraction=0.2)
    assert split.train["label_end_date"].max() < split.validation_start
    assert split.validation["label_end_date"].max() < split.test_start
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()

