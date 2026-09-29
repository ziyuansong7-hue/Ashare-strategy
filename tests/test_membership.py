import pandas as pd

from ashare_quant.data.securities import compress_membership_snapshots


def test_monthly_membership_snapshots_are_compressed_without_hiding_reentries():
    snapshots = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-29",
                    "2024-03-29",
                    "2024-01-31",
                    "2024-03-29",
                ]
            ),
            "code": ["000001", "000001", "000001", "600000", "600000"],
            "security_name": ["A", "A", "A", "B", "B"],
            "size_bucket": ["MID", "MID", "LARGE", "LARGE", "LARGE"],
        }
    )

    intervals = compress_membership_snapshots(snapshots)

    stock_a = intervals.loc[intervals["code"] == "000001"]
    assert stock_a["size_bucket"].tolist() == ["MID", "LARGE"]
    assert stock_a["effective_from"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-03-29"),
    ]
    assert stock_a["effective_to"].tolist()[0] == pd.Timestamp("2024-03-28")

    stock_b = intervals.loc[intervals["code"] == "600000"]
    assert len(stock_b) == 2
    assert stock_b.iloc[0]["effective_to"] == pd.Timestamp("2024-02-28")
    assert stock_b.iloc[1]["effective_from"] == pd.Timestamp("2024-03-29")
