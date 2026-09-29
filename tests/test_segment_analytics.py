import pandas as pd
import pytest

from ashare_quant.analytics import segment_model_report


def test_segment_model_report_separates_large_and_mid_rank_quality():
    rows = []
    for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
        for bucket, direction in [("LARGE", 1.0), ("MID", -1.0)]:
            for rank in range(1, 6):
                rows.append(
                    {
                        "date": date,
                        "size_bucket": bucket,
                        "xgboost_score": float(rank),
                        "target_excess_return": direction * rank / 100.0,
                    }
                )
    report = segment_model_report(pd.DataFrame(rows), ["xgboost_score"])

    large_ic = report.loc[report["size_bucket"] == "LARGE", "mean_rank_ic"].iloc[0]
    mid_ic = report.loc[report["size_bucket"] == "MID", "mean_rank_ic"].iloc[0]
    assert large_ic == pytest.approx(1.0)
    assert mid_ic == pytest.approx(-1.0)
