import pandas as pd

from ashare_quant.data.enrichment import (
    add_market_cap_estimate,
    add_price_limits,
    apply_industry_history,
    normalize_industry_history,
)


def test_point_in_time_industry_and_market_metadata_are_applied_without_lookahead():
    history = normalize_industry_history(
        pd.DataFrame(
            {
                "股票代码": ["000001", "000001"],
                "纳入时间": ["2020-01-01", "2022-01-01"],
                "行业代码": ["A", "B"],
                "行业名称": ["OLD", "NEW"],
            }
        )
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-06-01", "2023-06-01"]),
            "code": ["000001", "000001"],
            "board": ["MAIN", "MAIN"],
            "is_st": [False, False],
            "status_is_point_in_time": [True, True],
            "preclose": [10.0, 20.0],
            "raw_close": [10.1, 20.2],
            "volume": [1_000_000, 2_000_000],
            "turnover": [0.01, 0.02],
        }
    )
    enriched = add_price_limits(add_market_cap_estimate(apply_industry_history(bars, history)))

    assert enriched["industry"].tolist() == ["OLD", "NEW"]
    assert enriched["industry_is_point_in_time"].all()
    assert enriched["market_cap"].tolist() == [1_010_000_000.0, 2_020_000_000.0]
    assert enriched["limit_up"].tolist() == [11.0, 22.0]
    assert enriched["limit_down"].tolist() == [9.0, 18.0]


def test_chinext_uses_twenty_percent_limit_after_reform():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "board": ["CHINEXT"],
            "is_st": [False],
            "status_is_point_in_time": [True],
            "preclose": [10.0],
        }
    )
    enriched = add_price_limits(frame)
    assert enriched.loc[0, "limit_up"] == 12.0
    assert enriched.loc[0, "limit_down"] == 8.0


def test_missing_turnover_marks_market_cap_unavailable_instead_of_crashing():
    frame = pd.DataFrame({"raw_close": [10.0], "volume": [1_000]})
    enriched = add_market_cap_estimate(frame)
    assert pd.isna(enriched.loc[0, "market_cap"])
    assert enriched.loc[0, "market_cap_source"] == "UNAVAILABLE"


def test_current_industry_fallback_remains_explicitly_non_point_in_time():
    history = normalize_industry_history(
        pd.DataFrame(
            {
                "code": ["000001"],
                "effective_from": ["2020-01-01"],
                "industry_code": ["BANK"],
                "industry_name": ["Bank"],
                "industry_standard": ["CURRENT_SNAPSHOT_BACKFILL"],
                "is_point_in_time": [False],
            }
        )
    )
    bars = pd.DataFrame({"date": [pd.Timestamp("2021-01-04")], "code": ["000001"]})

    enriched = apply_industry_history(bars, history)

    assert history.loc[0, "industry_standard"] == "CURRENT_SNAPSHOT_BACKFILL"
    assert bool(history.loc[0, "is_point_in_time"]) is False
    assert enriched.loc[0, "industry"] == "Bank"
    assert bool(enriched.loc[0, "industry_is_point_in_time"]) is False
