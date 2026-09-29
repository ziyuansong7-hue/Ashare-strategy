import pandas as pd

from ashare_quant.data.providers.akshare_provider import AKShareProvider


class FallbackOnlyAKShare:
    def stock_zh_a_hist(self, **kwargs):
        raise ConnectionError("primary stock endpoint down")

    def stock_zh_a_hist_tx(self, **kwargs):
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "volume": [100_000],
                "amount": [1_000_000.0],
                "turnover": [0.01],
            }
        )

    def index_zh_a_hist(self, **kwargs):
        raise ConnectionError("primary index endpoint down")

    def stock_zh_index_daily_tx(self, **kwargs):
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [3000.0],
                "high": [3050.0],
                "low": [2990.0],
                "close": [3040.0],
                "amount": [123_456.0],
            }
        )

    def stock_board_industry_name_em(self):
        return pd.DataFrame(
            {"板块名称": ["银行", "软件"], "板块代码": ["BK0475", "BK0737"]}
        )

    def stock_board_industry_cons_em(self, symbol):
        members = {
            "银行": pd.DataFrame({"代码": ["000001", "600000"]}),
            "软件": pd.DataFrame({"代码": ["000002"]}),
        }
        return members[symbol]


class SinaFallbackAKShare(FallbackOnlyAKShare):
    def stock_zh_a_daily(self, **kwargs):
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [11.0],
                "high": [11.5],
                "low": [10.8],
                "close": [11.2],
                "volume": [120_000],
                "amount": [1_200_000.0],
                "turnover": [0.02],
            }
        )

class CNInfoIndustryFallbackAKShare(FallbackOnlyAKShare):
    def stock_board_industry_name_em(self):
        raise ConnectionError("Eastmoney industry endpoint down")

    def stock_industry_change_cninfo(self, symbol, start_date, end_date):
        return pd.DataFrame(
            {
                "证券代码": [symbol, symbol],
                "变更日期": ["2020-01-01", "2024-01-01"],
                "行业编码": ["OLD", "NEW"],
                "行业大类": ["旧行业", "新行业"],
            }
        )


def _provider() -> AKShareProvider:
    provider = AKShareProvider.__new__(AKShareProvider)
    provider._ak = FallbackOnlyAKShare()
    provider.timeout_seconds = 1.0
    return provider


def test_stock_history_falls_back_to_tencent_endpoint():
    frame = _provider().fetch_stock_daily("000001", "20240101", "20240131", adjust="qfq")

    assert len(frame) == 1
    assert frame.attrs["source_endpoint"] == "tencent_stock_zh_a_hist_tx"


def test_stock_history_prefers_sina_before_paginated_tencent_fallback():
    provider = AKShareProvider.__new__(AKShareProvider)
    provider._ak = SinaFallbackAKShare()
    provider.timeout_seconds = 1.0

    frame = provider.fetch_stock_daily("000001", "20240101", "20240131", adjust="qfq")

    assert frame.loc[0, "close"] == 11.2
    assert frame.attrs["source_endpoint"] == "sina_stock_zh_a_daily"


def test_index_history_falls_back_and_normalizes_volume():
    frame = _provider().fetch_index_daily("000001", "20240101", "20240131")

    assert frame.loc[0, "volume"] == 123_456.0
    assert frame.loc[0, "amount"] == 0.0
    assert frame.attrs["source_endpoint"] == "tencent_stock_zh_index_daily_tx"


def test_current_industry_snapshot_filters_requested_codes():
    frame = _provider().fetch_current_industry_snapshot({"000001", "000002"})

    assert frame.to_dict("records") == [
        {"code": "000001", "industry_code": "BK0475", "industry_name": "银行"},
        {"code": "000002", "industry_code": "BK0737", "industry_name": "软件"},
    ]


def test_current_industry_snapshot_falls_back_to_latest_cninfo_record():
    provider = AKShareProvider.__new__(AKShareProvider)
    provider._ak = CNInfoIndustryFallbackAKShare()
    provider.timeout_seconds = 1.0

    frame = provider.fetch_current_industry_snapshot({"000001"})

    assert frame.to_dict("records") == [
        {"code": "000001", "industry_code": "NEW", "industry_name": "新行业"}
    ]
    assert frame.attrs["source_endpoint"] == "cninfo_stock_industry_change_latest"
