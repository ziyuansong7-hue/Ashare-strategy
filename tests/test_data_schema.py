import pandas as pd

from ashare_quant.data.schema import normalize_and_validate_bars


def test_symbol_codes_keep_leading_zero_and_duplicates_are_deduplicated():
    bars = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "code": 1,
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 1000,
                "amount": 10000,
            },
            {
                "date": "2024-01-02",
                "code": "000001",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.6,
                "volume": 1000,
                "amount": 10000,
            },
        ]
    )
    clean, report = normalize_and_validate_bars(bars)
    assert clean["code"].tolist() == ["000001"]
    assert clean["close"].iloc[0] == 10.6
    assert report.duplicate_rows_removed == 1

