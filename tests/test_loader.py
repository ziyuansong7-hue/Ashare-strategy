from pathlib import Path

import pandas as pd

from ashare_quant.data.loader import load_bars


def test_market_root_loads_only_bars_subdirectory(tmp_path: Path):
    root = tmp_path / "market"
    bars_dir = root / "bars"
    bars_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "code": ["000001"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000],
            "amount": [10_200.0],
        }
    ).to_parquet(bars_dir / "000001.parquet", index=False)
    pd.DataFrame({"date": ["2024-01-02"], "benchmark_key": ["sse_composite"]}).to_parquet(
        root / "benchmarks.parquet", index=False
    )

    bars, report = load_bars(root)

    assert bars["code"].tolist() == ["000001"]
    assert report.rows == 1


def test_market_root_enforces_point_in_time_membership(tmp_path: Path):
    root = tmp_path / "market"
    bars_dir = root / "bars"
    bars_dir.mkdir(parents=True)
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    pd.DataFrame(
        {
            "date": dates,
            "code": ["000001"] * 4,
            "open": [10.0] * 4,
            "high": [10.5] * 4,
            "low": [9.5] * 4,
            "close": [10.1] * 4,
            "volume": [1000] * 4,
            "amount": [10_100.0] * 4,
        }
    ).to_parquet(bars_dir / "000001.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["000001"],
            "effective_from": ["2024-01-02"],
            "effective_to": ["2024-01-03"],
            "membership_mode": ["point_in_time"],
        }
    ).to_parquet(root / "universe_membership.parquet", index=False)

    bars, report = load_bars(root)

    assert bars["date"].tolist() == dates.tolist()
    assert bars["is_universe_member"].tolist() == [False, True, True, False]
    assert report.rows == 4


def test_current_snapshot_membership_filters_stale_bar_files(tmp_path: Path):
    root = tmp_path / "market"
    bars_dir = root / "bars"
    bars_dir.mkdir(parents=True)
    for code in ["000001", "600000"]:
        pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "code": [code],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.1],
                "volume": [1000],
                "amount": [10_100.0],
            }
        ).to_parquet(bars_dir / f"{code}.parquet", index=False)
    pd.DataFrame(
        {
            "code": ["000001"],
            "membership_mode": ["current_snapshot"],
            "size_bucket": ["LARGE"],
        }
    ).to_parquet(root / "universe_membership.parquet", index=False)

    bars, report = load_bars(root)

    assert bars["code"].tolist() == ["000001"]
    assert bars["size_bucket"].tolist() == ["LARGE"]
    assert report.rows == 1
