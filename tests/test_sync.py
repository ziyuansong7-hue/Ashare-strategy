from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.audit import audit_market_data
from ashare_quant.data.sync import MarketDataSynchronizer, SyncConfig


class FakeProvider:
    name = "fake"
    version = "1.0"

    def __init__(self) -> None:
        self.calls: dict[tuple[str, str], int] = {}

    def fetch_security_master(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": ["000001", "600000", "510300"],
                "security_name": ["测试深股", "测试沪股", "应被过滤的ETF"],
            }
        )

    def fetch_index_constituents(self, index_code: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "snapshot_date": [pd.Timestamp("2024-01-10")] * 2,
                "index_code": [index_code] * 2,
                "index_name": ["测试指数"] * 2,
                "code": ["000001", "600000"],
                "security_name": ["测试深股", "测试沪股"],
            }
        )

    def _history(self, start_date: str, end_date: str, multiplier: float = 1.0) -> pd.DataFrame:
        dates = pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        values = pd.Series(range(len(dates)), dtype=float) + 10.0
        return pd.DataFrame(
            {
                "date": dates,
                "open": values * multiplier,
                "high": (values + 1.0) * multiplier,
                "low": (values - 1.0) * multiplier,
                "close": (values + 0.5) * multiplier,
                "volume": 100_000,
                "amount": 1_000_000.0,
                "turnover": 1.0,
            }
        )

    def fetch_stock_daily(
        self, code: str, start_date: str, end_date: str, *, adjust: str
    ) -> pd.DataFrame:
        key = (code, adjust)
        self.calls[key] = self.calls.get(key, 0) + 1
        if key == ("000001", "") and self.calls[key] == 1:
            raise ConnectionError("temporary provider failure")
        return self._history(start_date, end_date, multiplier=0.9 if adjust == "qfq" else 1.0)

    def fetch_index_daily(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._history(start_date, end_date)


class EnrichedFakeProvider(FakeProvider):
    def fetch_industry_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": ["000001", "600000"],
                "effective_from": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "effective_to": [pd.NaT, pd.NaT],
                "industry_code": ["BANK", "BANK"],
                "industry_name": ["Bank", "Bank"],
                "industry_standard": ["TEST", "TEST"],
                "is_point_in_time": [True, True],
            }
        )

    def fetch_corporate_actions(self, code: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": [code],
                "ex_date": [pd.Timestamp("2024-01-08")],
                "cash_dividend_per_share": [0.1],
                "bonus_share_ratio": [0.0],
                "rights_share_ratio": [0.0],
                "rights_price": [float("nan")],
                "source": ["test"],
            }
        )


class CurrentIndustryFallbackProvider(FakeProvider):
    def fetch_industry_history(self) -> pd.DataFrame:
        raise ConnectionError("historical industry endpoint unavailable")

    def fetch_current_industry_snapshot(self, codes: set[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": sorted(codes),
                "industry_code": ["BANK"] * len(codes),
                "industry_name": ["Bank"] * len(codes),
            }
        )

class FakeStatusProvider:
    def fetch_stock_status(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        dates = pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        return pd.DataFrame(
            {
                "date": dates,
                "preclose": 10.0,
                "is_trading": True,
                "is_st": False,
                "status_is_point_in_time": True,
                "status_source": "test",
            }
        )

    def close(self) -> None:
        pass


class HistoricalMembershipStatusProvider(FakeStatusProvider):
    def fetch_index_membership_snapshots(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "snapshot_date": pd.to_datetime(["2024-01-10", "2024-01-10"]),
                "provider_snapshot_date": pd.to_datetime(["2024-01-08", "2024-01-08"]),
                "code": ["000001", "600000"],
                "security_name": ["测试深股", "测试沪股"],
                "universe_segment": ["csi300", "csi500"],
                "size_bucket": ["LARGE", "MID"],
            }
        )


class SegmentedFakeProvider(FakeProvider):
    def fetch_security_master(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "code": ["000001", "600000", "000002", "600001"],
                "security_name": ["大盘深股", "大盘沪股", "中盘深股", "中盘沪股"],
            }
        )

    def fetch_index_constituents(self, index_code: str) -> pd.DataFrame:
        codes = ["000001", "600000"] if index_code == "000300" else ["000002", "600001"]
        return pd.DataFrame(
            {
                "snapshot_date": [pd.Timestamp("2024-01-10")] * 2,
                "index_code": [index_code] * 2,
                "index_name": ["测试指数"] * 2,
                "code": codes,
                "security_name": [f"测试{code}" for code in codes],
            }
        )


def _config(tmp_path: Path, *, allow_backfill: bool) -> SyncConfig:
    return SyncConfig(
        data_dir=tmp_path / "market",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-15"),
        universe="csi800",
        workers=2,
        retries=2,
        retry_backoff_seconds=0,
        allow_current_universe_backfill=allow_backfill,
    )


def test_default_sync_universe_is_csi800(tmp_path: Path):
    config = SyncConfig(
        data_dir=tmp_path / "market",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-01-15"),
    )

    assert config.universe == "csi800"


def test_csi800_is_built_from_labeled_csi300_and_csi500_segments(tmp_path: Path):
    synchronizer = MarketDataSynchronizer(
        SegmentedFakeProvider(), _config(tmp_path, allow_backfill=True)
    )

    master, membership, point_in_time, warnings = synchronizer._resolve_universe()

    assert point_in_time is False
    assert not warnings or all("survivorship bias" in warning for warning in warnings)
    assert set(membership.loc[membership["size_bucket"] == "LARGE", "code"]) == {
        "000001",
        "600000",
    }
    assert set(membership.loc[membership["size_bucket"] == "MID", "code"]) == {
        "000002",
        "600001",
    }
    assert set(master["size_bucket"]) == {"LARGE", "MID"}


def test_current_snapshot_backfill_requires_explicit_acknowledgement(tmp_path: Path):
    with pytest.raises(ValueError, match="survivorship bias"):
        MarketDataSynchronizer(FakeProvider(), _config(tmp_path, allow_backfill=False)).run()


def test_sync_writes_raw_adjusted_bars_benchmarks_and_report(tmp_path: Path):
    provider = FakeProvider()
    config = _config(tmp_path, allow_backfill=True)
    report = MarketDataSynchronizer(provider, config).run()

    assert report["summary"] == {
        "requested_symbols": 2,
        "successful_symbols": 2,
        "skipped_complete_symbols": 0,
        "failed_symbols": 0,
        "size_bucket_counts": {"LARGE": 2},
    }
    assert report["quality_gates"]["point_in_time_membership"] is False
    assert report["quality_gates"]["research_ready"] is False
    assert report["evidence_grade"] == "ENGINEERING_ONLY"
    assert report["quality_gates"]["universe_completeness"] is False
    assert provider.calls[("000001", "")] == 2

    market_dir = tmp_path / "market"
    assert (market_dir / "security_master.parquet").exists()
    assert (market_dir / "universe_membership.parquet").exists()
    assert (market_dir / "benchmarks.parquet").exists()
    assert (market_dir / "metadata" / "download_report.json").exists()

    master = pd.read_parquet(market_dir / "security_master.parquet")
    assert set(master["code"]) == {"000001", "600000"}
    assert "510300" not in set(master["code"])

    bars = pd.read_parquet(market_dir / "bars" / "000001.parquet")
    assert {"open", "raw_open", "close", "raw_close", "instrument_type"}.issubset(bars.columns)
    assert (bars["open"] < bars["raw_open"]).all()

    second_report = MarketDataSynchronizer(provider, config).run()
    assert second_report["summary"]["skipped_complete_symbols"] == 2

    audit = audit_market_data(market_dir, min_rows_per_symbol=1)
    assert audit["passed"] is True
    assert audit["summary"]["benchmarks"] == [
        "csi300",
        "csi500",
        "csi800",
        "sse_composite",
        "szse_component",
    ]
    assert report["quality_gates"]["required_benchmarks_downloaded"] is True


def test_sync_persists_historical_status_industry_and_corporate_actions(tmp_path: Path):
    report = MarketDataSynchronizer(
        EnrichedFakeProvider(),
        _config(tmp_path, allow_backfill=True),
        status_provider=FakeStatusProvider(),
    ).run()

    assert report["quality_gates"]["historical_security_status"] is True
    assert report["quality_gates"]["point_in_time_industry"] is True
    assert report["quality_gates"]["corporate_actions"] is True
    market_dir = tmp_path / "market"
    assert (market_dir / "industry_history.parquet").exists()
    assert (market_dir / "corporate_actions.parquet").exists()
    bars = pd.read_parquet(market_dir / "bars" / "000001.parquet")
    assert bars["status_is_point_in_time"].all()
    assert {"market_cap", "limit_up", "limit_down", "adjustment_factor"}.issubset(bars)


def test_sync_current_industry_fallback_cannot_pass_point_in_time_gate(tmp_path: Path):
    report = MarketDataSynchronizer(
        CurrentIndustryFallbackProvider(), _config(tmp_path, allow_backfill=True)
    ).run()

    assert report["quality_gates"]["point_in_time_industry"] is False
    assert any("current industry snapshot" in warning.lower() for warning in report["warnings"])
    history = pd.read_parquet(tmp_path / "market" / "industry_history.parquet")
    assert not history["is_point_in_time"].any()
    assert set(history["industry_standard"]) == {"FREE_CURRENT_SNAPSHOT_BACKFILL"}


def test_free_historical_csi800_uses_point_in_time_membership_without_industry_backfill(
    tmp_path: Path,
):
    report = MarketDataSynchronizer(
        EnrichedFakeProvider(),
        _config(tmp_path, allow_backfill=False),
        status_provider=HistoricalMembershipStatusProvider(),
    ).run()

    assert report["request"]["membership_source"] == "baostock_month_end_history"
    assert report["quality_gates"]["point_in_time_membership"] is True
    assert report["quality_gates"]["historical_industry_required"] is False
    assert report["quality_gates"]["point_in_time_industry"] is False
    history = pd.read_parquet(tmp_path / "market" / "industry_history.parquet")
    assert history.empty


def test_free_profile_can_keep_historical_membership_while_skipping_daily_status(tmp_path: Path):
    config = _config(tmp_path, allow_backfill=False)
    config = SyncConfig(
        **{
            **config.__dict__,
            "fetch_historical_status": False,
        }
    )
    status_provider = HistoricalMembershipStatusProvider()

    report = MarketDataSynchronizer(
        EnrichedFakeProvider(),
        config,
        status_provider=status_provider,
    ).run()

    assert report["request"]["membership_source"] == "baostock_month_end_history"
    assert report["quality_gates"]["point_in_time_membership"] is True
    assert report["quality_gates"]["historical_security_status"] is False
