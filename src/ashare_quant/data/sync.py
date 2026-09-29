from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from ashare_quant.data.audit import audit_market_data
from ashare_quant.data.enrichment import (
    add_market_cap_estimate,
    add_price_limits,
    empty_corporate_actions,
    normalize_industry_history,
)
from ashare_quant.data.providers.base import MarketDataProvider
from ashare_quant.data.schema import normalize_and_validate_bars
from ashare_quant.data.securities import (
    BENCHMARKS,
    UNIVERSE_INDEX_CODES,
    UNIVERSE_SIZE_BUCKETS,
    canonical_security_master,
    compress_membership_snapshots,
    load_point_in_time_membership,
)
from ashare_quant.data.storage import atomic_write_json, atomic_write_parquet, merge_by_keys

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class SyncConfig:
    data_dir: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    universe: str = "csi800"
    membership_file: str | None = None
    industry_file: str | None = None
    workers: int = 4
    retries: int = 3
    retry_backoff_seconds: float = 1.0
    refresh_mode: str = "missing"
    max_symbols: int | None = None
    allow_current_universe_backfill: bool = False
    fetch_historical_status: bool = True
    default_lot_size: int = 100

    def __post_init__(self) -> None:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be later than end_date")
        if self.refresh_mode not in {"missing", "full"}:
            raise ValueError("refresh_mode must be 'missing' or 'full'")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.retries < 1:
            raise ValueError("retries must be at least 1")

    @property
    def start_yyyymmdd(self) -> str:
        return self.start_date.strftime("%Y%m%d")

    @property
    def end_yyyymmdd(self) -> str:
        return self.end_date.strftime("%Y%m%d")


def _call_with_retry(
    function: Callable[[], T],
    *,
    retries: int,
    backoff_seconds: float,
    operation: str,
) -> tuple[T, int]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return function(), attempt
        except Exception as exc:  # noqa: BLE001 - providers raise heterogeneous exceptions
            last_error = exc
            if attempt == retries:
                break
            LOGGER.warning("%s failed on attempt %s/%s: %s", operation, attempt, retries, exc)
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise RuntimeError(f"{operation} failed after {retries} attempts: {last_error}") from last_error


class MarketDataSynchronizer:
    def __init__(
        self,
        provider: MarketDataProvider,
        config: SyncConfig,
        *,
        status_provider: Any | None = None,
    ) -> None:
        self.provider = provider
        self.status_provider = status_provider
        self.config = config
        self.root = config.data_dir
        self.bars_dir = self.root / "bars"
        self.metadata_dir = self.root / "metadata"
        self.actions_dir = self.root / "corporate_actions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bars_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.actions_dir.mkdir(parents=True, exist_ok=True)
        self._completion_receipts = self._load_completion_receipts()

    def _load_completion_receipts(self) -> dict[str, dict[str, Any]]:
        """Load the last audited request as a resumable per-symbol receipt.

        File dates alone cannot distinguish an IPO/delisting boundary from an interrupted
        download.  The previous manifest can: a successful symbol means the provider was
        queried for that request range and returned all history that existed in it.
        """
        path = self.metadata_dir / "download_report.json"
        if self.config.refresh_mode != "missing" or not path.exists():
            return {}
        try:
            import json

            report = json.loads(path.read_text(encoding="utf-8"))
            request = report.get("request", {})
            request_start = pd.Timestamp(request["start_date"])
            request_end = pd.Timestamp(request["end_date"])
        except (KeyError, TypeError, ValueError, OSError):
            return {}
        receipts: dict[str, dict[str, Any]] = {}
        for row in report.get("stocks", []):
            if row.get("status") not in {"SUCCESS", "SKIPPED_COMPLETE"}:
                continue
            receipts[str(row.get("code"))] = {
                **row,
                "request_start": request_start,
                "request_end": request_end,
            }
        return receipts

    def _retry(self, function: Callable[[], T], operation: str) -> tuple[T, int]:
        return _call_with_retry(
            function,
            retries=self.config.retries,
            backoff_seconds=self.config.retry_backoff_seconds,
            operation=operation,
        )

    def _provider_master_or_fallback(self, fallback: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        try:
            raw, _ = self._retry(self.provider.fetch_security_master, "fetch security master")
            return canonical_security_master(raw, source=self.provider.name), warnings
        except Exception as exc:
            if fallback is None:
                raise
            warnings.append(
                "Security-master endpoint failed; built a selected-universe master from constituent "
                f"names only. Error: {exc}"
            )
            reduced = fallback[["code", "security_name"]].drop_duplicates("code", keep="last")
            return canonical_security_master(reduced, source=f"{self.provider.name}:constituent_fallback"), warnings

    def _has_cached_historical_membership(self) -> bool:
        path = self.root / "universe_membership.parquet"
        if self.config.refresh_mode != "missing" or not path.exists():
            return False
        try:
            cached = pd.read_parquet(path, columns=["membership_source"])
        except (KeyError, ValueError):
            return False
        return cached["membership_source"].astype(str).eq(
            "baostock_month_end_history"
        ).any()

    def _resolve_universe(self) -> tuple[pd.DataFrame, pd.DataFrame, bool, list[str]]:
        warnings: list[str] = []
        if self.config.membership_file:
            membership = load_point_in_time_membership(self.config.membership_file)
            active = membership.loc[
                (membership["effective_from"] <= self.config.end_date)
                & (
                    membership["effective_to"].isna()
                    | (membership["effective_to"] >= self.config.start_date)
                )
            ]
            master, master_warnings = self._provider_master_or_fallback(
                active.assign(security_name=active.get("security_name", active["code"]))
            )
            warnings.extend(master_warnings)
            point_in_time = True
            if set(membership["size_bucket"]) == {"UNKNOWN"}:
                warnings.append(
                    "Point-in-time membership has no LARGE/MID size_bucket labels; "
                    "size-segment portfolio constraints and diagnostics will be unavailable."
                )
        elif (
            self.config.universe == "csi800"
            and (
                (
                    self.status_provider is not None
                    and hasattr(self.status_provider, "fetch_index_membership_snapshots")
                )
                or self._has_cached_historical_membership()
            )
        ):
            cached_membership_path = self.root / "universe_membership.parquet"
            cached_master_path = self.root / "security_master.parquet"
            membership = pd.DataFrame()
            if self.config.refresh_mode == "missing" and cached_membership_path.exists():
                cached = pd.read_parquet(cached_membership_path)
                source_ok = cached.get("membership_source", pd.Series(dtype=str)).eq(
                    "baostock_month_end_history"
                ).any()
                starts_early_enough = (
                    not cached.empty
                    and pd.Timestamp(cached["effective_from"].min())
                    <= self.config.start_date + pd.offsets.MonthEnd(1)
                )
                active_at_end = cached.loc[
                    (pd.to_datetime(cached["effective_from"]) <= self.config.end_date)
                    & (
                        pd.to_datetime(cached["effective_to"], errors="coerce").isna()
                        | (
                            pd.to_datetime(cached["effective_to"], errors="coerce")
                            >= self.config.end_date
                        )
                    )
                ]
                if source_ok and starts_early_enough and active_at_end["code"].nunique() >= 700:
                    membership = cached
                    warnings.append("Reused cached BaoStock historical membership intervals.")
            if membership.empty:
                if self.status_provider is None or not hasattr(
                    self.status_provider, "fetch_index_membership_snapshots"
                ):
                    raise ValueError(
                        "Cached historical CSI 800 membership does not cover the requested range, "
                        "and BaoStock historical membership retrieval is disabled."
                    )
                snapshots, _ = self._retry(
                    lambda: self.status_provider.fetch_index_membership_snapshots(
                        self.config.start_yyyymmdd, self.config.end_yyyymmdd
                    ),
                    "fetch historical CSI 300/500 membership snapshots",
                )
                membership = compress_membership_snapshots(snapshots)
            membership_codes = set(membership["code"].astype(str))
            if self.config.refresh_mode == "missing" and cached_master_path.exists():
                cached_master = pd.read_parquet(cached_master_path)
                if membership_codes.issubset(set(cached_master["code"].astype(str))):
                    master = cached_master
                else:
                    master, master_warnings = self._provider_master_or_fallback(
                        membership.assign(
                            security_name=membership.get("security_name", membership["code"])
                        )
                    )
                    warnings.extend(master_warnings)
            else:
                master, master_warnings = self._provider_master_or_fallback(
                    membership.assign(
                        security_name=membership.get("security_name", membership["code"])
                    )
                )
                warnings.extend(master_warnings)
            point_in_time = True
        elif self.config.universe == "all_a":
            master, master_warnings = self._provider_master_or_fallback()
            warnings.extend(master_warnings)
            snapshot_date = pd.Timestamp.today().normalize()
            membership = master[["code", "security_name"]].copy()
            membership["snapshot_date"] = snapshot_date
            membership["effective_from"] = snapshot_date
            membership["effective_to"] = pd.NaT
            membership["universe"] = "all_a_current"
            membership["membership_mode"] = "current_snapshot"
            membership["size_bucket"] = "UNKNOWN"
            point_in_time = False
        else:
            if self.config.universe not in UNIVERSE_INDEX_CODES:
                raise ValueError(
                    f"Unknown universe {self.config.universe!r}; choose all_a, "
                    f"{', '.join(sorted(UNIVERSE_INDEX_CODES))}, or provide --membership-file"
                )
            segment_definitions = (
                [("csi300", "LARGE"), ("csi500", "MID")]
                if self.config.universe == "csi800"
                else [
                    (
                        self.config.universe,
                        UNIVERSE_SIZE_BUCKETS.get(self.config.universe, "UNKNOWN"),
                    )
                ]
            )
            segment_frames: list[pd.DataFrame] = []
            for segment, size_bucket in segment_definitions:
                index_code = UNIVERSE_INDEX_CODES[segment]
                segment_frame, _ = self._retry(
                    lambda code=index_code: self.provider.fetch_index_constituents(code),
                    f"fetch {segment} constituents",
                )
                segment_frame = segment_frame.copy()
                segment_frame["universe_segment"] = segment
                segment_frame["size_bucket"] = size_bucket
                segment_frames.append(segment_frame)
            constituents = pd.concat(segment_frames, ignore_index=True)
            overlapping = constituents["code"].duplicated(keep=False)
            if overlapping.any():
                overlap_count = int(constituents.loc[overlapping, "code"].nunique())
                warnings.append(
                    f"{overlap_count} codes appeared in multiple size segments; LARGE takes priority."
                )
                constituents["_bucket_order"] = constituents["size_bucket"].map(
                    {"LARGE": 0, "MID": 1, "SMALL": 2, "UNKNOWN": 3}
                )
                constituents = (
                    constituents.sort_values(["_bucket_order", "code"], kind="stable")
                    .drop_duplicates("code", keep="first")
                    .drop(columns="_bucket_order")
                )
            master, master_warnings = self._provider_master_or_fallback(constituents)
            warnings.extend(master_warnings)
            membership = constituents.copy()
            snapshot = membership["snapshot_date"].dropna().max()
            if pd.isna(snapshot):
                snapshot = pd.Timestamp.today().normalize()
            membership["snapshot_date"] = membership["snapshot_date"].fillna(snapshot)
            membership["effective_from"] = membership["snapshot_date"]
            membership["effective_to"] = pd.NaT
            membership["universe"] = self.config.universe
            membership["membership_mode"] = "current_snapshot"
            point_in_time = False

        codes = sorted(membership["code"].drop_duplicates())
        if self.config.max_symbols is not None:
            codes = codes[: self.config.max_symbols]
            membership = membership.loc[membership["code"].isin(codes)].copy()
            warnings.append(f"Universe truncated to {len(codes)} symbols by max_symbols")

        if not point_in_time:
            snapshot_date = pd.Timestamp(membership["snapshot_date"].max())
            if self.config.start_date < snapshot_date:
                warning = (
                    f"{self.config.universe} membership is a current snapshot dated "
                    f"{snapshot_date.date()}, not historical point-in-time membership. Backfilling it to "
                    f"{self.config.start_date.date()} introduces survivorship bias."
                )
                warnings.append(warning)
                if not self.config.allow_current_universe_backfill:
                    raise ValueError(
                        warning
                        + " Re-run with --allow-current-universe-backfill only if you accept this limitation, "
                        "or supply --membership-file with effective_from/effective_to."
                    )

        selected_master = master.loc[master["code"].isin(codes)].copy()
        missing_codes = sorted(set(codes).difference(selected_master["code"]))
        if missing_codes:
            fallback_names = membership.set_index("code").get("security_name", pd.Series(dtype=str))
            missing_rows = pd.DataFrame(
                {
                    "code": missing_codes,
                    "security_name": [fallback_names.get(code, code) for code in missing_codes],
                }
            )
            selected_master = pd.concat(
                [selected_master, canonical_security_master(missing_rows, source="membership_fallback")],
                ignore_index=True,
            )
            warnings.append(f"{len(missing_codes)} constituent codes were absent from the provider master")

        if "size_bucket" in membership:
            bucket_map = (
                membership.sort_values("effective_from", kind="stable")
                .drop_duplicates("code", keep="last")
                .set_index("code")["size_bucket"]
            )
            selected_master["size_bucket"] = (
                selected_master["code"].map(bucket_map).fillna("UNKNOWN")
            )

        return selected_master.sort_values("code"), membership, point_in_time, warnings

    def _sync_benchmark(self, key: str, definition: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame, attempts = self._retry(
            lambda: self.provider.fetch_index_daily(
                definition["index_code"],
                self.config.start_yyyymmdd,
                self.config.end_yyyymmdd,
            ),
            f"fetch benchmark {key}",
        )
        if frame.empty:
            raise ValueError(f"Benchmark {key} returned no rows")
        data = frame.copy()
        data["benchmark_key"] = key
        data["index_code"] = definition["index_code"]
        data["index_name"] = definition["index_name"]
        data["source"] = self.provider.name
        data["source_endpoint"] = frame.attrs.get("source_endpoint", self.provider.name)
        return data, {
            "key": key,
            "index_code": definition["index_code"],
            "rows": len(data),
            "start_date": str(data["date"].min().date()),
            "end_date": str(data["date"].max().date()),
            "attempts": attempts,
            "source_endpoint": frame.attrs.get("source_endpoint", self.provider.name),
            "status": "SUCCESS",
        }

    def _bar_path(self, code: str) -> Path:
        return self.bars_dir / f"{code}.parquet"

    def _existing_covers(
        self,
        path: Path,
        expected_start: pd.Timestamp,
        expected_end: pd.Timestamp,
        code: str,
    ) -> bool:
        if self.config.refresh_mode == "full" or not path.exists():
            return False
        existing = pd.read_parquet(path)
        if existing.empty:
            return False
        dates = pd.to_datetime(existing["date"])
        covers = dates.min() <= expected_start and dates.max() >= expected_end
        receipt = self._completion_receipts.get(code)
        if receipt is not None:
            receipt_covers = (
                receipt["request_start"] <= expected_start
                and receipt["request_end"] >= expected_end
            )
            status_ok = (
                not self.config.fetch_historical_status
                or bool(receipt.get("historical_status_complete", False))
            )
            actions_ok = (
                not hasattr(self.provider, "fetch_corporate_actions")
                or (
                    receipt.get("corporate_actions_status")
                    in {"SUCCESS", "SKIPPED_COMPLETE"}
                    and (self.actions_dir / f"{code}.parquet").exists()
                )
            )
            covers = covers or (receipt_covers and status_ok and actions_ok)
        if self.status_provider is not None and self.config.fetch_historical_status:
            covers = covers and "status_is_point_in_time" in existing and bool(
                existing["status_is_point_in_time"].fillna(False).all()
            )
        if hasattr(self.provider, "fetch_corporate_actions"):
            covers = covers and (self.actions_dir / f"{code}.parquet").exists()
        return covers

    def _sync_stock(
        self,
        security: pd.Series,
        expected_start: pd.Timestamp,
        expected_end: pd.Timestamp,
    ) -> dict[str, Any]:
        code = str(security["code"])
        path = self._bar_path(code)
        action_path = self.actions_dir / f"{code}.parquet"
        if self._existing_covers(path, expected_start, expected_end, code):
            existing = pd.read_parquet(path)
            existing_dates = existing["date"]
            return {
                "code": code,
                "security_name": security["security_name"],
                "status": "SKIPPED_COMPLETE",
                "rows": len(existing_dates),
                "start_date": str(pd.to_datetime(existing_dates).min().date()),
                "end_date": str(pd.to_datetime(existing_dates).max().date()),
                "attempts": 0,
                "historical_status_complete": bool(
                    existing.get("status_is_point_in_time", pd.Series(False)).fillna(False).all()
                ),
                "corporate_actions_status": "SKIPPED_COMPLETE"
                if action_path.exists()
                else "NOT_AVAILABLE",
            }

        raw, raw_attempts = self._retry(
            lambda: self.provider.fetch_stock_daily(
                code,
                expected_start.strftime("%Y%m%d"),
                expected_end.strftime("%Y%m%d"),
                adjust="",
            ),
            f"fetch raw bars {code}",
        )
        adjusted, adjusted_attempts = self._retry(
            lambda: self.provider.fetch_stock_daily(
                code,
                expected_start.strftime("%Y%m%d"),
                expected_end.strftime("%Y%m%d"),
                adjust="qfq",
            ),
            f"fetch adjusted bars {code}",
        )
        if raw.empty or adjusted.empty:
            raise ValueError(f"{code} returned empty raw or adjusted history")
        raw_endpoint = raw.attrs.get("source_endpoint", self.provider.name)
        adjusted_endpoint = adjusted.attrs.get("source_endpoint", self.provider.name)

        # Some public endpoints ignore end_date for the unadjusted series and include the
        # still-incomplete current session.  Enforce the benchmark-aligned request window
        # before demanding an exact raw/adjusted calendar match.
        raw = raw.loc[
            (pd.to_datetime(raw["date"]) >= expected_start)
            & (pd.to_datetime(raw["date"]) <= expected_end)
        ].copy()
        adjusted = adjusted.loc[
            (pd.to_datetime(adjusted["date"]) >= expected_start)
            & (pd.to_datetime(adjusted["date"]) <= expected_end)
        ].copy()
        if raw.empty or adjusted.empty:
            raise ValueError(f"{code} has no raw or adjusted rows inside the common research window")

        adjusted_prices = adjusted[["date", "open", "high", "low", "close"]].copy()
        raw_columns = [column for column in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if column in raw]
        raw_prices = raw[raw_columns].rename(
            columns={
                "open": "raw_open",
                "high": "raw_high",
                "low": "raw_low",
                "close": "raw_close",
            }
        )
        combined = adjusted_prices.merge(raw_prices, on="date", how="inner", validate="one_to_one")
        unmatched_calendar_rows = len(raw) + len(adjusted) - (2 * len(combined))
        if len(combined) != len(raw) or len(combined) != len(adjusted):
            if unmatched_calendar_rows > 1:
                raise ValueError(
                    f"{code} raw/adjusted date mismatch: raw={len(raw)}, adjusted={len(adjusted)}, "
                    f"matched={len(combined)}, unmatched={unmatched_calendar_rows}"
                )
            LOGGER.warning(
                "%s dropped %s isolated raw/adjusted calendar row inside the common window",
                code,
                unmatched_calendar_rows,
            )

        combined["code"] = code
        combined["security_name"] = security["security_name"]
        combined["exchange"] = security["exchange"]
        combined["board"] = security["board"]
        combined["instrument_type"] = security["instrument_type"]
        combined["adjustment_factor"] = combined["close"] / combined["raw_close"]
        status_error = None
        if self.status_provider is not None and self.config.fetch_historical_status:
            try:
                status, _ = self._retry(
                    lambda: self.status_provider.fetch_stock_status(
                        code, self.config.start_yyyymmdd, self.config.end_yyyymmdd
                    ),
                    f"fetch historical status {code}",
                )
                combined = combined.merge(status, on="date", how="left", validate="one_to_one")
            except Exception as exc:  # noqa: BLE001 - failure is recorded as a quality gate
                status_error = str(exc)
        if "status_is_point_in_time" not in combined:
            combined["status_is_point_in_time"] = False
        combined["status_is_point_in_time"] = combined["status_is_point_in_time"].fillna(False)
        if "is_st" not in combined:
            combined["is_st"] = bool(security["is_st_current"])
        else:
            combined["is_st"] = combined["is_st"].fillna(bool(security["is_st_current"]))
        volume_trading = (combined["volume"] > 0) & (combined["amount"] > 0)
        if "is_trading" in combined:
            combined["is_trading"] = combined["is_trading"].fillna(False) & volume_trading
        else:
            combined["is_trading"] = volume_trading
        if "preclose" not in combined:
            combined["preclose"] = combined["raw_close"].shift(1)
        else:
            combined["preclose"] = combined["preclose"].fillna(combined["raw_close"].shift(1))
        if "status_source" not in combined:
            combined["status_source"] = "current_master_fallback"
        combined = add_price_limits(combined)
        combined = add_market_cap_estimate(combined)
        combined["lot_size"] = self.config.default_lot_size
        combined["data_source"] = self.provider.name
        combined["raw_source_endpoint"] = raw_endpoint
        combined["adjusted_source_endpoint"] = adjusted_endpoint
        combined["downloaded_at_utc"] = datetime.now(UTC)

        clean, quality = normalize_and_validate_bars(
            combined,
            default_lot_size=self.config.default_lot_size,
        )
        if path.exists():
            existing = pd.read_parquet(path)
            clean = merge_by_keys(existing, clean, ["date", "code"])
        atomic_write_parquet(clean, path)
        action_status = "NOT_AVAILABLE"
        action_error = None
        if hasattr(self.provider, "fetch_corporate_actions"):
            try:
                actions, _ = self._retry(
                    lambda: self.provider.fetch_corporate_actions(code),
                    f"fetch corporate actions {code}",
                )
                if actions.empty:
                    actions = empty_corporate_actions()
                else:
                    actions = actions.loc[
                        (actions["ex_date"] >= self.config.start_date)
                        & (actions["ex_date"] <= self.config.end_date)
                    ].copy()
                atomic_write_parquet(actions, action_path)
                action_status = "SUCCESS"
            except Exception as exc:  # noqa: BLE001 - failure is recorded per symbol
                action_status = "FAILED"
                action_error = str(exc)
        return {
            "code": code,
            "security_name": security["security_name"],
            "status": "SUCCESS",
            "rows": len(clean),
            "start_date": str(clean["date"].min().date()),
            "end_date": str(clean["date"].max().date()),
            "attempts": max(raw_attempts, adjusted_attempts),
            "raw_source_endpoint": raw_endpoint,
            "adjusted_source_endpoint": adjusted_endpoint,
            "invalid_rows_removed": quality.invalid_rows_removed,
            "unmatched_calendar_rows_dropped": unmatched_calendar_rows,
            "historical_status_complete": bool(clean["status_is_point_in_time"].fillna(False).all()),
            "status_error": status_error,
            "corporate_actions_status": action_status,
            "corporate_actions_error": action_error,
        }

    def run(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        master, membership, point_in_time, warnings = self._resolve_universe()
        active_membership = membership.loc[
            (pd.to_datetime(membership["effective_from"]) <= self.config.end_date)
            & (
                pd.to_datetime(membership["effective_to"], errors="coerce").isna()
                | (pd.to_datetime(membership["effective_to"], errors="coerce") >= self.config.end_date)
            )
        ].drop_duplicates("code", keep="last")
        bucket_counts = (
            {
                str(bucket): int(count)
                for bucket, count in active_membership["size_bucket"]
                .value_counts()
                .items()
            }
            if "size_bucket" in membership
            else {}
        )
        if self.config.max_symbols is not None:
            universe_complete = False
        elif self.config.membership_file:
            universe_complete = int(master["code"].nunique()) >= 200
        elif self.config.universe == "csi800":
            universe_complete = (
                int(bucket_counts.get("LARGE", 0)) >= 270
                and int(bucket_counts.get("MID", 0)) >= 450
            )
        elif self.config.universe == "csi300":
            universe_complete = int(master["code"].nunique()) >= 270
        elif self.config.universe == "csi500":
            universe_complete = int(master["code"].nunique()) >= 450
        elif self.config.universe == "csi1000":
            universe_complete = int(master["code"].nunique()) >= 900
        else:
            universe_complete = int(master["code"].nunique()) >= 1_000
        if not universe_complete:
            warnings.append(
                "Universe completeness gate failed; the symbol count or LARGE/MID segment counts "
                "are below the minimum expected for this research universe."
            )
        atomic_write_parquet(master, self.root / "security_master.parquet")
        atomic_write_parquet(membership, self.root / "universe_membership.parquet")

        point_in_time_industry = False
        free_historical_membership = bool(
            point_in_time
            and not self.config.membership_file
            and self.config.universe == "csi800"
            and membership.get("membership_source", pd.Series(dtype=str))
            .eq("baostock_month_end_history")
            .any()
        )
        industry_required = not free_historical_membership

        def industry_coverage(history: pd.DataFrame) -> tuple[bool, int]:
            if "is_point_in_time" in history:
                history = history.loc[history["is_point_in_time"].fillna(False).astype(bool)]
            membership_starts = (
                membership.assign(
                    effective_from=pd.to_datetime(membership["effective_from"], errors="coerce")
                )
                .groupby("code")["effective_from"]
                .min()
            )
            industry_starts = history.groupby("code")["effective_from"].min()
            covered = 0
            for code in master["code"].astype(str):
                required = max(
                    self.config.start_date,
                    pd.Timestamp(membership_starts.get(code, self.config.start_date)),
                )
                first_industry = industry_starts.get(code, pd.NaT)
                if pd.notna(first_industry) and pd.Timestamp(first_industry) <= required:
                    covered += 1
            return covered == len(master), covered

        if self.config.industry_file:
            try:
                industry_path = Path(self.config.industry_file)
                raw_industry = (
                    pd.read_parquet(industry_path)
                    if industry_path.suffix.lower() in {".parquet", ".pq"}
                    else pd.read_csv(industry_path, dtype={"code": "string"})
                )
                industry_history = normalize_industry_history(raw_industry)
                selected_codes = set(master["code"].astype(str))
                industry_history = industry_history.loc[
                    industry_history["code"].astype(str).isin(selected_codes)
                ].copy()
                atomic_write_parquet(industry_history, self.root / "industry_history.parquet")
                point_in_time_industry, covered_count = industry_coverage(industry_history)
                if not point_in_time_industry:
                    warnings.append(
                        f"Supplied industry history covers the required start date for "
                        f"{covered_count}/{len(selected_codes)} symbols."
                    )
            except Exception as exc:  # noqa: BLE001 - failure is recorded as a quality gate
                warnings.append(f"Supplied point-in-time industry history failed: {exc}")
        elif free_historical_membership:
            industry_required = False
            empty_industry = pd.DataFrame(
                columns=[
                    "code",
                    "effective_from",
                    "effective_to",
                    "industry_code",
                    "industry_name",
                    "industry_standard",
                    "is_point_in_time",
                ]
            )
            atomic_write_parquet(empty_industry, self.root / "industry_history.parquet")
            warnings.append(
                "Historical industry neutralization is intentionally disabled in the free-data "
                "research profile. Current industry snapshots are not backfilled into history."
            )
        elif hasattr(self.provider, "fetch_industry_history"):
            try:
                industry_history, _ = self._retry(
                    self.provider.fetch_industry_history, "fetch point-in-time industry history"
                )
                selected_codes = set(master["code"].astype(str))
                industry_history = industry_history.loc[
                    industry_history["code"].astype(str).isin(selected_codes)
                    & (industry_history["effective_from"] <= self.config.end_date)
                ].copy()
                atomic_write_parquet(industry_history, self.root / "industry_history.parquet")
                point_in_time_industry, covered_count = industry_coverage(industry_history)
                if not point_in_time_industry:
                    warnings.append(
                        f"Point-in-time industry history covers the required start date for "
                        f"{covered_count}/{len(selected_codes)} symbols."
                    )
            except Exception as exc:  # noqa: BLE001 - failure is recorded as a quality gate
                warnings.append(f"Point-in-time industry history failed: {exc}")
                if hasattr(self.provider, "fetch_current_industry_snapshot"):
                    try:
                        selected_codes = set(master["code"].astype(str))
                        current_industry, _ = self._retry(
                            lambda: self.provider.fetch_current_industry_snapshot(selected_codes),
                            "fetch current industry snapshot fallback",
                        )
                        current_industry = current_industry.copy()
                        current_industry["effective_from"] = self.config.start_date
                        current_industry["industry_standard"] = "FREE_CURRENT_SNAPSHOT_BACKFILL"
                        current_industry["is_point_in_time"] = False
                        current_industry = normalize_industry_history(current_industry)
                        current_industry = current_industry.loc[
                            current_industry["code"].astype(str).isin(selected_codes)
                        ].copy()
                        atomic_write_parquet(
                            current_industry, self.root / "industry_history.parquet"
                        )
                        warnings.append(
                            "Historical industry data was unavailable. A current industry snapshot "
                            f"was explicitly backfilled for {current_industry['code'].nunique()}/"
                            f"{len(selected_codes)} symbols so engineering or limited research can "
                            "run. It is marked is_point_in_time=false and cannot pass the formal "
                            "research quality gate."
                        )
                    except Exception as fallback_exc:  # noqa: BLE001 - gate remains failed
                        warnings.append(f"Current industry snapshot fallback failed: {fallback_exc}")

        benchmark_frames: list[pd.DataFrame] = []
        benchmark_results: list[dict[str, Any]] = []
        for key, definition in BENCHMARKS.items():
            try:
                frame, result = self._sync_benchmark(key, definition)
                benchmark_frames.append(frame)
                benchmark_results.append(result)
            except Exception as exc:  # noqa: BLE001 - failure is recorded per benchmark
                benchmark_results.append(
                    {
                        "key": key,
                        "index_code": definition["index_code"],
                        "status": "FAILED",
                        "error": str(exc),
                    }
                )
                warnings.append(f"Benchmark {key} failed: {exc}")

        expected_start = self.config.start_date
        expected_end = self.config.end_date
        if benchmark_frames:
            benchmarks = pd.concat(benchmark_frames, ignore_index=True)
            expected_start = max(self.config.start_date, pd.Timestamp(benchmarks["date"].min()))
            expected_end = min(self.config.end_date, pd.Timestamp(benchmarks["date"].max()))
            benchmark_path = self.root / "benchmarks.parquet"
            if benchmark_path.exists():
                benchmarks = merge_by_keys(
                    pd.read_parquet(benchmark_path),
                    benchmarks,
                    ["date", "benchmark_key"],
                )
            atomic_write_parquet(benchmarks, benchmark_path)

        stock_results: list[dict[str, Any]] = []
        records = [row for _, row in master.sort_values("code").iterrows()]

        def stock_request_bounds(code: str) -> tuple[pd.Timestamp, pd.Timestamp]:
            rows = membership.loc[membership["code"].astype(str).eq(code)].copy()
            if rows.empty:
                return expected_start, expected_end
            first_membership = pd.Timestamp(rows["effective_from"].min())
            stock_start = max(expected_start, first_membership - pd.Timedelta(days=252))
            effective_to = pd.to_datetime(rows["effective_to"], errors="coerce")
            active_through_end = bool(effective_to.isna().any() or (effective_to >= expected_end).any())
            if active_through_end:
                stock_end = expected_end
            else:
                stock_end = min(expected_end, pd.Timestamp(effective_to.max()) + pd.Timedelta(days=60))
            return stock_start, stock_end

        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = {
                executor.submit(
                    self._sync_stock,
                    security,
                    *stock_request_bounds(str(security["code"])),
                ): str(security["code"])
                for security in records
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                code = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - failure is recorded per symbol
                    result = {"code": code, "status": "FAILED", "error": str(exc)}
                stock_results.append(result)
                if completed % 25 == 0 or completed == len(futures):
                    LOGGER.info("Stock sync progress: %s/%s", completed, len(futures))

        successful = sum(result["status"] == "SUCCESS" for result in stock_results)
        skipped = sum(result["status"] == "SKIPPED_COMPLETE" for result in stock_results)
        failed = sum(result["status"] == "FAILED" for result in stock_results)
        all_benchmarks_ok = all(result["status"] == "SUCCESS" for result in benchmark_results)
        status_history_available = self.config.fetch_historical_status and bool(stock_results) and all(
            result.get("historical_status_complete", False)
            for result in stock_results
            if result.get("status") != "FAILED"
        ) and failed == 0
        actions_available = bool(stock_results) and all(
            result.get("corporate_actions_status") in {"SUCCESS", "SKIPPED_COMPLETE"}
            for result in stock_results
            if result.get("status") != "FAILED"
        ) and failed == 0
        action_frames = [pd.read_parquet(path) for path in sorted(self.actions_dir.glob("*.parquet"))]
        if action_frames:
            combined_actions = pd.concat(action_frames, ignore_index=True)
            if not combined_actions.empty:
                combined_actions = combined_actions.drop_duplicates(["code", "ex_date"], keep="last")
            atomic_write_parquet(combined_actions, self.root / "corporate_actions.parquet")
        storage_audit = audit_market_data(
            self.root,
            min_rows_per_symbol=1,
            write_report=True,
        )
        research_ready = (
            point_in_time
            and universe_complete
            and failed == 0
            and all_benchmarks_ok
            and status_history_available
            and (point_in_time_industry or not industry_required)
            and actions_available
            and storage_audit["passed"]
        )
        core_market_data_complete = (
            universe_complete and failed == 0 and all_benchmarks_ok and storage_audit["passed"]
        )
        if research_ready and not industry_required:
            evidence_grade = "REPRODUCIBLE_FREE_DATA"
        elif research_ready:
            evidence_grade = "FORMAL"
        elif core_market_data_complete and self.config.max_symbols is None:
            evidence_grade = "LIMITED_RESEARCH"
        else:
            evidence_grade = "ENGINEERING_ONLY"
        if not status_history_available:
            warnings.append(
                "Historical ST/suspension status is not provided by this free-data path; current ST status "
                "must not be treated as point-in-time history."
            )
        if not actions_available:
            warnings.append("Corporate-action history is incomplete; dividend/share-event accounting is gated.")

        report = {
            "run_id": started_at.strftime("%Y%m%dT%H%M%SZ"),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "provider": {"name": self.provider.name, "version": self.provider.version},
            "evidence_grade": evidence_grade,
            "request": {
                "universe": self.config.universe,
                "membership_source": (
                    "file"
                    if self.config.membership_file
                    else (
                        "baostock_month_end_history"
                        if free_historical_membership
                        else "current_snapshot"
                    )
                ),
                "membership_file": self.config.membership_file,
                "industry_file": self.config.industry_file,
                "start_date": str(self.config.start_date.date()),
                "end_date": str(self.config.end_date.date()),
                "refresh_mode": self.config.refresh_mode,
                "workers": self.config.workers,
                "max_symbols": self.config.max_symbols,
                "fetch_historical_status": self.config.fetch_historical_status,
            },
            "quality_gates": {
                "point_in_time_membership": point_in_time,
                "universe_completeness": universe_complete,
                "historical_security_status": status_history_available,
                "point_in_time_industry": point_in_time_industry,
                "historical_industry_required": industry_required,
                "corporate_actions": actions_available,
                "raw_and_adjusted_prices": failed == 0,
                "required_benchmarks_downloaded": all_benchmarks_ok,
                "storage_audit_passed": storage_audit["passed"],
                "research_ready": research_ready,
            },
            "summary": {
                "requested_symbols": len(records),
                "successful_symbols": successful,
                "skipped_complete_symbols": skipped,
                "failed_symbols": failed,
                "size_bucket_counts": bucket_counts,
            },
            "benchmarks": benchmark_results,
            "storage_audit": storage_audit["summary"],
            "stocks": sorted(stock_results, key=lambda item: item["code"]),
            "warnings": list(dict.fromkeys(warnings)),
        }
        atomic_write_json(report, self.metadata_dir / "download_report.json")
        atomic_write_json(report, self.metadata_dir / f"download_report_{report['run_id']}.json")
        if self.status_provider is not None and hasattr(self.status_provider, "close"):
            self.status_provider.close()
        return report
