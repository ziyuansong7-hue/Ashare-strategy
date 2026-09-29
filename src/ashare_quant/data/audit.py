from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.securities import BENCHMARKS
from ashare_quant.data.storage import atomic_write_json

BAR_REQUIRED = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
RAW_PRICE_COLUMNS = {"raw_open", "raw_high", "raw_low", "raw_close"}


def audit_market_data(
    data_dir: str | Path,
    *,
    min_rows_per_symbol: int = 120,
    write_report: bool = True,
) -> dict[str, Any]:
    root = Path(data_dir)
    failures: list[str] = []
    warnings: list[str] = []
    symbol_results: list[dict[str, Any]] = []

    master_path = root / "security_master.parquet"
    if not master_path.exists():
        failures.append("security_master.parquet is missing")
        master = pd.DataFrame()
    else:
        master = pd.read_parquet(master_path)
        required_master = {"code", "instrument_type", "exchange"}
        missing = required_master.difference(master.columns)
        if missing:
            failures.append(f"security master missing columns: {sorted(missing)}")
        else:
            master["code"] = master["code"].astype(str).str.zfill(6)
            if master["code"].duplicated().any():
                failures.append("security master contains duplicate codes")
            non_stocks = master.loc[master["instrument_type"] != "STOCK", "code"].tolist()
            if non_stocks:
                failures.append(f"security master contains non-stock instruments: {non_stocks[:10]}")
            unknown_exchange = master.loc[master["exchange"] == "UNKNOWN", "code"].tolist()
            if unknown_exchange:
                failures.append(f"security master contains unknown exchanges: {unknown_exchange[:10]}")

    bars_dir = root / "bars"
    bar_paths = sorted(bars_dir.glob("*.parquet")) if bars_dir.exists() else []
    if not bar_paths:
        failures.append("no per-symbol Parquet files found under bars/")

    bar_codes: set[str] = set()
    total_rows = 0
    point_in_time_status_rows = 0
    market_cap_rows = 0
    all_start_dates: list[pd.Timestamp] = []
    all_end_dates: list[pd.Timestamp] = []
    for path in bar_paths:
        code = path.stem.zfill(6)
        bar_codes.add(code)
        result: dict[str, Any] = {"code": code, "file": path.name, "status": "PASS"}
        try:
            frame = pd.read_parquet(path)
            missing = BAR_REQUIRED.difference(frame.columns)
            if missing:
                raise ValueError(f"missing columns {sorted(missing)}")
            if not RAW_PRICE_COLUMNS.issubset(frame.columns):
                raise ValueError(f"missing raw price columns {sorted(RAW_PRICE_COLUMNS.difference(frame.columns))}")
            normalized_codes = frame["code"].astype(str).str.zfill(6)
            if set(normalized_codes) != {code}:
                raise ValueError("file name and row codes do not match")
            dates = pd.to_datetime(frame["date"], errors="coerce")
            if dates.isna().any():
                raise ValueError("invalid dates")
            if frame.duplicated(["date", "code"]).any():
                raise ValueError("duplicate date/code rows")
            numeric = frame[["open", "high", "low", "close", *sorted(RAW_PRICE_COLUMNS)]].apply(
                pd.to_numeric, errors="coerce"
            )
            if numeric.isna().any().any() or (numeric <= 0).any().any():
                raise ValueError("non-positive or missing adjusted/raw prices")
            if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
                raise ValueError("invalid adjusted high price")
            if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
                raise ValueError("invalid adjusted low price")
            result.update(
                {
                    "rows": len(frame),
                    "start_date": str(dates.min().date()),
                    "end_date": str(dates.max().date()),
                }
            )
            if "status_is_point_in_time" in frame:
                covered = int(frame["status_is_point_in_time"].fillna(False).sum())
                point_in_time_status_rows += covered
                result["point_in_time_status_coverage"] = covered / len(frame)
            else:
                result["point_in_time_status_coverage"] = 0.0
            if "market_cap" in frame:
                covered = int(pd.to_numeric(frame["market_cap"], errors="coerce").gt(0).sum())
                market_cap_rows += covered
                result["market_cap_coverage"] = covered / len(frame)
            else:
                result["market_cap_coverage"] = 0.0
            if len(frame) < min_rows_per_symbol:
                result["status"] = "WARN"
                result["warning"] = f"only {len(frame)} rows; expected at least {min_rows_per_symbol}"
                warnings.append(f"{code} has only {len(frame)} rows")
            total_rows += len(frame)
            all_start_dates.append(dates.min())
            all_end_dates.append(dates.max())
        except Exception as exc:  # noqa: BLE001 - audit must isolate any corrupt file
            result.update({"status": "FAIL", "error": str(exc)})
            failures.append(f"{code}: {exc}")
        symbol_results.append(result)

    if not master.empty and "code" in master:
        master_codes = set(master["code"].astype(str).str.zfill(6))
        missing_files = sorted(master_codes.difference(bar_codes))
        orphan_files = sorted(bar_codes.difference(master_codes))
        if missing_files:
            failures.append(f"{len(missing_files)} master symbols have no bar file")
        if orphan_files:
            failures.append(f"bar files absent from security master: {orphan_files[:10]}")

    benchmark_path = root / "benchmarks.parquet"
    benchmark_keys: list[str] = []
    if not benchmark_path.exists():
        failures.append("benchmarks.parquet is missing")
    else:
        benchmarks = pd.read_parquet(benchmark_path)
        required = {"date", "benchmark_key", "close"}
        missing = required.difference(benchmarks.columns)
        if missing:
            failures.append(f"benchmark data missing columns: {sorted(missing)}")
        else:
            benchmark_keys = sorted(benchmarks["benchmark_key"].dropna().astype(str).unique())
            expected_keys = set(BENCHMARKS)
            absent = sorted(expected_keys.difference(benchmark_keys))
            if absent:
                failures.append(f"missing required benchmarks: {absent}")
            if benchmarks.duplicated(["date", "benchmark_key"]).any():
                failures.append("benchmark data contains duplicate date/key rows")

    industry_path = root / "industry_history.parquet"
    industry_symbols = 0
    if industry_path.exists():
        industry = pd.read_parquet(industry_path)
        industry_symbols = int(industry.get("code", pd.Series(dtype=str)).nunique())
    else:
        warnings.append("industry_history.parquet is missing")
    action_path = root / "corporate_actions.parquet"
    corporate_action_rows = 0
    if action_path.exists():
        corporate_action_rows = len(pd.read_parquet(action_path))
    else:
        warnings.append("corporate_actions.parquet is missing")

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_dir": str(root.resolve()),
        "passed": not failures,
        "summary": {
            "symbols": len(bar_paths),
            "rows": total_rows,
            "start_date": str(min(all_start_dates).date()) if all_start_dates else None,
            "end_date": str(max(all_end_dates).date()) if all_end_dates else None,
            "benchmarks": benchmark_keys,
            "failures": len(failures),
            "warnings": len(warnings),
            "point_in_time_status_coverage": point_in_time_status_rows / total_rows
            if total_rows
            else 0.0,
            "market_cap_coverage": market_cap_rows / total_rows if total_rows else 0.0,
            "industry_history_symbols": industry_symbols,
            "corporate_action_rows": corporate_action_rows,
        },
        "failures": failures,
        "warnings": warnings,
        "symbols": symbol_results,
    }
    if write_report:
        atomic_write_json(report, root / "metadata" / "audit_report.json")
    return report
