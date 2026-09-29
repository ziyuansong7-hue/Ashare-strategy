"""Download one A-share daily history file through AKShare.

This utility creates data in the canonical schema consumed by ``ashare-quant run`` and
preserves six-character symbols. It is deliberately separate from model training.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import re
import sqlite3

import pandas as pd

from ashare_quant.data.schema import normalize_and_validate_bars, normalize_symbol


def _validate_date(value: str) -> str:
    try:
        parsed = pd.to_datetime(value, format="%Y%m%d", errors="raise")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYYMMDD format") from exc
    return parsed.strftime("%Y%m%d")


def download_stock_daily(code: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError('AKShare is required: pip install -e ".[data]"') from exc

    symbol = normalize_symbol(code)
    if not re.fullmatch(r"\d{6}", symbol):
        raise ValueError(f"Invalid A-share symbol: {code!r}")

    raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=adjust,
    )
    if raw.empty:
        return raw

    frame = raw.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "price_change",
            "换手率": "turnover",
        }
    )
    frame["code"] = symbol
    clean, report = normalize_and_validate_bars(frame)
    print(
        f"Validated {report.rows} rows for {symbol} "
        f"({report.start_date} to {report.end_date}); removed {report.invalid_rows_removed} invalid rows"
    )
    return clean


def save_to_csv(frame: pd.DataFrame, code: str, output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{normalize_symbol(code)}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_to_sqlite(frame: pd.DataFrame, database: str | Path, table: str = "daily_price") -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("Unsafe SQLite table name")
    columns = ["code", "date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    payload = frame.reindex(columns=columns).copy()
    payload["date"] = payload["date"].dt.strftime("%Y-%m-%d")

    with sqlite3.connect(database) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                amount REAL NOT NULL,
                turnover REAL,
                PRIMARY KEY (code, date)
            )
            """
        )
        connection.executemany(
            f"""
            INSERT INTO {table}
                (code, date, open, high, low, close, volume, amount, turnover)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code, date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                amount=excluded.amount,
                turnover=excluded.turnover
            """,
            payload.itertuples(index=False, name=None),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one A-share daily history through AKShare")
    parser.add_argument("--code", required=True, help="Six-character symbol, for example 000001")
    parser.add_argument("--start", type=_validate_date, default="20150101")
    parser.add_argument("--end", type=_validate_date, default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--adjust", choices=["", "qfq", "hfq"], default="qfq")
    parser.add_argument("--format", choices=["csv", "sqlite", "both"], default="csv")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--database", default="data/ashare_quant.db")
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must not be later than --end")
    frame = download_stock_daily(args.code, args.start, args.end, args.adjust)
    if frame.empty:
        raise SystemExit("No rows returned; check the symbol, dates, and data-provider availability")

    if args.format in {"csv", "both"}:
        path = save_to_csv(frame, args.code, args.output_dir)
        print(f"Saved {len(frame)} rows to {path}")
    if args.format in {"sqlite", "both"}:
        Path(args.database).parent.mkdir(parents=True, exist_ok=True)
        save_to_sqlite(frame, args.database)
        print(f"Upserted {len(frame)} rows into {args.database}")


if __name__ == "__main__":
    main()

