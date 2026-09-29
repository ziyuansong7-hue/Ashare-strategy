from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "market_cap",
    "raw_open",
    "raw_close",
    "limit_up",
    "limit_down",
]


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    symbols: int
    start_date: str
    end_date: str
    duplicate_rows_removed: int
    invalid_rows_removed: int
    non_trading_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_symbol(value: object) -> str:
    text = str(value).strip().upper()
    text = text.removesuffix(".0")
    text = re.sub(r"^(SH|SZ|BJ)[.]", "", text)
    text = re.sub(r"[.](SH|SZ|BJ)$", "", text)
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


def normalize_and_validate_bars(
    bars: pd.DataFrame,
    *,
    default_lot_size: int = 100,
) -> tuple[pd.DataFrame, DataQualityReport]:
    missing = REQUIRED_COLUMNS.difference(bars.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {sorted(missing)}")

    frame = bars.copy()
    frame["code"] = frame["code"].map(normalize_symbol)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_count = int(frame.duplicated(["date", "code"], keep="last").sum())
    frame = frame.drop_duplicates(["date", "code"], keep="last")

    finite_required = np.isfinite(frame[["open", "high", "low", "close", "volume", "amount"]]).all(axis=1)
    valid = (
        frame["date"].notna()
        & frame["code"].str.fullmatch(r"\d{6}", na=False)
        & finite_required
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame[["volume", "amount"]] >= 0).all(axis=1)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    invalid_count = int((~valid).sum())
    frame = frame.loc[valid].copy()

    if "is_trading" not in frame:
        frame["is_trading"] = (frame["volume"] > 0) & (frame["amount"] > 0)
    else:
        frame["is_trading"] = frame["is_trading"].fillna(False).astype(bool)

    if "is_st" not in frame:
        frame["is_st"] = False
    else:
        frame["is_st"] = frame["is_st"].fillna(False).astype(bool)

    if "industry" not in frame:
        frame["industry"] = "UNKNOWN"
    frame["industry"] = frame["industry"].fillna("UNKNOWN").astype(str)

    if "lot_size" not in frame:
        frame["lot_size"] = default_lot_size
    frame["lot_size"] = (
        pd.to_numeric(frame["lot_size"], errors="coerce")
        .fillna(default_lot_size)
        .clip(lower=1)
        .astype(int)
    )

    frame = frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid rows remain after market-data validation")

    report = DataQualityReport(
        rows=len(frame),
        symbols=frame["code"].nunique(),
        start_date=str(frame["date"].min().date()),
        end_date=str(frame["date"].max().date()),
        duplicate_rows_removed=duplicate_count,
        invalid_rows_removed=invalid_count,
        non_trading_rows=int((~frame["is_trading"]).sum()),
    )
    return frame, report

