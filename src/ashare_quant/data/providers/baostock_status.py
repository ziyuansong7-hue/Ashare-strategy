from __future__ import annotations

import threading

import pandas as pd

from ashare_quant.data.schema import normalize_symbol
from ashare_quant.data.securities import exchange_from_code


class BaoStockStatusProvider:
    """Serialized BaoStock client for point-in-time ST and trading-status fields."""

    name = "baostock"

    def __init__(self) -> None:
        try:
            import baostock as bs
        except ImportError as exc:
            raise RuntimeError('BaoStock is required: pip install -e ".[data]"') from exc
        self._bs = bs
        self._lock = threading.Lock()
        self._logged_in = False

    @property
    def version(self) -> str:
        return str(getattr(self._bs, "__version__", "unknown"))

    def _login(self) -> None:
        if self._logged_in:
            return
        result = self._bs.login()
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {result.error_code} {result.error_msg}")
        self._logged_in = True

    def close(self) -> None:
        with self._lock:
            if self._logged_in:
                self._bs.logout()
                self._logged_in = False

    def fetch_stock_status(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = normalize_symbol(code)
        exchange = exchange_from_code(symbol)
        prefix = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}.get(exchange)
        if prefix is None:
            raise ValueError(f"Unsupported BaoStock symbol: {code}")
        fields = "date,preclose,tradestatus,isST"
        with self._lock:
            self._login()
            result = self._bs.query_history_k_data_plus(
                f"{prefix}.{symbol}",
                fields,
                start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3",
            )
            if result.error_code != "0":
                raise RuntimeError(
                    f"BaoStock status query failed for {symbol}: "
                    f"{result.error_code} {result.error_msg}"
                )
            rows = []
            while result.next():
                rows.append(result.get_row_data())
        frame = pd.DataFrame(rows, columns=fields.split(","))
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["preclose"] = pd.to_numeric(frame["preclose"], errors="coerce")
        frame["is_trading"] = frame["tradestatus"].astype(str).eq("1")
        frame["is_st"] = frame["isST"].astype(str).eq("1")
        frame["status_is_point_in_time"] = True
        frame["status_source"] = "baostock_history_k_data_plus"
        return frame[
            [
                "date",
                "preclose",
                "is_trading",
                "is_st",
                "status_is_point_in_time",
                "status_source",
            ]
        ]

    def fetch_index_membership_snapshots(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Return month-end historical CSI 300/500 snapshots from BaoStock."""
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        with self._lock:
            self._login()
            calendar = self._bs.query_trade_dates(
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
            if calendar.error_code != "0":
                raise RuntimeError(
                    "BaoStock trade-calendar query failed: "
                    f"{calendar.error_code} {calendar.error_msg}"
                )
            calendar_rows = []
            while calendar.next():
                calendar_rows.append(calendar.get_row_data())
            trading_days = pd.DataFrame(calendar_rows, columns=calendar.fields)
            if trading_days.empty:
                raise ValueError("BaoStock returned no trading dates for membership reconstruction")
            trading_days["calendar_date"] = pd.to_datetime(
                trading_days["calendar_date"], errors="coerce"
            )
            trading_days = trading_days.loc[
                trading_days["is_trading_day"].astype(str).eq("1")
            ]
            month_ends = (
                trading_days.groupby(trading_days["calendar_date"].dt.to_period("M"))[
                    "calendar_date"
                ]
                .max()
                .sort_values()
            )

            rows: list[dict[str, object]] = []
            queries = [
                ("csi300", "LARGE", self._bs.query_hs300_stocks),
                ("csi500", "MID", self._bs.query_zz500_stocks),
            ]
            for snapshot_date in month_ends:
                date_text = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
                for segment, bucket, query in queries:
                    result = query(date_text)
                    if result.error_code != "0":
                        raise RuntimeError(
                            f"BaoStock {segment} membership query failed for {date_text}: "
                            f"{result.error_code} {result.error_msg}"
                        )
                    while result.next():
                        values = dict(zip(result.fields, result.get_row_data(), strict=True))
                        rows.append(
                            {
                                "snapshot_date": pd.Timestamp(snapshot_date),
                                "provider_snapshot_date": pd.to_datetime(
                                    values.get("date"), errors="coerce"
                                ),
                                "code": normalize_symbol(values.get("code", "")),
                                "security_name": str(values.get("code_name", "")).strip(),
                                "universe_segment": segment,
                                "size_bucket": bucket,
                            }
                        )
        snapshots = pd.DataFrame(rows)
        if snapshots.empty:
            raise ValueError("BaoStock returned no historical CSI 300/500 membership rows")
        return snapshots
