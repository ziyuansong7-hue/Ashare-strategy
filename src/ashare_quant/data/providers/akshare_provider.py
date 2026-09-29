from __future__ import annotations

import logging
import re

import pandas as pd

from ashare_quant.data.enrichment import (
    merge_corporate_actions,
    normalize_dividend_actions,
    normalize_industry_history,
    normalize_rights_actions,
)
from ashare_quant.data.schema import normalize_symbol

LOGGER = logging.getLogger(__name__)

HISTORY_COLUMN_MAP = {
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


def _normalize_history(frame: pd.DataFrame, *, turnover_is_percent: bool = False) -> pd.DataFrame:
    data = frame.rename(columns=HISTORY_COLUMN_MAP).copy()
    if data.empty:
        return data
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Provider history response is missing columns: {sorted(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "amplitude",
        "pct_change",
        "price_change",
    ]:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if turnover_is_percent and "turnover" in data:
        data["turnover"] = data["turnover"] / 100.0
    return data.sort_values("date", kind="stable").reset_index(drop=True)


def _stock_market_symbol(code: str) -> str:
    symbol = normalize_symbol(code)
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    if symbol.startswith(("4", "8", "92")):
        return f"bj{symbol}"
    raise ValueError(f"Unsupported A-share code: {code}")


def _index_market_symbol(index_code: str) -> str:
    symbol = normalize_symbol(index_code)
    return f"sz{symbol}" if symbol.startswith("399") else f"sh{symbol}"


class AKShareProvider:
    """AKShare adapter with normalized English field names."""

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError('AKShare is required: pip install -e ".[data]"') from exc
        self._ak = ak
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "akshare_multi_endpoint"

    @property
    def version(self) -> str:
        return str(self._ak.__version__)

    def fetch_security_master(self) -> pd.DataFrame:
        try:
            raw = self._ak.stock_info_a_code_name()
        except Exception as primary_error:  # noqa: BLE001 - upstream exceptions are heterogeneous
            try:
                raw = self._ak.stock_zh_a_spot_em()
            except Exception as fallback_error:
                raise RuntimeError(
                    "Both AKShare security-master endpoints failed: "
                    f"stock_info_a_code_name={primary_error!r}; stock_zh_a_spot_em={fallback_error!r}"
                ) from fallback_error

        code_column = next((name for name in ["code", "代码", "证券代码"] if name in raw), None)
        name_column = next((name for name in ["name", "名称", "证券简称"] if name in raw), None)
        if code_column is None or name_column is None:
            raise ValueError(f"Unrecognized security-master columns: {list(raw.columns)}")
        return pd.DataFrame(
            {
                "code": raw[code_column].map(normalize_symbol),
                "security_name": raw[name_column].astype(str).str.strip(),
            }
        ).drop_duplicates("code", keep="last")

    def fetch_index_constituents(self, index_code: str) -> pd.DataFrame:
        raw = self._ak.index_stock_cons_csindex(symbol=index_code)
        mapping = {
            "日期": "snapshot_date",
            "指数代码": "index_code",
            "指数名称": "index_name",
            "成分券代码": "code",
            "成分券名称": "security_name",
            "交易所": "exchange_name",
        }
        data = raw.rename(columns=mapping).copy()
        required = {"code", "security_name"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Unrecognized constituent columns: {list(raw.columns)}")
        data["code"] = data["code"].map(normalize_symbol)
        if "index_code" in data:
            data["index_code"] = data["index_code"].astype(str).map(normalize_symbol)
        else:
            data["index_code"] = normalize_symbol(index_code)
        if "snapshot_date" in data:
            data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce").dt.normalize()
        else:
            data["snapshot_date"] = pd.Timestamp.today().normalize()
        return data.reset_index(drop=True)

    def fetch_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        adjust: str,
    ) -> pd.DataFrame:
        if adjust not in {"", "qfq", "hfq"}:
            raise ValueError(f"Unsupported adjustment mode: {adjust}")
        try:
            raw = self._ak.stock_zh_a_hist(
                symbol=normalize_symbol(code),
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=self.timeout_seconds,
            )
            if raw.empty:
                raise ValueError("Eastmoney endpoint returned no rows")
            result = _normalize_history(raw, turnover_is_percent=True)
            result.attrs["source_endpoint"] = "eastmoney_stock_zh_a_hist"
            return result
        except Exception as primary_error:  # noqa: BLE001 - fallback on any endpoint failure
            try:
                raw = self._ak.stock_zh_a_daily(
                    symbol=_stock_market_symbol(code),
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
                if raw.empty:
                    raise ValueError("Sina endpoint returned no rows")
                result = _normalize_history(raw)
                result.attrs["source_endpoint"] = "sina_stock_zh_a_daily"
                result.attrs["fallback_reason"] = repr(primary_error)
                return result
            except Exception as secondary_error:  # noqa: BLE001 - try final independent source
                try:
                    raw = self._ak.stock_zh_a_hist_tx(
                        symbol=_stock_market_symbol(code),
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                        timeout=self.timeout_seconds,
                    )
                    if raw.empty:
                        raise ValueError("Tencent endpoint returned no rows")
                    result = _normalize_history(raw)
                    result.attrs["source_endpoint"] = "tencent_stock_zh_a_hist_tx"
                    result.attrs["fallback_reason"] = (
                        f"eastmoney={primary_error!r}; sina={secondary_error!r}"
                    )
                    return result
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Stock history endpoints failed for {code}: "
                        f"eastmoney={primary_error!r}; sina={secondary_error!r}; "
                        f"tencent={fallback_error!r}"
                    ) from fallback_error

    def fetch_industry_history(self) -> pd.DataFrame:
        return normalize_industry_history(self._ak.stock_industry_clf_hist_sw())

    def fetch_current_industry_snapshot(self, codes: set[str] | None = None) -> pd.DataFrame:
        """Fetch a free current industry snapshot.

        This is intentionally not presented as point-in-time history.  The synchronizer
        may use it as an explicitly labelled engineering fallback when the historical
        SW endpoint is unavailable.
        """
        selected = {normalize_symbol(code) for code in codes} if codes else None
        try:
            result = self._fetch_eastmoney_industry_snapshot(selected)
            result.attrs["source_endpoint"] = "eastmoney_industry_boards"
            return result
        except Exception as board_error:
            if selected is None:
                raise RuntimeError(
                    f"Eastmoney current-industry snapshot failed: {board_error!r}"
                ) from board_error
            result = self._fetch_cninfo_industry_snapshot(selected)
            if result.empty:
                raise RuntimeError(
                    "Both current-industry fallbacks failed: "
                    f"eastmoney={board_error!r}; cninfo returned no usable rows"
                ) from board_error
            result.attrs["source_endpoint"] = "cninfo_stock_industry_change_latest"
            result.attrs["fallback_reason"] = repr(board_error)
            return result

    def _fetch_eastmoney_industry_snapshot(self, selected: set[str] | None) -> pd.DataFrame:
        boards = self._ak.stock_board_industry_name_em()
        name_column = next(
            (name for name in ["板块名称", "行业名称", "name"] if name in boards), None
        )
        board_code_column = next(
            (name for name in ["板块代码", "行业代码", "code"] if name in boards), None
        )
        if name_column is None:
            raise ValueError(f"Unrecognized industry-board columns: {list(boards.columns)}")

        records: list[dict[str, str]] = []
        found: set[str] = set()
        for _, board in boards.iterrows():
            industry_name = str(board[name_column]).strip()
            industry_code = (
                str(board[board_code_column]).strip()
                if board_code_column is not None
                else industry_name
            )
            try:
                constituents = self._ak.stock_board_industry_cons_em(symbol=industry_name)
            except Exception:  # noqa: BLE001 - different AKShare versions accept name or code
                constituents = self._ak.stock_board_industry_cons_em(symbol=industry_code)
            constituent_code_column = next(
                (name for name in ["代码", "股票代码", "证券代码", "code"] if name in constituents),
                None,
            )
            if constituent_code_column is None:
                continue
            for raw_code in constituents[constituent_code_column]:
                match = re.search(r"\d{1,6}", str(raw_code))
                if match is None:
                    continue
                code = normalize_symbol(match.group(0).zfill(6))
                if selected is not None and code not in selected:
                    continue
                records.append(
                    {
                        "code": code,
                        "industry_code": industry_code,
                        "industry_name": industry_name,
                    }
                )
                found.add(code)
            if selected is not None and found >= selected:
                break
        return pd.DataFrame(
            records, columns=["code", "industry_code", "industry_name"]
        ).drop_duplicates("code", keep="first")

    def _fetch_cninfo_industry_snapshot(self, selected: set[str]) -> pd.DataFrame:
        records: list[dict[str, str]] = []
        end_date = pd.Timestamp.today().strftime("%Y%m%d")
        for code in sorted(selected):
            try:
                history = self._ak.stock_industry_change_cninfo(
                    symbol=code,
                    start_date="19900101",
                    end_date=end_date,
                )
            except Exception as exc:  # noqa: BLE001 - retain partial coverage
                LOGGER.debug("CNINFO industry lookup failed for %s: %r", code, exc)
                continue
            if history.empty:
                continue
            date_column = next(
                (name for name in ["变更日期", "日期", "change_date"] if name in history), None
            )
            if date_column is not None:
                history = history.assign(
                    _change_date=pd.to_datetime(history[date_column], errors="coerce")
                ).sort_values("_change_date", kind="stable")
            latest = history.iloc[-1]
            industry_code_column = next(
                (name for name in ["行业编码", "分类标准编码", "industry_code"] if name in history),
                None,
            )
            industry_name_column = next(
                (
                    name
                    for name in ["行业大类", "行业门类", "行业中类", "行业次类", "行业名称"]
                    if name in history
                ),
                None,
            )
            if industry_name_column is None:
                continue
            industry_name = str(latest[industry_name_column]).strip()
            if not industry_name or industry_name.lower() == "nan":
                continue
            industry_code = (
                str(latest[industry_code_column]).strip()
                if industry_code_column is not None
                else industry_name
            )
            records.append(
                {"code": code, "industry_code": industry_code, "industry_name": industry_name}
            )
        return pd.DataFrame(records, columns=["code", "industry_code", "industry_name"])

    def fetch_corporate_actions(self, code: str) -> pd.DataFrame:
        dividends = self._ak.stock_history_dividend_detail(
            symbol=normalize_symbol(code), indicator="分红"
        )
        rights = self._ak.stock_history_dividend_detail(
            symbol=normalize_symbol(code), indicator="配股"
        )
        return merge_corporate_actions(
            normalize_dividend_actions(dividends, code),
            normalize_rights_actions(rights, code),
        )

    def fetch_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not re.fullmatch(r"\d{6}", index_code):
            raise ValueError(f"Invalid index code: {index_code}")
        try:
            raw = self._ak.index_zh_a_hist(
                symbol=index_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
            )
            if raw.empty:
                raise ValueError("Eastmoney index endpoint returned no rows")
            result = _normalize_history(raw)
            result.attrs["source_endpoint"] = "eastmoney_index_zh_a_hist"
            return result
        except Exception as primary_error:  # noqa: BLE001 - fallback on any endpoint failure
            try:
                raw = self._ak.stock_zh_index_daily_tx(
                    symbol=_index_market_symbol(index_code),
                    start_date=start_date,
                    end_date=end_date,
                )
                if raw.empty:
                    raise ValueError("Tencent index endpoint returned no rows")
                # Tencent labels index turnover volume as amount. Only OHLC is used for
                # benchmark returns, but keep the canonical history schema explicit.
                if "volume" not in raw and "amount" in raw:
                    raw["volume"] = raw["amount"]
                    raw["amount"] = 0.0
                result = _normalize_history(raw)
                result.attrs["source_endpoint"] = "tencent_stock_zh_index_daily_tx"
                result.attrs["fallback_reason"] = repr(primary_error)
                return result
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Index history endpoints failed for {index_code}: "
                    f"eastmoney={primary_error!r}; tencent={fallback_error!r}"
                ) from fallback_error
