from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_quant.data.schema import normalize_symbol

ACTION_COLUMNS = [
    "code",
    "ex_date",
    "cash_dividend_per_share",
    "bonus_share_ratio",
    "rights_share_ratio",
    "rights_price",
    "source",
]


def normalize_industry_history(raw: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "股票代码": "code",
        "symbol": "code",
        "纳入时间": "effective_from",
        "start_date": "effective_from",
        "行业代码": "industry_code",
        "industry_code": "industry_code",
        "行业名称": "industry_name",
        "行业名称一级": "industry_name",
        "industry_name": "industry_name",
    }
    data = raw.rename(columns={column: aliases.get(column, column) for column in raw.columns}).copy()
    required = {"code", "effective_from", "industry_code"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Industry history is missing columns: {sorted(missing)}")
    data["code"] = data["code"].map(normalize_symbol)
    data["effective_from"] = pd.to_datetime(data["effective_from"], errors="coerce").dt.normalize()
    data["industry_code"] = data["industry_code"].astype(str).str.strip()
    if "industry_name" not in data:
        data["industry_name"] = data["industry_code"]
    data["industry_name"] = data["industry_name"].fillna(data["industry_code"]).astype(str).str.strip()
    data = data.dropna(subset=["effective_from"]).drop_duplicates(
        ["code", "effective_from"], keep="last"
    )
    data = data.sort_values(["code", "effective_from"], kind="stable")
    next_start = data.groupby("code")["effective_from"].shift(-1)
    data["effective_to"] = next_start - pd.Timedelta(days=1)
    if "industry_standard" not in data:
        data["industry_standard"] = "SW"
    data["industry_standard"] = data["industry_standard"].fillna("UNKNOWN").astype(str)
    if "is_point_in_time" not in data:
        data["is_point_in_time"] = True
    else:
        point_in_time = data["is_point_in_time"]
        if not pd.api.types.is_bool_dtype(point_in_time):
            point_in_time = point_in_time.astype(str).str.strip().str.lower().map(
                {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
            )
        data["is_point_in_time"] = point_in_time.fillna(False).astype(bool)
    return data[
        [
            "code",
            "effective_from",
            "effective_to",
            "industry_code",
            "industry_name",
            "industry_standard",
            "is_point_in_time",
        ]
    ].reset_index(drop=True)


def apply_industry_history(bars: pd.DataFrame, industry_history: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or industry_history.empty:
        return bars
    left = bars.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    left["code"] = left["code"].map(normalize_symbol)
    left["_input_order"] = range(len(left))
    history = industry_history.copy()
    history["code"] = history["code"].map(normalize_symbol)
    history["effective_from"] = pd.to_datetime(history["effective_from"]).dt.normalize()
    history["effective_to"] = pd.to_datetime(history["effective_to"], errors="coerce").dt.normalize()
    joined = left.merge(
        history[
            [
                "code",
                "effective_from",
                "effective_to",
                "industry_name",
                "industry_code",
                "is_point_in_time",
            ]
        ],
        on="code",
        how="left",
    )
    active = (joined["date"] >= joined["effective_from"]) & (
        joined["effective_to"].isna() | (joined["date"] <= joined["effective_to"])
    )
    active_rows = joined.loc[active].sort_values("effective_from").drop_duplicates(
        ["_input_order"], keep="last"
    )
    industry_by_order = active_rows.set_index("_input_order")
    left["industry"] = left["_input_order"].map(industry_by_order["industry_name"])
    left["industry_code"] = left["_input_order"].map(industry_by_order["industry_code"])
    left["industry_is_point_in_time"] = (
        left["_input_order"].map(industry_by_order["is_point_in_time"]).fillna(False).astype(bool)
    )
    left["industry"] = left["industry"].fillna("UNKNOWN")
    return left.drop(columns="_input_order")


def empty_corporate_actions() -> pd.DataFrame:
    return pd.DataFrame(columns=ACTION_COLUMNS)


def normalize_dividend_actions(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    aliases = {
        "除权除息日": "ex_date",
        "送股": "bonus_per_10",
        "转增": "transfer_per_10",
        "派息": "cash_per_10",
    }
    data = raw.rename(columns={column: aliases.get(column, column) for column in raw.columns}).copy()
    if data.empty:
        return empty_corporate_actions()
    required = {"ex_date", "bonus_per_10", "transfer_per_10", "cash_per_10"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dividend response is missing columns: {sorted(missing)}")
    data["ex_date"] = pd.to_datetime(data["ex_date"], errors="coerce").dt.normalize()
    for column in ["bonus_per_10", "transfer_per_10", "cash_per_10"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    result = pd.DataFrame(
        {
            "code": normalize_symbol(code),
            "ex_date": data["ex_date"],
            "cash_dividend_per_share": data["cash_per_10"] / 10.0,
            "bonus_share_ratio": (data["bonus_per_10"] + data["transfer_per_10"]) / 10.0,
            "rights_share_ratio": 0.0,
            "rights_price": np.nan,
            "source": "akshare_sina_dividend",
        }
    )
    return result.dropna(subset=["ex_date"]).reset_index(drop=True)


def normalize_rights_actions(raw: pd.DataFrame, code: str) -> pd.DataFrame:
    aliases = {"除权日": "ex_date", "配股方案": "rights_per_10", "配股价格": "rights_price"}
    data = raw.rename(columns={column: aliases.get(column, column) for column in raw.columns}).copy()
    if data.empty:
        return empty_corporate_actions()
    required = {"ex_date", "rights_per_10", "rights_price"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Rights response is missing columns: {sorted(missing)}")
    data["ex_date"] = pd.to_datetime(data["ex_date"], errors="coerce").dt.normalize()
    data["rights_per_10"] = pd.to_numeric(data["rights_per_10"], errors="coerce").fillna(0.0)
    data["rights_price"] = pd.to_numeric(data["rights_price"], errors="coerce")
    result = pd.DataFrame(
        {
            "code": normalize_symbol(code),
            "ex_date": data["ex_date"],
            "cash_dividend_per_share": 0.0,
            "bonus_share_ratio": 0.0,
            "rights_share_ratio": data["rights_per_10"] / 10.0,
            "rights_price": data["rights_price"],
            "source": "akshare_sina_rights",
        }
    )
    return result.dropna(subset=["ex_date"]).reset_index(drop=True)


def merge_corporate_actions(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return empty_corporate_actions()
    combined = pd.concat(nonempty, ignore_index=True)
    return (
        combined.groupby(["code", "ex_date"], as_index=False)
        .agg(
            cash_dividend_per_share=("cash_dividend_per_share", "sum"),
            bonus_share_ratio=("bonus_share_ratio", "sum"),
            rights_share_ratio=("rights_share_ratio", "sum"),
            rights_price=("rights_price", "max"),
            source=("source", lambda values: "+".join(sorted(set(values)))),
        )
        .sort_values(["ex_date", "code"], kind="stable")
        .reset_index(drop=True)
    )


def add_market_cap_estimate(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    turnover_source = result.get("turnover", pd.Series(np.nan, index=result.index))
    turnover = pd.to_numeric(turnover_source, errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce")
    raw_close = pd.to_numeric(result["raw_close"], errors="coerce")
    valid = turnover.gt(0) & volume.gt(0) & raw_close.gt(0)
    result["float_shares_estimate"] = np.where(valid, volume / turnover, np.nan)
    result["market_cap"] = result["float_shares_estimate"] * raw_close
    result["market_cap_source"] = np.where(
        valid, "turnover_implied_float_market_cap", "UNAVAILABLE"
    )
    return result


def add_price_limits(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    board = result["board"].astype(str)
    is_st = result["is_st"].fillna(False).astype(bool)
    rate = pd.Series(0.10, index=result.index)
    rate.loc[is_st] = 0.05
    rate.loc[(board == "STAR") & ~is_st] = 0.20
    rate.loc[(board == "CHINEXT") & (dates >= pd.Timestamp("2020-08-24")) & ~is_st] = 0.20
    rate.loc[(board == "BSE") & ~is_st] = 0.30
    preclose = pd.to_numeric(result["preclose"], errors="coerce")
    result["limit_up"] = (preclose * (1.0 + rate)).round(2)
    result["limit_down"] = (preclose * (1.0 - rate)).round(2)
    result["limit_rate"] = rate
    result["limit_rule_is_point_in_time"] = preclose.notna() & result[
        "status_is_point_in_time"
    ].fillna(False)
    return result
