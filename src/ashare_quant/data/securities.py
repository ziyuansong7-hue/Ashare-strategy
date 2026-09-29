from __future__ import annotations

import re

import pandas as pd

from .schema import normalize_symbol

UNIVERSE_INDEX_CODES = {
    "csi300": "000300",
    "csi500": "000905",
    "csi800": "000906",
    "csi1000": "000852",
}

UNIVERSE_SIZE_BUCKETS = {
    "csi300": "LARGE",
    "csi500": "MID",
    "csi1000": "SMALL",
}

BENCHMARKS = {
    "csi800": {"index_code": "000906", "index_name": "中证800"},
    "csi300": {"index_code": "000300", "index_name": "沪深300"},
    "csi500": {"index_code": "000905", "index_name": "中证500"},
    "sse_composite": {"index_code": "000001", "index_name": "上证综指"},
    "szse_component": {"index_code": "399001", "index_name": "深证成指"},
}


def exchange_from_code(code: str) -> str:
    symbol = normalize_symbol(code)
    if symbol.startswith(("4", "8", "92")):
        return "BSE"
    if symbol.startswith(("0", "3")):
        return "SZSE"
    if symbol.startswith("6"):
        return "SSE"
    return "UNKNOWN"


def board_from_code(code: str) -> str:
    symbol = normalize_symbol(code)
    if symbol.startswith(("300", "301")):
        return "CHINEXT"
    if symbol.startswith(("688", "689")):
        return "STAR"
    if exchange_from_code(symbol) == "BSE":
        return "BSE"
    if exchange_from_code(symbol) in {"SSE", "SZSE"}:
        return "MAIN"
    return "UNKNOWN"


def canonical_security_master(raw: pd.DataFrame, *, source: str) -> pd.DataFrame:
    required = {"code", "security_name"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Security master is missing columns: {sorted(missing)}")
    master = raw.copy()
    master["code"] = master["code"].map(normalize_symbol)
    master["security_name"] = master["security_name"].astype(str).str.strip()
    master = master.loc[master["code"].str.fullmatch(r"\d{6}", na=False)].copy()
    master["exchange"] = master["code"].map(exchange_from_code)
    master["board"] = master["code"].map(board_from_code)
    # The upstream endpoint is intended for A shares, but a prefix-level guard prevents
    # common fund/index namespaces from silently entering an equity universe.
    master = master.loc[master["exchange"] != "UNKNOWN"].copy()
    master["instrument_type"] = "STOCK"
    master["is_st_current"] = master["security_name"].str.match(r"^(\*?ST|S\*?ST)", flags=re.IGNORECASE)
    master["source"] = source
    master["as_of_date"] = pd.Timestamp.today().normalize()
    return master[
        [
            "code",
            "security_name",
            "exchange",
            "board",
            "instrument_type",
            "is_st_current",
            "source",
            "as_of_date",
        ]
    ].drop_duplicates("code", keep="last").sort_values("code").reset_index(drop=True)


def load_point_in_time_membership(path: str) -> pd.DataFrame:
    membership = pd.read_csv(path, dtype={"code": "string"})
    required = {"code", "effective_from", "effective_to"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(
            "Point-in-time membership file must contain code, effective_from, effective_to; "
            f"missing={sorted(missing)}"
        )
    membership["code"] = membership["code"].map(normalize_symbol)
    membership["effective_from"] = pd.to_datetime(membership["effective_from"], errors="raise").dt.normalize()
    membership["effective_to"] = pd.to_datetime(membership["effective_to"], errors="coerce").dt.normalize()
    if (~membership["code"].str.fullmatch(r"\d{6}", na=False)).any():
        raise ValueError("Membership file contains invalid six-character stock codes")
    if (
        membership["effective_to"].notna()
        & (membership["effective_to"] < membership["effective_from"])
    ).any():
        raise ValueError("Membership effective_to must not be earlier than effective_from")
    membership["membership_mode"] = "point_in_time"
    if "size_bucket" not in membership:
        membership["size_bucket"] = "UNKNOWN"
    membership["size_bucket"] = membership["size_bucket"].fillna("UNKNOWN").astype(str).str.upper()
    allowed_buckets = {"LARGE", "MID", "SMALL", "UNKNOWN"}
    invalid_buckets = sorted(set(membership["size_bucket"]).difference(allowed_buckets))
    if invalid_buckets:
        raise ValueError(f"Membership file contains invalid size_bucket values: {invalid_buckets}")
    return membership.sort_values(["effective_from", "code"]).reset_index(drop=True)


def compress_membership_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Compress monthly point-in-time snapshots into non-overlapping effective intervals."""
    required = {"snapshot_date", "code", "size_bucket"}
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"Membership snapshots are missing columns: {sorted(missing)}")
    data = snapshots.copy()
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="raise").dt.normalize()
    data["code"] = data["code"].map(normalize_symbol)
    data["size_bucket"] = data["size_bucket"].astype(str).str.upper()
    data["_bucket_order"] = data["size_bucket"].map(
        {"LARGE": 0, "MID": 1, "SMALL": 2, "UNKNOWN": 3}
    ).fillna(4)
    data = (
        data.sort_values(["snapshot_date", "code", "_bucket_order"], kind="stable")
        .drop_duplicates(["snapshot_date", "code"], keep="first")
        .drop(columns="_bucket_order")
    )
    snapshot_dates = pd.DatetimeIndex(sorted(data["snapshot_date"].unique()))
    index_by_date = {date: index for index, date in enumerate(snapshot_dates)}
    data["_snapshot_index"] = data["snapshot_date"].map(index_by_date).astype(int)
    data = data.sort_values(["code", "_snapshot_index"], kind="stable")
    previous_index = data.groupby("code")["_snapshot_index"].shift()
    previous_bucket = data.groupby("code")["size_bucket"].shift()
    new_run = (
        previous_index.isna()
        | data["_snapshot_index"].ne(previous_index + 1)
        | data["size_bucket"].ne(previous_bucket)
    )
    data["_run"] = new_run.groupby(data["code"]).cumsum()
    aggregations: dict[str, tuple[str, str]] = {
        "effective_from": ("snapshot_date", "min"),
        "_last_snapshot_index": ("_snapshot_index", "max"),
        "size_bucket": ("size_bucket", "last"),
    }
    if "security_name" in data:
        aggregations["security_name"] = ("security_name", "last")
    intervals = data.groupby(["code", "_run"], as_index=False).agg(**aggregations)

    def effective_to(last_index: int) -> pd.Timestamp | pd.NaT:
        next_index = int(last_index) + 1
        if next_index >= len(snapshot_dates):
            return pd.NaT
        return pd.Timestamp(snapshot_dates[next_index]) - pd.Timedelta(days=1)

    intervals["effective_to"] = intervals["_last_snapshot_index"].map(effective_to)
    intervals["snapshot_date"] = intervals["effective_from"]
    intervals["universe"] = "csi800"
    intervals["membership_mode"] = "point_in_time"
    intervals["membership_source"] = "baostock_month_end_history"
    return intervals.drop(columns=["_run", "_last_snapshot_index"]).sort_values(
        ["effective_from", "code"], kind="stable"
    ).reset_index(drop=True)
