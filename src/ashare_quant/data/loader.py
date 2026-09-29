from __future__ import annotations

from pathlib import Path

import pandas as pd

from .enrichment import apply_industry_history, empty_corporate_actions
from .schema import DataQualityReport, normalize_and_validate_bars


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"code": "string"})
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported market-data file: {path}")


def _apply_point_in_time_membership(
    bars: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    if "code" not in membership:
        raise ValueError("Universe membership is missing the code column")
    eligible_codes = set(membership["code"].astype(str).str.zfill(6))
    eligible = bars.loc[bars["code"].astype(str).str.zfill(6).isin(eligible_codes)].copy()
    eligible["code"] = eligible["code"].astype(str).str.zfill(6)
    membership = membership.copy()
    membership["code"] = membership["code"].astype(str).str.zfill(6)
    if "size_bucket" in membership:
        membership["size_bucket"] = (
            membership["size_bucket"].fillna("UNKNOWN").astype(str).str.upper()
        )
    if "membership_mode" not in membership:
        if "size_bucket" not in membership:
            return eligible
        return eligible.drop(columns="size_bucket", errors="ignore").merge(
            membership[["code", "size_bucket"]].drop_duplicates("code", keep="last"),
            on="code",
            how="left",
        )
    point_in_time = membership.loc[membership["membership_mode"] == "point_in_time"].copy()
    if point_in_time.empty:
        # A current snapshot has no historical effective dates, but it must still constrain the
        # files that enter the model. This prevents stale files from a previous universe sync from
        # silently contaminating a smoke test.
        if "size_bucket" not in membership:
            return eligible
        return eligible.drop(columns="size_bucket", errors="ignore").merge(
            membership[["code", "size_bucket"]].drop_duplicates("code", keep="last"),
            on="code",
            how="left",
        )
    required = {"code", "effective_from", "effective_to"}
    missing = required.difference(point_in_time.columns)
    if missing:
        raise ValueError(f"Point-in-time membership is missing columns: {sorted(missing)}")

    point_in_time["effective_from"] = pd.to_datetime(point_in_time["effective_from"]).dt.normalize()
    point_in_time["effective_to"] = pd.to_datetime(
        point_in_time["effective_to"], errors="coerce"
    ).dt.normalize()
    dated = eligible.drop(columns="size_bucket", errors="ignore").copy()
    dated["code"] = dated["code"].astype(str).str.zfill(6)
    dated["date"] = pd.to_datetime(dated["date"]).dt.normalize()
    dated["_input_order"] = range(len(dated))
    membership_columns = ["code", "effective_from", "effective_to"]
    if "size_bucket" in point_in_time:
        membership_columns.append("size_bucket")
    joined = dated.merge(
        point_in_time[membership_columns],
        on="code",
        how="inner",
    )
    active = (joined["date"] >= joined["effective_from"]) & (
        joined["effective_to"].isna() | (joined["date"] <= joined["effective_to"])
    )
    active_rows = (
        joined.loc[active]
        .sort_values(["_input_order", "effective_from"], kind="stable")
        .drop_duplicates("_input_order", keep="last")
        .set_index("_input_order")
    )
    # Keep pre-membership price history so newly admitted stocks retain causal factor lookback.
    # Membership controls signal eligibility, not whether historical bars may exist.
    dated["is_universe_member"] = dated["_input_order"].isin(active_rows.index)
    if "size_bucket" in active_rows:
        dated["size_bucket"] = (
            dated["_input_order"].map(active_rows["size_bucket"]).fillna("UNKNOWN")
        )
    return dated.drop(columns="_input_order").reset_index(drop=True)


def load_bars(
    path: str | Path,
    *,
    default_lot_size: int = 100,
) -> tuple[pd.DataFrame, DataQualityReport]:
    source = Path(path)
    membership_path: Path | None = None
    industry_path: Path | None = None
    if source.is_dir():
        # A synchronized market-data root also contains benchmarks and metadata.
        # Only the per-symbol bars belong in the model input table.
        if (source / "bars").is_dir():
            candidate = source / "universe_membership.parquet"
            if candidate.exists():
                membership_path = candidate
            industry_candidate = source / "industry_history.parquet"
            if industry_candidate.exists():
                industry_path = industry_candidate
            source = source / "bars"
        files = sorted([*source.glob("*.csv"), *source.glob("*.parquet"), *source.glob("*.pq")])
        if not files:
            raise FileNotFoundError(f"No CSV or Parquet files found under {source}")
        frames = []
        for file in files:
            current = _read_one(file)
            if "code" not in current and file.stem.isdigit():
                current["code"] = file.stem.zfill(6)
            frames.append(current)
        raw = pd.concat(frames, ignore_index=True)
    elif source.is_file():
        raw = _read_one(source)
    else:
        raise FileNotFoundError(source)

    if membership_path is not None:
        raw = _apply_point_in_time_membership(raw, pd.read_parquet(membership_path))
    if industry_path is not None:
        raw = apply_industry_history(raw, pd.read_parquet(industry_path))

    return normalize_and_validate_bars(raw, default_lot_size=default_lot_size)


def load_corporate_actions(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir() and (source / "corporate_actions.parquet").exists():
        source = source / "corporate_actions.parquet"
    if not source.exists():
        return empty_corporate_actions()
    actions = pd.read_parquet(source)
    required = {
        "code",
        "ex_date",
        "cash_dividend_per_share",
        "bonus_share_ratio",
        "rights_share_ratio",
        "rights_price",
    }
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"Corporate actions are missing columns: {sorted(missing)}")
    actions["code"] = actions["code"].astype(str).str.zfill(6)
    actions["ex_date"] = pd.to_datetime(actions["ex_date"], errors="raise").dt.normalize()
    return actions.sort_values(["ex_date", "code"], kind="stable").reset_index(drop=True)
