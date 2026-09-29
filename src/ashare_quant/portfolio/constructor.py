from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ashare_quant.config import PortfolioConfig


def _size_constraints_active(candidates: pd.DataFrame) -> bool:
    if "size_bucket" not in candidates:
        return False
    buckets = set(candidates["size_bucket"].dropna().astype(str).str.upper())
    return {"LARGE", "MID"}.issubset(buckets)


def _max_mid_positions(config: PortfolioConfig) -> int:
    return math.ceil(config.top_k * config.max_mid_position_fraction)


def _industry_constraints_active(candidates: pd.DataFrame) -> bool:
    if "industry" not in candidates:
        return False
    industries = candidates["industry"].fillna("UNKNOWN").astype(str)
    if not industries.ne("UNKNOWN").any():
        return False
    if "industry_is_point_in_time" in candidates:
        return bool(candidates["industry_is_point_in_time"].fillna(False).all())
    return True


def validate_portfolio_universe(candidates: pd.DataFrame, config: PortfolioConfig) -> None:
    """Fail before model training when the requested portfolio cannot be constructed."""
    if candidates.empty:
        raise ValueError("Portfolio universe is empty")
    groups = candidates.groupby("date", sort=True) if "date" in candidates else [(None, candidates)]
    max_per_industry = max(1, math.floor(config.top_k * config.max_industry_weight))
    for date, group in groups:
        eligible = group.loc[group["tradable"]] if "tradable" in group else group
        label = str(pd.Timestamp(date).date()) if date is not None else "input"
        if len(eligible) < config.top_k:
            raise ValueError(
                f"Portfolio preflight failed on {label}: only {len(eligible)} tradable candidates "
                f"for top_k={config.top_k}."
            )
        if len(eligible) <= config.exit_rank:
            raise ValueError(
                f"Portfolio preflight failed on {label}: exit_rank={config.exit_rank} must be below "
                f"the {len(eligible)}-stock candidate pool so holdings can leave the buffer."
            )
        target_exposure = 1.0 - config.cash_buffer
        if config.top_k * config.max_stock_weight < target_exposure - 1e-9:
            raise ValueError(
                "Portfolio preflight failed: stock caps cannot reach the configured target exposure."
            )
        industries = eligible.get("industry", pd.Series("UNKNOWN", index=eligible.index))
        capacity = int(industries.astype(str).value_counts().clip(upper=max_per_industry).sum())
        if _industry_constraints_active(eligible) and capacity < config.top_k:
            unknown_count = int((industries.astype(str) == "UNKNOWN").sum())
            raise ValueError(
                f"Portfolio preflight failed on {label}: industry caps can fill only {capacity} of "
                f"{config.top_k} positions ({unknown_count} candidates have UNKNOWN industry)."
            )
        if _size_constraints_active(eligible):
            large_count = int(
                eligible["size_bucket"].astype(str).str.upper().eq("LARGE").sum()
            )
            required_large = config.top_k - _max_mid_positions(config)
            if large_count < required_large:
                raise ValueError(
                    f"Portfolio preflight failed on {label}: size diversification requires "
                    f"{required_large} LARGE candidates but only {large_count} are available."
                )


def select_portfolio(
    ranked: pd.DataFrame,
    current_holdings: set[str],
    config: PortfolioConfig,
) -> pd.DataFrame:
    """Select a diversified portfolio with a rank buffer to reduce turnover."""
    candidates = ranked.sort_values(["score", "code"], ascending=[False, True], kind="stable").copy()
    candidates["rank"] = range(1, len(candidates) + 1)
    candidates = candidates.loc[candidates["tradable"]]

    if config.entry_rank < config.top_k:
        raise ValueError("entry_rank must be greater than or equal to top_k")
    if config.exit_rank < config.entry_rank:
        raise ValueError("exit_rank must be greater than or equal to entry_rank")
    if len(candidates) < config.top_k:
        raise ValueError(
            f"Only {len(candidates)} tradable candidates are available for top_k={config.top_k}; "
            "refusing to run a portfolio that cannot perform the configured stock selection."
        )
    if current_holdings and len(candidates) <= config.exit_rank:
        raise ValueError(
            f"exit_rank={config.exit_rank} is not below the {len(candidates)}-stock candidate pool; "
            "existing holdings could never leave the rank buffer."
        )
    target_exposure = 1.0 - config.cash_buffer
    if config.top_k * config.max_stock_weight < target_exposure - 1e-9:
        raise ValueError(
            "top_k * max_stock_weight is below the configured target exposure"
        )

    size_constraints_active = _size_constraints_active(candidates)
    max_mid_positions = _max_mid_positions(config)
    if config.max_replacements_per_rebalance is not None:
        if config.max_replacements_per_rebalance < 0:
            raise ValueError("max_replacements_per_rebalance must be non-negative")
        if config.min_replacement_rank_improvement < 0:
            raise ValueError("min_replacement_rank_improvement must be non-negative")

    eligible_holdings = candidates.loc[candidates["code"].isin(current_holdings)].copy()
    eligible_holdings = eligible_holdings.sort_values("rank", kind="stable").head(config.top_k)
    preferred_codes = eligible_holdings["code"].tolist()

    # Missing or non-tradable holdings create unavoidable vacancies and are replaced first.
    entry_pool = candidates.loc[
        ~candidates["code"].isin(set(preferred_codes))
        & (candidates["rank"] <= config.entry_rank)
    ].sort_values("rank", kind="stable")
    vacancies = max(config.top_k - len(preferred_codes), 0)
    vacancy_entries = entry_pool.head(vacancies)
    preferred_codes.extend(vacancy_entries["code"].tolist())

    # Once the portfolio is full, replacements are deliberately scarce.  A holding must be
    # outside the exit buffer and the incoming stock must beat it by a meaningful rank gap.
    replacement_limit = config.max_replacements_per_rebalance
    if replacement_limit is None:
        replacement_limit = config.top_k
    remaining_entries = entry_pool.loc[~entry_pool["code"].isin(preferred_codes)]
    replacement_count = 0
    for incoming in remaining_entries.itertuples(index=False):
        if replacement_count >= replacement_limit or not preferred_codes:
            break
        preferred = candidates.loc[candidates["code"].isin(preferred_codes)]
        worst = preferred.sort_values("rank", ascending=False, kind="stable").iloc[0]
        if int(worst["rank"]) <= config.exit_rank:
            break
        rank_improvement = int(worst["rank"]) - int(incoming.rank)
        if rank_improvement < config.min_replacement_rank_improvement:
            continue
        preferred_codes.remove(str(worst["code"]))
        preferred_codes.append(str(incoming.code))
        replacement_count += 1

    preferred = candidates.loc[candidates["code"].isin(preferred_codes)].copy()
    preferred["_retention_priority"] = ~preferred["code"].isin(current_holdings)
    preferred = preferred.sort_values(
        ["_retention_priority", "rank"], kind="stable"
    ).drop(columns="_retention_priority")
    fallback = candidates.loc[~candidates["code"].isin(set(preferred_codes))]
    ordered = pd.concat([preferred, fallback], ignore_index=True)
    industry_constraints_active = _industry_constraints_active(candidates)
    max_per_industry = (
        max(1, math.floor(config.top_k * config.max_industry_weight))
        if industry_constraints_active
        else config.top_k
    )
    industry_counts: dict[str, int] = {}
    mid_positions = 0
    selected_rows = []
    selected_codes: set[str] = set()

    for row in ordered.itertuples(index=False):
        if row.code in selected_codes:
            continue
        industry = str(row.industry)
        if industry_counts.get(industry, 0) >= max_per_industry:
            continue
        size_bucket = str(getattr(row, "size_bucket", "UNKNOWN")).upper()
        if size_constraints_active and size_bucket == "MID" and mid_positions >= max_mid_positions:
            continue
        selected_rows.append(row._asdict())
        selected_codes.add(row.code)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if size_bucket == "MID":
            mid_positions += 1
        if len(selected_rows) >= config.top_k:
            break

    if not selected_rows:
        return candidates.iloc[0:0].assign(target_weight=pd.Series(dtype=float))

    if len(selected_rows) < config.top_k:
        unknown_count = int((candidates["industry"].astype(str) == "UNKNOWN").sum())
        raise ValueError(
            f"Industry constraints selected only {len(selected_rows)} of {config.top_k} required stocks "
            f"({unknown_count} candidates have UNKNOWN industry). Provide valid industry data or revise "
            "the documented concentration policy."
        )

    selected = pd.DataFrame(selected_rows)
    if size_constraints_active:
        required_large = config.top_k - max_mid_positions
        selected_large = int(
            selected["size_bucket"].astype(str).str.upper().eq("LARGE").sum()
        )
        if selected_large < required_large:
            raise ValueError(
                f"Size constraints selected only {selected_large} LARGE stocks; "
                f"at least {required_large} are required for top_k={config.top_k}."
            )
    if config.weighting_method == "equal":
        desired = np.ones(len(selected), dtype=float)
    elif config.weighting_method == "inverse_volatility":
        if "idio_vol_60" not in selected:
            raise ValueError("inverse_volatility weighting requires idio_vol_60")
        volatility = pd.to_numeric(selected["idio_vol_60"], errors="coerce")
        floor = max(float(volatility[volatility > 0].quantile(0.1)), 1e-6)
        desired = 1.0 / volatility.clip(lower=floor).fillna(volatility.median()).to_numpy()
    elif config.weighting_method == "score_tilt":
        score = pd.to_numeric(selected["score"], errors="coerce").fillna(0.0)
        standardized = (score - score.mean()) / (score.std(ddof=0) + 1e-12)
        desired = np.exp(standardized.clip(-2.0, 2.0).to_numpy())
    else:
        raise ValueError(f"Unknown weighting method: {config.weighting_method}")
    selected["target_weight"] = _allocate_constrained_weights(
        selected["industry"].astype(str),
        desired,
        target_exposure=target_exposure,
        max_stock_weight=config.max_stock_weight,
        max_industry_weight=(
            config.max_industry_weight if industry_constraints_active else target_exposure
        ),
        size_buckets=(
            selected["size_bucket"].astype(str).str.upper()
            if size_constraints_active
            else None
        ),
        max_mid_cap_weight=config.max_mid_cap_weight,
    )
    return selected


def _allocate_constrained_weights(
    industries: pd.Series,
    desired: np.ndarray,
    *,
    target_exposure: float,
    max_stock_weight: float,
    max_industry_weight: float,
    size_buckets: pd.Series | None = None,
    max_mid_cap_weight: float = 1.0,
) -> np.ndarray:
    """Water-fill desired weights under per-stock and per-industry exposure caps."""
    desired = np.asarray(desired, dtype=float)
    desired = np.where(np.isfinite(desired) & (desired > 0), desired, 1.0)
    weights = np.zeros(len(desired), dtype=float)
    industry_values = industries.to_numpy()
    size_values = size_buckets.to_numpy() if size_buckets is not None else None
    for _ in range(100):
        deficit = target_exposure - float(weights.sum())
        if deficit <= 1e-10:
            break
        stock_room = np.maximum(max_stock_weight - weights, 0.0)
        industry_room = {
            industry: max(
                max_industry_weight - float(weights[industry_values == industry].sum()), 0.0
            )
            for industry in np.unique(industry_values)
        }
        eligible = np.array(
            [stock_room[index] > 1e-12 and industry_room[industry] > 1e-12
             for index, industry in enumerate(industry_values)]
        )
        if size_values is not None:
            mid_room = max(
                max_mid_cap_weight - float(weights[size_values == "MID"].sum()), 0.0
            )
            if mid_room <= 1e-12:
                eligible &= size_values != "MID"
        if not eligible.any():
            break
        proposal = np.zeros(len(desired), dtype=float)
        proposal[eligible] = deficit * desired[eligible] / desired[eligible].sum()
        proposal = np.minimum(proposal, stock_room)
        for industry, room in industry_room.items():
            mask = industry_values == industry
            total = float(proposal[mask].sum())
            if total > room and total > 0:
                proposal[mask] *= room / total
        if size_values is not None:
            mid_mask = size_values == "MID"
            mid_room = max(
                max_mid_cap_weight - float(weights[mid_mask].sum()), 0.0
            )
            proposed_mid = float(proposal[mid_mask].sum())
            if proposed_mid > mid_room and proposed_mid > 0:
                proposal[mid_mask] *= mid_room / proposed_mid
        if proposal.sum() <= 1e-12:
            break
        weights += proposal
    if target_exposure - weights.sum() > 1e-8:
        raise ValueError("Unable to allocate target exposure under stock and industry caps")
    return weights


def validate_research_universe(candidates: pd.DataFrame, minimum: int) -> None:
    """Require a genuinely cross-sectional universe on every model date."""
    counts = candidates.loc[candidates["tradable"]].groupby("date")["code"].nunique()
    if counts.empty or int(counts.min()) < minimum:
        smallest = int(counts.min()) if not counts.empty else 0
        raise ValueError(
            f"Research universe has only {smallest} tradable stocks on its smallest date; "
            f"at least {minimum} are required for a formal cross-sectional run."
        )
