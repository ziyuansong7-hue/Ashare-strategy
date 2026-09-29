from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.config import CostConfig, PortfolioConfig
from ashare_quant.portfolio.constructor import select_portfolio


def load_positions(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["code", "quantity"])
    positions = pd.read_csv(path, dtype={"code": "string"})
    required = {"code", "quantity"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"Positions file is missing columns: {sorted(missing)}")
    positions["code"] = positions["code"].astype(str).str.zfill(6)
    positions["quantity"] = pd.to_numeric(positions["quantity"], errors="raise").astype(int)
    if (positions["quantity"] < 0).any() or positions["code"].duplicated().any():
        raise ValueError("Positions must contain unique codes and non-negative quantities")
    return positions.loc[positions["quantity"] > 0, ["code", "quantity"]]


def _estimated_fees(notional: float, side: str, costs: CostConfig) -> float:
    commission = max(costs.minimum_commission, notional * costs.commission_bps / 10_000.0)
    transfer = notional * costs.transfer_bps / 10_000.0
    tax = notional * costs.sell_tax_bps / 10_000.0 if side == "SELL" else 0.0
    return commission + transfer + tax


def plan_orders(
    scores: pd.DataFrame,
    bars: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    cash: float,
    portfolio_config: PortfolioConfig,
    cost_config: CostConfig,
    score_column: str = "xgboost_score",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create an indicative next-session order plan from the latest research scores."""
    if cash < 0:
        raise ValueError("cash must be non-negative")
    if score_column not in scores:
        raise ValueError(f"Scores are missing {score_column!r}")
    signal_date = pd.Timestamp(pd.to_datetime(scores["date"]).max()).normalize()
    latest_scores = scores.loc[pd.to_datetime(scores["date"]).dt.normalize() == signal_date].copy()
    market = bars.loc[pd.to_datetime(bars["date"]) <= signal_date].copy()
    market = market.sort_values(["code", "date"], kind="stable").drop_duplicates("code", keep="last")
    price_column = "raw_close" if "raw_close" in market else "close"
    market["reference_price"] = pd.to_numeric(market[price_column], errors="coerce")
    needed_market = [
        column
        for column in [
            "code", "date", "reference_price", "is_trading", "is_st", "lot_size",
            "industry", "idio_vol_60",
            "size_bucket",
        ]
        if column in market
    ]
    candidates = latest_scores.merge(
        market[needed_market], on="code", how="left", suffixes=("", "_market")
    )
    if "industry_market" in candidates:
        candidates["industry"] = candidates.get("industry", candidates["industry_market"]).fillna(
            candidates["industry_market"]
        )
    candidates["score"] = pd.to_numeric(candidates[score_column], errors="coerce")
    default_true = pd.Series(True, index=candidates.index)
    default_false = pd.Series(False, index=candidates.index)
    candidates["tradable"] = (
        candidates.get("tradable", default_true).fillna(False).astype(bool)
        & candidates.get("is_trading", default_true).fillna(False).astype(bool)
        & ~candidates.get("is_st", default_false).fillna(False).astype(bool)
        & candidates["reference_price"].gt(0)
    )
    current = dict(zip(positions["code"], positions["quantity"], strict=False))
    selected = select_portfolio(candidates, set(current), portfolio_config)
    target_weights = dict(zip(selected["code"], selected["target_weight"], strict=False))

    price_map = market.set_index("code")["reference_price"].to_dict()
    equity = float(cash) + sum(
        quantity * float(price_map.get(code, np.nan))
        for code, quantity in current.items()
        if pd.notna(price_map.get(code, np.nan))
    )
    score_lookup = latest_scores.set_index("code")
    model_prefix = score_column.removesuffix("_score")
    rows: list[dict[str, object]] = []
    for code in sorted(set(current) | set(target_weights)):
        price = float(price_map.get(code, np.nan))
        if not np.isfinite(price) or price <= 0:
            rows.append(
                {
                    "signal_date": signal_date,
                    "code": code,
                    "action": "REVIEW",
                    "reason": "NO_CURRENT_REFERENCE_PRICE",
                    "current_quantity": current.get(code, 0),
                }
            )
            continue
        market_row = market.loc[market["code"] == code].iloc[0]
        lot = int(market_row.get("lot_size", portfolio_config.default_lot_size))
        target_weight = float(target_weights.get(code, 0.0))
        current_quantity = current.get(code, 0)
        current_weight = current_quantity * price / equity if equity else 0.0
        if (
            target_weight > 0
            and current_quantity > 0
            and abs(current_weight - target_weight)
            <= portfolio_config.rebalance_weight_tolerance
        ):
            target_quantity = current_quantity
            reason = "WITHIN_REBALANCE_BAND"
        else:
            target_quantity = int(equity * target_weight / price) // lot * lot
            reason = "TARGET_REBALANCE"
        delta = target_quantity - current_quantity
        action = "BUY" if delta > 0 else "SELL" if delta < 0 else "HOLD"
        notional = abs(delta) * price
        if action != "HOLD" and notional < portfolio_config.min_trade_value and target_quantity != 0:
            action = "HOLD"
            reason = "BELOW_MIN_TRADE_VALUE"
            delta = 0
            notional = 0.0
        fees = _estimated_fees(notional, action, cost_config) if action in {"BUY", "SELL"} else 0.0
        score_row = score_lookup.loc[code] if code in score_lookup.index else pd.Series(dtype=object)
        rows.append(
            {
                "signal_date": signal_date,
                "execution_instruction": "NEXT_TRADING_DAY_OPEN_INDICATIVE",
                "code": code,
                "security_name": score_row.get("security_name", ""),
                "industry": market_row.get("industry", score_row.get("industry", "UNKNOWN")),
                "size_bucket": market_row.get(
                    "size_bucket", score_row.get("size_bucket", "UNKNOWN")
                ),
                "action": action,
                "reason": reason,
                "current_quantity": current.get(code, 0),
                "target_quantity": target_quantity,
                "order_quantity": abs(delta),
                "reference_price": price,
                "estimated_notional": notional,
                "estimated_fees": fees,
                "target_weight": target_weight,
                "model_score": score_row.get(score_column, np.nan),
                "factor_reason_1": score_row.get(
                    f"{model_prefix}_reason_1", score_row.get("factor_reason_1", "")
                ),
                "factor_reason_2": score_row.get(
                    f"{model_prefix}_reason_2", score_row.get("factor_reason_2", "")
                ),
            }
        )
    orders = pd.DataFrame(rows)
    if not orders.empty:
        priority = pd.Categorical(orders["action"], ["SELL", "BUY", "HOLD", "REVIEW"], ordered=True)
        orders = orders.assign(_priority=priority).sort_values(
            ["_priority", "estimated_notional"], ascending=[True, False], na_position="last"
        ).drop(columns="_priority")
    summary = {
        "signal_date": str(signal_date.date()),
        "execution_instruction": "Use next trading day's open only after rechecking suspension/limit status.",
        "cash_before": float(cash),
        "estimated_equity": equity,
        "target_positions": len(target_weights),
        "target_exposure": float(sum(target_weights.values())),
        "estimated_buy_notional": float(orders.loc[orders["action"] == "BUY", "estimated_notional"].sum()),
        "estimated_sell_notional": float(orders.loc[orders["action"] == "SELL", "estimated_notional"].sum()),
        "estimated_fees": float(orders.get("estimated_fees", pd.Series(dtype=float)).sum()),
        "warning": "Indicative orders only; prices, limits, suspensions, cash and fees must be rechecked at execution.",
    }
    return orders.reset_index(drop=True), summary
