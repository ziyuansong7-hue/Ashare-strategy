from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.config import CostConfig, PortfolioConfig
from ashare_quant.portfolio import select_portfolio


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame
    rebalance_summary: pd.DataFrame
    corporate_actions: pd.DataFrame


@dataclass(frozen=True)
class BacktestMarketContext:
    market_by_date: pd.DataFrame
    benchmark_returns: pd.Series
    trading_dates: pd.DatetimeIndex


def prepare_backtest_market(bars: pd.DataFrame) -> BacktestMarketContext:
    """Build immutable market indexes once for repeated portfolio-policy experiments."""
    market = bars.sort_values(["date", "code"], kind="stable").copy()
    market["_benchmark_return"] = market.groupby("code", sort=False)["close"].pct_change(
        fill_method=None
    )
    benchmark_market = market
    if "is_universe_member" in market:
        benchmark_market = market.loc[market["is_universe_member"].fillna(False).astype(bool)]
    benchmark_returns = benchmark_market.groupby("date")["_benchmark_return"].mean().fillna(0.0)
    trading_dates = pd.DatetimeIndex(sorted(market["date"].unique()))
    market_by_date = market.set_index(["date", "code"], drop=False).sort_index()
    return BacktestMarketContext(
        market_by_date=market_by_date,
        benchmark_returns=benchmark_returns,
        trading_dates=trading_dates,
    )


def _row_price(row: pd.Series, raw_column: str, adjusted_column: str) -> float:
    raw_value = row.get(raw_column)
    if raw_value is not None and pd.notna(raw_value):
        return float(raw_value)
    return float(row[adjusted_column])


def _execution_price(row: pd.Series, side: str, costs: CostConfig) -> float:
    base = _row_price(row, "raw_open", "open")
    direction = 1.0 if side == "BUY" else -1.0
    return base * (1.0 + direction * costs.slippage_bps / 10_000.0)


def _is_locked(row: pd.Series, side: str) -> bool:
    if not bool(row.get("is_trading", True)) or float(row.get("volume", 0)) <= 0:
        return True
    open_price = _row_price(row, "raw_open", "open")
    if side == "BUY" and pd.notna(row.get("limit_up")):
        return open_price >= float(row["limit_up"]) * (1.0 - 1e-6)
    if side == "SELL" and pd.notna(row.get("limit_down")):
        return open_price <= float(row["limit_down"]) * (1.0 + 1e-6)
    return False


def _fees(notional: float, side: str, costs: CostConfig) -> float:
    commission = max(costs.minimum_commission, notional * costs.commission_bps / 10_000.0)
    transfer = notional * costs.transfer_bps / 10_000.0
    sell_tax = notional * costs.sell_tax_bps / 10_000.0 if side == "SELL" else 0.0
    return commission + transfer + sell_tax


def _maximum_fill_quantity(row: pd.Series, requested: int, config: PortfolioConfig) -> int:
    lot_size = int(row.get("lot_size", config.default_lot_size))
    participation_cap = int(float(row["volume"]) * config.max_participation_rate)
    capped = min(abs(requested), participation_cap)
    return max(0, capped // lot_size * lot_size)


def _schedule_signals(signals: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.DataFrame]:
    schedule: dict[pd.Timestamp, pd.DataFrame] = {}
    for signal_date, group in signals.groupby("date", sort=True):
        position = trading_dates.searchsorted(pd.Timestamp(signal_date), side="right")
        if position < len(trading_dates):
            scheduled = group.copy()
            scheduled["signal_date"] = pd.Timestamp(signal_date)
            schedule[trading_dates[position]] = scheduled
    return schedule


def run_backtest(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    portfolio_config: PortfolioConfig,
    cost_config: CostConfig,
    end_date: pd.Timestamp | None = None,
    corporate_actions: pd.DataFrame | None = None,
    market_context: BacktestMarketContext | None = None,
    capture_positions: bool = True,
) -> BacktestResult:
    if signals.empty:
        raise ValueError("At least one signal date is required for backtesting")

    context = market_context or prepare_backtest_market(bars)
    benchmark_returns = context.benchmark_returns
    trading_dates = context.trading_dates
    market_by_date = context.market_by_date
    schedule = _schedule_signals(signals, trading_dates)
    if not schedule:
        raise ValueError("No signal has a following trading day")

    first_date = min(schedule)
    final_date = pd.Timestamp(end_date) if end_date is not None else trading_dates.max()
    active_dates = trading_dates[(trading_dates >= first_date) & (trading_dates <= final_date)]

    cash = float(portfolio_config.initial_cash)
    universe_equal_weight_nav = float(portfolio_config.initial_cash)
    holdings: dict[str, int] = {}
    last_close: dict[str, float] = {}
    orders: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    actions_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if corporate_actions is not None and not corporate_actions.empty:
        actions = corporate_actions.copy()
        actions["ex_date"] = pd.to_datetime(actions["ex_date"]).dt.normalize()
        actions_by_date = {
            pd.Timestamp(action_date): group
            for action_date, group in actions.groupby("ex_date", sort=True)
        }

    for date in active_dates:
        day = market_by_date.xs(date, level="date", drop_level=True)
        for action in actions_by_date.get(pd.Timestamp(date), pd.DataFrame()).itertuples(index=False):
            code = str(action.code)
            quantity_before = holdings.get(code, 0)
            if quantity_before <= 0:
                continue
            cash_dividend = quantity_before * float(action.cash_dividend_per_share)
            bonus_quantity = int(np.floor(quantity_before * float(action.bonus_share_ratio)))
            rights_quantity = int(np.floor(quantity_before * float(action.rights_share_ratio)))
            cash += cash_dividend
            holdings[code] = quantity_before + bonus_quantity
            action_rows.append(
                {
                    "date": date,
                    "code": code,
                    "quantity_before": quantity_before,
                    "cash_dividend": cash_dividend,
                    "bonus_quantity": bonus_quantity,
                    "quantity_after": holdings[code],
                    "rights_eligible_quantity": rights_quantity,
                    "rights_price": getattr(action, "rights_price", np.nan),
                    "rights_policy": "NOT_SUBSCRIBED",
                }
            )
        for code in set(holdings).intersection(day.index):
            last_close[code] = _row_price(day.loc[code], "raw_close", "close")

        open_value = cash + sum(
            quantity * _row_price(day.loc[code], "raw_open", "open")
            if code in day.index
            else quantity * last_close.get(code, 0.0)
            for code, quantity in holdings.items()
        )

        traded_notional = 0.0
        fees_paid = 0.0
        if date in schedule:
            ranked = schedule[date]
            signal_date = pd.Timestamp(ranked["signal_date"].iloc[0])
            holdings_before = set(holdings)
            ranked = ranked.loc[ranked["code"].isin(day.index)].copy()
            market_columns = [
                column
                for column in [
                    "code",
                    "open",
                    "raw_open",
                    "volume",
                    "is_trading",
                    "is_st",
                    "lot_size",
                    "limit_up",
                    "limit_down",
                ]
                if column in day.columns
            ]
            ranked = ranked.drop(columns=[column for column in market_columns if column != "code"], errors="ignore")
            ranked = ranked.merge(day[market_columns].reset_index(drop=True), on="code", how="left")
            ranked["tradable"] = ranked["tradable"] & ranked["is_trading"] & ~ranked["is_st"]
            selected = select_portfolio(ranked, set(holdings), portfolio_config)
            target_weights = dict(zip(selected["code"], selected["target_weight"], strict=False))
            selected_buckets = (
                selected.get("size_bucket", pd.Series("UNKNOWN", index=selected.index))
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
            )
            large_mask = selected_buckets.eq("LARGE")
            mid_mask = selected_buckets.eq("MID")

            requested_orders: list[tuple[str, str, int]] = []
            all_codes = set(holdings) | set(target_weights)
            score_by_code = ranked.set_index("code")["score"].to_dict()
            for code in sorted(all_codes):
                if code not in day.index:
                    continue
                row = day.loc[code]
                lot_size = int(row.get("lot_size", portfolio_config.default_lot_size))
                reference_price = _row_price(row, "raw_open", "open")
                target_weight = target_weights.get(code, 0.0)
                current_quantity = holdings.get(code, 0)
                current_weight = (
                    current_quantity * reference_price / open_value if open_value else 0.0
                )
                if (
                    target_weight > 0
                    and current_quantity > 0
                    and abs(current_weight - target_weight)
                    <= portfolio_config.rebalance_weight_tolerance
                ):
                    # Keep an existing position unchanged inside the no-trade band.  Full exits
                    # and new entries are never suppressed by this rule.
                    target_quantity = current_quantity
                else:
                    target_value = open_value * target_weight
                    target_quantity = int(target_value / reference_price) // lot_size * lot_size
                difference = target_quantity - current_quantity
                if difference:
                    requested_orders.append((code, "BUY" if difference > 0 else "SELL", difference))

            requested_orders.sort(
                key=lambda item: (
                    0 if item[1] == "SELL" else 1,
                    (
                        float(score_by_code.get(item[0], float("-inf")))
                        if item[1] == "SELL"
                        else -float(score_by_code.get(item[0], float("-inf")))
                    ),
                    item[0],
                )
            )
            for code, side, requested in requested_orders:
                row = day.loc[code]
                order_record: dict[str, Any] = {
                    "date": date,
                    "code": code,
                    "side": side,
                    "requested_quantity": abs(int(requested)),
                    "filled_quantity": 0,
                    "price": np.nan,
                    "notional": 0.0,
                    "fees": 0.0,
                    "status": "REJECTED",
                    "reason": "",
                }
                reference_notional = abs(requested) * _row_price(row, "raw_open", "open")
                is_full_exit = side == "SELL" and target_weights.get(code, 0.0) == 0.0
                if reference_notional < portfolio_config.min_trade_value and not is_full_exit:
                    order_record["reason"] = "BELOW_MIN_TRADE_VALUE"
                    orders.append(order_record)
                    continue
                if _is_locked(row, side):
                    order_record["reason"] = "SUSPENDED_OR_LIMIT_LOCKED"
                    orders.append(order_record)
                    continue

                quantity = _maximum_fill_quantity(row, requested, portfolio_config)
                if side == "SELL":
                    quantity = min(quantity, holdings.get(code, 0))
                price = _execution_price(row, side, cost_config)

                if side == "BUY" and quantity > 0:
                    lot_size = int(row.get("lot_size", portfolio_config.default_lot_size))
                    while quantity > 0:
                        notional = quantity * price
                        fee = _fees(notional, side, cost_config)
                        if notional + fee <= cash:
                            break
                        quantity -= lot_size

                if quantity <= 0:
                    order_record["reason"] = "CAPACITY_OR_CASH_LIMIT"
                    orders.append(order_record)
                    continue

                notional = quantity * price
                fee = _fees(notional, side, cost_config)
                if side == "SELL":
                    cash += notional - fee
                    holdings[code] = holdings.get(code, 0) - quantity
                    if holdings[code] == 0:
                        del holdings[code]
                else:
                    cash -= notional + fee
                    holdings[code] = holdings.get(code, 0) + quantity

                traded_notional += notional
                fees_paid += fee
                order_record.update(
                    {
                        "filled_quantity": quantity,
                        "price": price,
                        "notional": notional,
                        "fees": fee,
                        "status": "FILLED" if quantity == abs(requested) else "PARTIALLY_FILLED",
                        "reason": "",
                    }
                )
                orders.append(order_record)

            rebalance_rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": date,
                    "selected_count": len(selected),
                    "entered_count": len(set(holdings).difference(holdings_before)),
                    "exited_count": len(holdings_before.difference(holdings)),
                    "target_exposure": float(sum(target_weights.values())),
                    "traded_notional": traded_notional,
                    "fees": fees_paid,
                    "turnover": traded_notional / open_value if open_value else 0.0,
                    "large_selected_count": int(large_mask.sum()),
                    "mid_selected_count": int(mid_mask.sum()),
                    "large_target_weight": float(selected.loc[large_mask, "target_weight"].sum()),
                    "mid_target_weight": float(selected.loc[mid_mask, "target_weight"].sum()),
                }
            )

        market_value = 0.0
        for code, quantity in holdings.items():
            if code in day.index:
                last_close[code] = _row_price(day.loc[code], "raw_close", "close")
            close_price = last_close.get(code, 0.0)
            value = quantity * close_price
            market_value += value
            if capture_positions:
                positions.append(
                    {
                        "date": date,
                        "code": code,
                        "quantity": quantity,
                        "close": close_price,
                        "market_value": value,
                        "size_bucket": str(day.loc[code].get("size_bucket", "UNKNOWN"))
                        if code in day.index
                        else "UNKNOWN",
                        "industry": str(day.loc[code].get("industry", "UNKNOWN"))
                        if code in day.index
                        else "UNKNOWN",
                    }
                )
        universe_equal_weight_nav *= 1.0 + float(benchmark_returns.get(date, 0.0))
        nav_rows.append(
            {
                "date": date,
                "cash": cash,
                "market_value": market_value,
                "nav": cash + market_value,
                "universe_equal_weight_nav": universe_equal_weight_nav,
                "holdings": len(holdings),
            }
        )

    return BacktestResult(
        nav=pd.DataFrame(nav_rows),
        orders=pd.DataFrame(orders),
        positions=pd.DataFrame(positions),
        rebalance_summary=pd.DataFrame(rebalance_rows),
        corporate_actions=pd.DataFrame(action_rows),
    )
