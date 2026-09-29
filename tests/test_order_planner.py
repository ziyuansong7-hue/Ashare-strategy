import pandas as pd

from ashare_quant.config import CostConfig, PortfolioConfig
from ashare_quant.portfolio import plan_orders


def test_order_planner_sells_departure_and_buys_target_positions():
    date = pd.Timestamp("2024-03-29")
    scores = pd.DataFrame(
        {
            "date": [date] * 3,
            "code": ["000001", "000002", "000003"],
            "industry": ["BANK", "TECH", "UTILITY"],
            "xgboost_score": [3.0, 2.0, 1.0],
            "xgboost_reason_1": ["momentum_60_5", "trend_quality_60", "reversal_5"],
            "xgboost_reason_2": ["trend_quality_60", "momentum_20_5", "clv_5"],
        }
    )
    bars = pd.DataFrame(
        {
            "date": [date] * 3,
            "code": ["000001", "000002", "000003"],
            "open": [10.0] * 3,
            "high": [10.0] * 3,
            "low": [10.0] * 3,
            "close": [10.0] * 3,
            "raw_close": [10.0] * 3,
            "volume": [100_000] * 3,
            "amount": [1_000_000] * 3,
            "industry": ["BANK", "TECH", "UTILITY"],
            "is_trading": [True] * 3,
            "is_st": [False] * 3,
            "lot_size": [100] * 3,
        }
    )
    positions = pd.DataFrame({"code": ["000003"], "quantity": [100]})
    orders, summary = plan_orders(
        scores,
        bars,
        positions,
        cash=99_000,
        portfolio_config=PortfolioConfig(
            top_k=2,
            entry_rank=2,
            exit_rank=2,
            max_stock_weight=0.5,
            max_industry_weight=1.0,
            cash_buffer=0.1,
            min_trade_value=1_000,
        ),
        cost_config=CostConfig(),
    )

    actions = dict(zip(orders["code"], orders["action"], strict=False))
    assert actions == {"000003": "SELL", "000001": "BUY", "000002": "BUY"}
    assert summary["estimated_equity"] == 100_000
    assert summary["target_positions"] == 2
    assert summary["target_exposure"] == 0.9
