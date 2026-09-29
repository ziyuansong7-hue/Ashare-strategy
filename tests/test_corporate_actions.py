import pandas as pd

from ashare_quant.backtest import run_backtest
from ashare_quant.config import CostConfig, PortfolioConfig


def test_backtest_books_cash_dividend_bonus_shares_and_discloses_rights_policy():
    dates = pd.bdate_range("2024-01-31", periods=4)
    bars = pd.DataFrame(
        {
            "date": dates,
            "code": ["000001"] * len(dates),
            "open": [10.0] * len(dates),
            "high": [10.0] * len(dates),
            "low": [10.0] * len(dates),
            "close": [10.0] * len(dates),
            "raw_open": [10.0] * len(dates),
            "raw_close": [10.0] * len(dates),
            "volume": [100_000] * len(dates),
            "amount": [1_000_000] * len(dates),
            "industry": ["BANK"] * len(dates),
            "is_trading": [True] * len(dates),
            "is_st": [False] * len(dates),
            "tradable": [True] * len(dates),
            "lot_size": [100] * len(dates),
        }
    )
    signals = pd.DataFrame(
        {
            "date": [dates[0]],
            "code": ["000001"],
            "score": [1.0],
            "industry": ["BANK"],
            "tradable": [True],
        }
    )
    actions = pd.DataFrame(
        {
            "code": ["000001"],
            "ex_date": [dates[2]],
            "cash_dividend_per_share": [0.2],
            "bonus_share_ratio": [0.1],
            "rights_share_ratio": [0.3],
            "rights_price": [8.0],
        }
    )
    result = run_backtest(
        bars,
        signals,
        portfolio_config=PortfolioConfig(
            top_k=1,
            entry_rank=1,
            exit_rank=1,
            max_stock_weight=1.0,
            max_industry_weight=1.0,
            max_participation_rate=1.0,
            initial_cash=100_000,
            cash_buffer=0.0,
        ),
        cost_config=CostConfig(
            commission_bps=0,
            minimum_commission=0,
            sell_tax_bps=0,
            transfer_bps=0,
            slippage_bps=0,
        ),
        corporate_actions=actions,
    )

    booked = result.corporate_actions.iloc[0]
    assert booked["quantity_before"] == 10_000
    assert booked["cash_dividend"] == 2_000
    assert booked["bonus_quantity"] == 1_000
    assert booked["quantity_after"] == 11_000
    assert booked["rights_eligible_quantity"] == 3_000
    assert booked["rights_policy"] == "NOT_SUBSCRIBED"
