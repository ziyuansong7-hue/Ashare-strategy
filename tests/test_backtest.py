import pandas as pd
import pytest

from ashare_quant.backtest import prepare_backtest_market, run_backtest
from ashare_quant.config import CostConfig, PortfolioConfig


def test_limit_locked_buy_is_rejected():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        {
            "date": dates,
            "code": ["000001"] * 3,
            "open": [10.0, 11.0, 10.5],
            "high": [10.5, 11.0, 10.8],
            "low": [9.8, 11.0, 10.2],
            "close": [10.0, 11.0, 10.6],
            "volume": [100_000] * 3,
            "amount": [1_000_000] * 3,
            "industry": ["TEST"] * 3,
            "is_trading": [True] * 3,
            "is_st": [False] * 3,
            "tradable": [True] * 3,
            "lot_size": [100] * 3,
            "limit_up": [11.0] * 3,
            "limit_down": [9.0] * 3,
        }
    )
    signals = pd.DataFrame(
        {
            "date": [dates[0]],
            "code": ["000001"],
            "score": [1.0],
            "industry": ["TEST"],
            "tradable": [True],
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
            initial_cash=100_000,
        ),
        cost_config=CostConfig(),
    )
    assert result.orders.iloc[0]["status"] == "REJECTED"
    assert result.orders.iloc[0]["reason"] == "SUSPENDED_OR_LIMIT_LOCKED"
    assert result.nav.iloc[-1]["nav"] == 100_000


def test_month_end_signals_execute_next_session_and_rotate_holdings():
    dates = pd.bdate_range("2024-01-30", "2024-03-04")
    bars = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 100_000,
                "amount": 1_000_000,
                "industry": industry,
                "is_trading": True,
                "is_st": False,
                "tradable": True,
                "lot_size": 100,
            }
            for date in dates
            for code, industry in [("000001", "BANK"), ("000002", "PROPERTY")]
        ]
    )
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "code": ["000001", "000002", "000001", "000002"],
            "score": [2.0, 1.0, 1.0, 2.0],
            "industry": ["BANK", "PROPERTY", "BANK", "PROPERTY"],
            "tradable": True,
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
        ),
        cost_config=CostConfig(
            commission_bps=0,
            minimum_commission=0,
            sell_tax_bps=0,
            transfer_bps=0,
            slippage_bps=0,
        ),
    )

    rebalances = result.rebalance_summary
    assert rebalances["signal_date"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]
    assert rebalances["execution_date"].tolist() == [
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]
    assert rebalances["target_exposure"].tolist() == [0.98, 0.98]
    assert rebalances.loc[1, "entered_count"] == 1
    assert rebalances.loc[1, "exited_count"] == 1


def test_equal_weight_benchmark_uses_only_point_in_time_members():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100_000,
                "amount": 1_000_000,
                "industry": "TEST",
                "is_trading": True,
                "is_st": False,
                "tradable": is_member,
                "is_universe_member": is_member,
                "lot_size": 100,
            }
            for date_index, date in enumerate(dates)
            for code, close, is_member in [
                ("000001", 10.0 * (1.1**date_index), True),
                ("000002", 10.0 * (2.0**date_index), False),
            ]
        ]
    )
    signals = pd.DataFrame(
        {
            "date": [dates[0]],
            "code": ["000001"],
            "score": [1.0],
            "industry": ["TEST"],
            "tradable": [True],
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
        ),
        cost_config=CostConfig(
            commission_bps=0,
            minimum_commission=0,
            sell_tax_bps=0,
            transfer_bps=0,
            slippage_bps=0,
        ),
    )

    assert result.nav["universe_equal_weight_nav"].tolist() == pytest.approx(
        [110_000.0, 121_000.0]
    )


def test_repeated_research_context_preserves_results_and_prioritizes_high_score_buys():
    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = pd.DataFrame(
        [
            {
                "date": date,
                "code": code,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 100_000,
                "amount": 1_000_000,
                "industry": "TEST",
                "is_trading": True,
                "is_st": False,
                "tradable": True,
                "lot_size": 100,
            }
            for date in dates
            for code in ["000001", "000002"]
        ]
    )
    signals = pd.DataFrame(
        {
            "date": [dates[0], dates[0]],
            "code": ["000001", "000002"],
            "score": [1.0, 2.0],
            "industry": ["TEST", "TEST"],
            "tradable": True,
        }
    )
    portfolio = PortfolioConfig(
        top_k=2,
        entry_rank=2,
        exit_rank=2,
        max_stock_weight=0.50,
        max_industry_weight=1.0,
        max_participation_rate=1.0,
        initial_cash=100_000,
    )
    costs = CostConfig(
        commission_bps=0,
        minimum_commission=0,
        sell_tax_bps=0,
        transfer_bps=0,
        slippage_bps=0,
    )

    ordinary = run_backtest(
        bars,
        signals,
        portfolio_config=portfolio,
        cost_config=costs,
    )
    repeated = run_backtest(
        bars,
        signals,
        portfolio_config=portfolio,
        cost_config=costs,
        market_context=prepare_backtest_market(bars),
        capture_positions=False,
    )

    assert ordinary.orders["code"].tolist() == ["000002", "000001"]
    pd.testing.assert_frame_equal(ordinary.nav, repeated.nav)
    pd.testing.assert_frame_equal(ordinary.orders, repeated.orders)
    assert repeated.positions.empty
