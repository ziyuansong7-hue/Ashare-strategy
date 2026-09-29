"""Execution-aware daily portfolio backtester."""

from .engine import BacktestMarketContext, BacktestResult, prepare_backtest_market, run_backtest

__all__ = ["BacktestMarketContext", "BacktestResult", "prepare_backtest_market", "run_backtest"]
