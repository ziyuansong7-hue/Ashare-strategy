"""Factor and portfolio analytics."""

from .diagnostics import write_research_diagnostics
from .factors import factor_report, segment_model_report
from .performance import annual_return_table, attach_benchmark_nav, performance_metrics
from .validation import (
    flatten_backtest_metrics,
    maximum_drawdown_episode,
    parameter_robustness_summary,
    portfolio_regime_attribution,
    rank_ic_summary,
    scale_costs,
    selected_return_summary,
    size_exposure_attribution,
    summarize_backtest,
)

__all__ = [
    "annual_return_table",
    "attach_benchmark_nav",
    "factor_report",
    "flatten_backtest_metrics",
    "maximum_drawdown_episode",
    "parameter_robustness_summary",
    "performance_metrics",
    "portfolio_regime_attribution",
    "rank_ic_summary",
    "scale_costs",
    "segment_model_report",
    "selected_return_summary",
    "size_exposure_attribution",
    "summarize_backtest",
    "write_research_diagnostics",
]
