from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

RAW_FEATURES = [
    "momentum_20_5",
    "momentum_60_5",
    "trend_quality_60",
    "reversal_5",
    "high_position_120",
    "idio_vol_60",
    "downside_vol_60",
    "volume_z20",
    "price_volume_confirmation_5",
    "clv_5",
]

MODEL_FEATURES = [f"{name}_z" for name in RAW_FEATURES]

BASELINE_WEIGHTS = {
    "momentum_20_5_z": 0.16,
    "momentum_60_5_z": 0.18,
    "trend_quality_60_z": 0.16,
    "reversal_5_z": 0.10,
    "high_position_120_z": 0.10,
    "idio_vol_60_z": -0.10,
    "downside_vol_60_z": -0.08,
    "volume_z20_z": 0.02,
    "price_volume_confirmation_5_z": 0.06,
    "clv_5_z": 0.04,
}


@dataclass(frozen=True)
class ResearchConfig:
    horizon_days: int = 20
    min_history_days: int = 120
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    rebalance_frequency: str = "month_end"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    min_cross_section_size: int = 200
    min_liquidity_percentile: float = 0.20
    walk_forward_window: str = "expanding"
    walk_forward_train_months: int = 36
    walk_forward_validation_months: int = 12
    walk_forward_test_months: int = 12
    walk_forward_step_months: int = 12


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    early_stopping_rounds: int = 30
    random_state: int = 42


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int = 50
    entry_rank: int = 50
    exit_rank: int = 70
    max_stock_weight: float = 0.03
    max_industry_weight: float = 0.25
    max_participation_rate: float = 0.05
    initial_cash: float = 1_000_000.0
    default_lot_size: int = 100
    weighting_method: str = "equal"
    cash_buffer: float = 0.02
    min_trade_value: float = 1_000.0
    max_mid_position_fraction: float = 0.60
    max_mid_cap_weight: float = 0.70
    max_replacements_per_rebalance: int | None = None
    min_replacement_rank_improvement: int = 0
    rebalance_weight_tolerance: float = 0.0


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 3.0
    minimum_commission: float = 5.0
    sell_tax_bps: float = 5.0
    transfer_bps: float = 0.1
    slippage_bps: float = 5.0


@dataclass(frozen=True)
class AppConfig:
    research: ResearchConfig = field(default_factory=ResearchConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    costs: CostConfig = field(default_factory=CostConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()

    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)

    return AppConfig(
        research=ResearchConfig(**payload.get("research", {})),
        model=ModelConfig(**payload.get("model", {})),
        portfolio=PortfolioConfig(**payload.get("portfolio", {})),
        costs=CostConfig(**payload.get("costs", {})),
    )


def apply_portfolio_overrides(
    config: AppConfig,
    *,
    holdings: int | None = None,
    initial_cash: float | None = None,
    weighting_method: str | None = None,
    cash_buffer: float | None = None,
    min_trade_value: float | None = None,
    exit_rank: int | None = None,
    max_replacements_per_rebalance: int | None = None,
    min_replacement_rank_improvement: int | None = None,
    rebalance_weight_tolerance: float | None = None,
) -> AppConfig:
    """Resolve user-facing portfolio choices into internally consistent constraints."""
    portfolio = config.portfolio
    if holdings is not None:
        if holdings < 1:
            raise ValueError("holdings must be at least 1")
        buffer_ratio = portfolio.exit_rank / portfolio.top_k
        exit_rank = max(holdings + 1, math.ceil(holdings * buffer_ratio))
        equal_weight = 1.0 / holdings
        portfolio = replace(
            portfolio,
            top_k=holdings,
            entry_rank=holdings,
            exit_rank=exit_rank,
            max_stock_weight=max(portfolio.max_stock_weight, equal_weight),
            max_industry_weight=max(portfolio.max_industry_weight, equal_weight),
        )
    if initial_cash is not None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        portfolio = replace(portfolio, initial_cash=float(initial_cash))
    if weighting_method is not None:
        if weighting_method not in {"equal", "inverse_volatility", "score_tilt"}:
            raise ValueError("weighting_method must be equal, inverse_volatility, or score_tilt")
        portfolio = replace(portfolio, weighting_method=weighting_method)
    if cash_buffer is not None:
        if not 0 <= cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        portfolio = replace(portfolio, cash_buffer=float(cash_buffer))
    if min_trade_value is not None:
        if min_trade_value < 0:
            raise ValueError("min_trade_value must be non-negative")
        portfolio = replace(portfolio, min_trade_value=float(min_trade_value))
    if exit_rank is not None:
        if exit_rank < portfolio.entry_rank:
            raise ValueError("exit_rank must be greater than or equal to entry_rank")
        portfolio = replace(portfolio, exit_rank=int(exit_rank))
    if max_replacements_per_rebalance is not None:
        if max_replacements_per_rebalance < 0:
            raise ValueError("max_replacements_per_rebalance must be non-negative")
        portfolio = replace(
            portfolio,
            max_replacements_per_rebalance=int(max_replacements_per_rebalance),
        )
    if min_replacement_rank_improvement is not None:
        if min_replacement_rank_improvement < 0:
            raise ValueError("min_replacement_rank_improvement must be non-negative")
        portfolio = replace(
            portfolio,
            min_replacement_rank_improvement=int(min_replacement_rank_improvement),
        )
    if rebalance_weight_tolerance is not None:
        if not 0 <= rebalance_weight_tolerance < 1:
            raise ValueError("rebalance_weight_tolerance must be in [0, 1)")
        portfolio = replace(
            portfolio,
            rebalance_weight_tolerance=float(rebalance_weight_tolerance),
        )
    return replace(config, portfolio=portfolio)


def apply_research_overrides(
    config: AppConfig,
    *,
    walk_forward_window: str | None = None,
    train_months: int | None = None,
    validation_months: int | None = None,
    test_months: int | None = None,
    step_months: int | None = None,
) -> AppConfig:
    research = config.research
    if walk_forward_window is not None:
        if walk_forward_window not in {"expanding", "rolling"}:
            raise ValueError("walk_forward_window must be expanding or rolling")
        research = replace(research, walk_forward_window=walk_forward_window)
    values = {
        "walk_forward_train_months": train_months,
        "walk_forward_validation_months": validation_months,
        "walk_forward_test_months": test_months,
        "walk_forward_step_months": step_months,
    }
    for field_name, value in values.items():
        if value is not None:
            if value < 1:
                raise ValueError(f"{field_name} must be at least 1")
            research = replace(research, **{field_name: int(value)})
    if research.walk_forward_step_months != research.walk_forward_test_months:
        raise ValueError(
            "walk_forward_step_months must equal walk_forward_test_months so out-of-sample "
            "windows are contiguous and non-overlapping"
        )
    return replace(config, research=research)
