import pytest

from ashare_quant.config import AppConfig, apply_portfolio_overrides, apply_research_overrides


@pytest.mark.parametrize(
    ("holdings", "expected_exit_rank"),
    [(3, 5), (5, 7), (50, 70)],
)
def test_portfolio_size_override_keeps_constraints_consistent(holdings: int, expected_exit_rank: int):
    resolved = apply_portfolio_overrides(AppConfig(), holdings=holdings, initial_cash=200_000)
    portfolio = resolved.portfolio

    assert portfolio.top_k == holdings
    assert portfolio.entry_rank == holdings
    assert portfolio.exit_rank == expected_exit_rank
    assert portfolio.max_stock_weight >= 1.0 / holdings
    assert portfolio.max_industry_weight >= 1.0 / holdings
    assert portfolio.initial_cash == 200_000


def test_portfolio_override_rejects_invalid_user_inputs():
    with pytest.raises(ValueError, match="holdings must be at least 1"):
        apply_portfolio_overrides(AppConfig(), holdings=0)
    with pytest.raises(ValueError, match="initial_cash must be positive"):
        apply_portfolio_overrides(AppConfig(), initial_cash=0)


def test_walk_forward_overrides_require_contiguous_non_overlapping_windows():
    resolved = apply_research_overrides(
        AppConfig(),
        walk_forward_window="rolling",
        train_months=24,
        validation_months=6,
        test_months=6,
        step_months=6,
    )
    assert resolved.research.walk_forward_window == "rolling"
    assert resolved.research.walk_forward_train_months == 24
    with pytest.raises(ValueError, match="contiguous and non-overlapping"):
        apply_research_overrides(AppConfig(), test_months=12, step_months=6)
