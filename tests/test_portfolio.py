import pandas as pd
import pytest

from ashare_quant.config import PortfolioConfig
from ashare_quant.portfolio import select_portfolio, validate_portfolio_universe


def test_portfolio_respects_industry_cap():
    ranked = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(12)],
            "score": list(reversed(range(12))),
            "industry": ["BANK"] * 8 + ["TECH"] * 4,
            "tradable": True,
        }
    )
    config = PortfolioConfig(
        top_k=8,
        entry_rank=12,
        exit_rank=12,
        max_stock_weight=0.20,
        max_industry_weight=0.50,
    )
    selected = select_portfolio(ranked, set(), config)
    assert len(selected) == 8
    assert selected["industry"].value_counts().max() <= 4


def test_portfolio_rejects_an_undersized_universe():
    ranked = pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "score": [2.0, 1.0],
            "industry": ["BANK", "PROPERTY"],
            "tradable": True,
        }
    )

    with pytest.raises(ValueError, match="Only 2 tradable candidates"):
        select_portfolio(ranked, set(), PortfolioConfig(top_k=3, entry_rank=3, exit_rank=4))


def test_free_data_profile_disables_industry_cap_when_every_industry_is_unknown():
    ranked = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(20)],
            "score": list(reversed(range(20))),
            "industry": ["UNKNOWN"] * 20,
            "tradable": True,
        }
    )
    config = PortfolioConfig(
        top_k=10,
        entry_rank=10,
        exit_rank=15,
        max_stock_weight=0.10,
        max_industry_weight=0.25,
    )

    selected = select_portfolio(ranked, set(), config)

    assert len(selected) == 10
    assert abs(float(selected["target_weight"].sum()) - (1.0 - config.cash_buffer)) < 1e-9


def test_portfolio_rejects_a_rank_buffer_that_can_never_exit():
    ranked = pd.DataFrame(
        {
            "code": ["000001", "000002", "000003"],
            "score": [3.0, 2.0, 1.0],
            "industry": ["BANK", "TECH", "UTILITY"],
            "tradable": True,
        }
    )
    config = PortfolioConfig(
        top_k=2,
        entry_rank=2,
        exit_rank=3,
        max_stock_weight=0.50,
        max_industry_weight=1.0,
    )

    with pytest.raises(ValueError, match="could never leave the rank buffer"):
        select_portfolio(ranked, {"000001", "000002"}, config)


def test_preflight_rejects_top_k_larger_than_monthly_universe():
    candidates = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 20,
            "code": [f"{index:06d}" for index in range(20)],
            "industry": ["TEST"] * 20,
            "tradable": True,
        }
    )

    with pytest.raises(ValueError, match="only 20 tradable candidates for top_k=50"):
        validate_portfolio_universe(candidates, PortfolioConfig())


def test_portfolio_caps_mid_positions_and_weight_when_size_labels_exist():
    ranked = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(10)],
            "score": list(reversed(range(10))),
            "industry": [f"IND{index}" for index in range(10)],
            "size_bucket": ["MID"] * 5 + ["LARGE"] * 5,
            "tradable": True,
        }
    )
    config = PortfolioConfig(
        top_k=5,
        entry_rank=10,
        exit_rank=10,
        max_stock_weight=0.40,
        max_industry_weight=1.0,
        weighting_method="score_tilt",
        max_mid_position_fraction=0.60,
        max_mid_cap_weight=0.70,
    )

    selected = select_portfolio(ranked, set(), config)

    assert int(selected["size_bucket"].eq("MID").sum()) == 3
    assert int(selected["size_bucket"].eq("LARGE").sum()) == 2
    assert selected.loc[selected["size_bucket"] == "MID", "target_weight"].sum() <= 0.70 + 1e-9


def test_low_turnover_policy_caps_monthly_replacements():
    ranked = pd.DataFrame(
        {
            "code": [f"{index:06d}" for index in range(20)],
            "score": list(reversed(range(20))),
            "industry": ["UNKNOWN"] * 20,
            "tradable": True,
        }
    )
    current = {f"{index:06d}" for index in range(5, 10)}
    config = PortfolioConfig(
        top_k=5,
        entry_rank=5,
        exit_rank=6,
        max_stock_weight=0.20,
        max_industry_weight=1.0,
        max_replacements_per_rebalance=1,
        min_replacement_rank_improvement=3,
    )

    selected = select_portfolio(ranked, current, config)

    selected_codes = set(selected["code"])
    assert len(selected_codes - current) == 1
    assert len(current - selected_codes) == 1
    assert "000000" in selected_codes
