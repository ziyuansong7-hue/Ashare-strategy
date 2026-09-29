import pandas as pd
from pandas.testing import assert_frame_equal

from ashare_quant.data.schema import normalize_and_validate_bars
from ashare_quant.data.synthetic import generate_synthetic_bars
from ashare_quant.features import compute_raw_factors


def test_features_do_not_change_when_future_rows_are_appended():
    raw = generate_synthetic_bars(symbols=20, days=280, seed=7)
    bars, _ = normalize_and_validate_bars(raw)
    cutoff = pd.Timestamp(bars["date"].drop_duplicates().sort_values().iloc[239])
    historical = bars.loc[bars["date"] <= cutoff]

    short_features = compute_raw_factors(historical)
    full_features = compute_raw_factors(bars)
    full_until_cutoff = full_features.loc[full_features["date"] <= cutoff]

    columns = [
        "date",
        "code",
        "momentum_60_5",
        "trend_quality_60",
        "reversal_5",
        "high_position_120",
    ]
    assert_frame_equal(
        short_features[columns].reset_index(drop=True),
        full_until_cutoff[columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_bottom_liquidity_quantile_is_not_tradable():
    raw = generate_synthetic_bars(symbols=20, days=260, seed=11)
    liquidity_scale = pd.Series(pd.factorize(raw["code"])[0] + 1, index=raw.index)
    raw["amount"] = raw["amount"] * liquidity_scale
    bars, _ = normalize_and_validate_bars(raw)

    factors = compute_raw_factors(bars, min_liquidity_percentile=0.20)
    latest = factors.loc[factors["date"] == factors["date"].max()].sort_values(
        "liquidity_percentile"
    )

    assert latest.iloc[:4]["tradable"].eq(False).all()
    assert latest.iloc[-1]["tradable"]
