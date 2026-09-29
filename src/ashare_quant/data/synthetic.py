from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_bars(
    *,
    symbols: int = 80,
    days: int = 900,
    seed: int = 42,
    start: str = "2021-01-04",
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for integration testing only."""
    if symbols < 20 or days < 260:
        raise ValueError("Synthetic demo requires at least 20 symbols and 260 days")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=days)
    codes = [f"{index:06d}" for index in range(1, symbols + 1)]
    industries = [f"SECTOR_{index % 8}" for index in range(symbols)]
    large_cutoff = max(1, round(symbols * 300 / 800))
    size_buckets = ["LARGE" if index < large_cutoff else "MID" for index in range(symbols)]
    latent_quality = rng.normal(0, 1, symbols)
    latent_volatility = rng.uniform(0.008, 0.025, symbols)
    base_volume = rng.lognormal(mean=15.0, sigma=0.7, size=symbols)
    shares_outstanding = rng.lognormal(mean=19.0, sigma=0.6, size=symbols)

    close = rng.uniform(8, 45, symbols)
    return_history = np.zeros((days, symbols), dtype=float)
    rows: list[dict[str, object]] = []

    for day_index, date in enumerate(dates):
        market_return = rng.normal(0.00015, 0.007)
        sector_return = rng.normal(0, 0.004, 8)
        if day_index >= 20:
            medium_momentum = return_history[day_index - 20 : day_index - 5].mean(axis=0)
            short_return = return_history[day_index - 5 : day_index].sum(axis=0)
        else:
            medium_momentum = np.zeros(symbols)
            short_return = np.zeros(symbols)

        noise = rng.normal(0, latent_volatility)
        daily_return = (
            market_return
            + np.array([sector_return[index % 8] for index in range(symbols)])
            + 0.18 * medium_momentum
            - 0.035 * short_return
            + 0.00008 * latent_quality
            + noise
        )
        daily_return = np.clip(daily_return, -0.095, 0.095)
        overnight = rng.normal(0, 0.003, symbols)
        open_price = close * (1.0 + overnight)
        new_close = open_price * (1.0 + daily_return)
        intraday_spread = rng.uniform(0.002, 0.018, symbols)
        high = np.maximum(open_price, new_close) * (1.0 + intraday_spread)
        low = np.minimum(open_price, new_close) * (1.0 - intraday_spread)
        volume = (
            base_volume
            * (1.0 + 6.0 * np.abs(daily_return))
            * rng.lognormal(mean=0, sigma=0.22, size=symbols)
        ).astype(int)
        amount = volume * (open_price + new_close) / 2.0
        market_cap = shares_outstanding * new_close

        for symbol_index, code in enumerate(codes):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": open_price[symbol_index],
                    "high": high[symbol_index],
                    "low": low[symbol_index],
                    "close": new_close[symbol_index],
                    "volume": volume[symbol_index],
                    "amount": amount[symbol_index],
                    "industry": industries[symbol_index],
                    "size_bucket": size_buckets[symbol_index],
                    "market_cap": market_cap[symbol_index],
                    "is_trading": True,
                    "is_st": False,
                    "lot_size": 100,
                }
            )

        return_history[day_index] = daily_return
        close = new_close

    return pd.DataFrame(rows)
