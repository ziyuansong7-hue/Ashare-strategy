import numpy as np
import pandas as pd

from ashare_quant.analytics import annual_return_table, attach_benchmark_nav, performance_metrics


def test_external_benchmarks_are_aligned_without_future_fill():
    dates = pd.bdate_range("2024-01-02", periods=3)
    nav = pd.DataFrame(
        {
            "date": dates,
            "nav": [100.0, 110.0, 121.0],
            "universe_equal_weight_nav": [100.0, 105.0, 110.25],
        }
    )
    benchmarks = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), *dates],
            "benchmark_key": ["sse_composite"] * 4,
            "close": [100.0, 110.0, 121.0, 133.1],
        }
    )

    combined = attach_benchmark_nav(nav, benchmarks, initial_nav=100.0)

    assert np.allclose(combined["sse_composite_nav"], [110.0, 121.0, 133.1])
    metrics = performance_metrics(combined, initial_nav=100.0)
    assert set(metrics["benchmarks"]) == {"sse_composite", "universe_equal_weight"}
    assert metrics["benchmarks"]["sse_composite"]["beta"] == 0.0
    assert np.isclose(metrics["benchmarks"]["sse_composite"]["excess_total_return"], -0.121)


def test_annual_return_table_contains_strategy_and_all_benchmarks():
    nav = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-01-02"]),
            "nav": [110.0, 121.0],
            "sse_composite_nav": [105.0, 107.1],
        }
    )

    table = annual_return_table(nav, initial_nav=100.0)

    assert table["year"].tolist() == [2023, 2024]
    assert {"nav", "sse_composite"}.issubset(table.columns)
