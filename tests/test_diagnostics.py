from types import SimpleNamespace

import pandas as pd

from ashare_quant.analytics.diagnostics import (
    factor_year_report,
    market_regime_report,
    replacement_report,
    selection_report,
    write_research_diagnostics,
)
from ashare_quant.config import MODEL_FEATURES


def _predictions() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.to_datetime(["2024-01-31", "2024-02-29"])):
        for rank in range(4):
            row = {
                "date": date,
                "code": f"00000{rank + 1}",
                "size_bucket": "LARGE" if rank < 2 else "MID",
                "forward_return": (rank + 1) / 100,
                "target_excess_return": (rank - 1.5) / 100,
                "baseline_score": float(rank),
                "xgboost_score": float(rank if date_index == 0 else 3 - rank),
            }
            row.update({feature: float(rank) for feature in MODEL_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_diagnostic_tables_measure_selection_regime_and_replacements():
    predictions = _predictions()
    selected = selection_report(
        predictions, ["baseline_score", "xgboost_score"], top_k=2
    )
    all_baseline = selected.loc[
        (selected["period"] == "ALL") & (selected["model"] == "baseline")
    ].set_index("group")
    assert (
        all_baseline.loc["selected", "mean_excess_return"]
        > all_baseline.loc["not_selected", "mean_excess_return"]
    )

    replacements = replacement_report(predictions, ["xgboost_score"], top_k=2)
    assert len(replacements) == 1
    assert replacements.iloc[0]["entrants"] == 2

    benchmark_dates = pd.bdate_range("2023-07-01", "2024-02-29")
    benchmarks = pd.DataFrame(
        {
            "date": benchmark_dates,
            "benchmark_key": "csi800",
            "close": range(100, 100 + len(benchmark_dates)),
        }
    )
    benchmarks["date"] = benchmarks["date"].astype("datetime64[ms]")
    regimes = market_regime_report(
        predictions,
        benchmarks,
        ["xgboost_score"],
        top_k=2,
        lookback_days=5,
        threshold=0.001,
    )
    assert set(regimes["regime"]) == {"BULL"}
    assert not factor_year_report(predictions).empty


def test_diagnostic_pack_writes_reusable_outputs(tmp_path):
    predictions = _predictions()
    dates = pd.bdate_range("2024-02-01", periods=3)
    result = SimpleNamespace(
        nav=pd.DataFrame(
            {
                "date": dates,
                "cash": [10.0, 10.0, 10.0],
                "nav": [100.0, 101.0, 102.0],
                "universe_equal_weight_nav": [100.0, 100.5, 101.0],
            }
        ),
        orders=pd.DataFrame(
            {
                "filled_quantity": [100, 0],
                "fees": [1.0, 0.0],
                "reason": [None, "LIMIT_LOCKED"],
            }
        ),
        rebalance_summary=pd.DataFrame(
            {"turnover": [0.5], "entered_count": [2], "exited_count": [1]}
        ),
    )

    summary = write_research_diagnostics(
        predictions,
        {"baseline": result, "xgboost": result},
        tmp_path,
        benchmarks=None,
        top_k=2,
        initial_cash=100.0,
        prefix="test",
    )

    assert summary["xgboost"]["rejected_orders"] == 1
    assert (tmp_path / "test_factor_year_report.csv").exists()
    assert (tmp_path / "test_execution_diagnostics.csv").exists()
    assert (tmp_path / "test_diagnostic_summary.json").exists()
