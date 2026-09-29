from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ashare_quant.config import load_config
from ashare_quant.data import load_bars, load_corporate_actions
from ashare_quant.features import compute_raw_factors
from ashare_quant.pipeline import run_saved_prediction_backtest

LOGGER = logging.getLogger(__name__)


def _summary_row(name: str, metadata: dict[str, object]) -> dict[str, object]:
    metrics = metadata["metrics"]
    diagnostics = metadata["diagnostics"]["xgboost"]
    config = metadata["config"]["portfolio"]
    return {
        "scenario": name,
        "initial_cash": config["initial_cash"],
        "holdings": config["top_k"],
        "end_nav": metrics["end_nav"],
        "total_return": metrics["total_return"],
        "annualized_return": metrics["annualized_return"],
        "annualized_volatility": metrics["annualized_volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "average_rebalance_turnover": metrics["average_rebalance_turnover"],
        "total_fees": metrics["total_fees"],
        "filled_orders": metrics["filled_orders"],
        "rejected_orders": metrics["rejected_orders"],
        "average_entries": diagnostics["average_entries"],
        "average_exits": diagnostics["average_exits"],
        "csi300_total_return": metrics["benchmarks"]["csi300"]["total_return"],
        "csi300_excess_total_return": metrics["benchmarks"]["csi300"][
            "excess_total_return"
        ],
        "sse_total_return": metrics["benchmarks"]["sse_composite"]["total_return"],
        "sse_excess_total_return": metrics["benchmarks"]["sse_composite"][
            "excess_total_return"
        ],
        "szse_total_return": metrics["benchmarks"]["szse_component"]["total_return"],
        "szse_excess_total_return": metrics["benchmarks"]["szse_component"][
            "excess_total_return"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run institutional and personal low-turnover portfolios on frozen OOS scores."
    )
    parser.add_argument("--data", type=Path, default=Path("data/csi800_pit_market"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/csi800_pit_walk_forward/walk_forward_predictions.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/low_turnover"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictions = pd.read_csv(args.predictions, dtype={"code": "string"})
    predictions["code"] = predictions["code"].astype(str).str.zfill(6)
    bars, _ = load_bars(args.data)
    LOGGER.info("Computing the shared causal factor/execution frame once")
    factor_frame = compute_raw_factors(bars)
    benchmarks = pd.read_parquet(args.data / "benchmarks.parquet")
    corporate_actions = load_corporate_actions(args.data)

    scenarios = {
        "institution": Path("configs/low_turnover_institution.toml"),
        "personal": Path("configs/low_turnover_personal.toml"),
    }
    rows = []
    for name, config_path in scenarios.items():
        LOGGER.info("Running %s low-turnover backtest", name)
        metadata = run_saved_prediction_backtest(
            factor_frame,
            predictions,
            load_config(config_path),
            args.output / name,
            benchmarks=benchmarks,
            corporate_actions=corporate_actions,
            source_predictions=str(args.predictions),
        )
        rows.append(_summary_row(name, metadata))

    args.output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "comparison.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
