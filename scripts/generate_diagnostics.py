from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ashare_quant.analytics import write_research_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate research diagnostics from saved predictions and ledgers."
    )
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="walk_forward")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    args = parser.parse_args()

    predictions_path = args.artifacts / f"{args.prefix}_predictions.csv"
    predictions = pd.read_csv(predictions_path, dtype={"code": "string"})
    benchmarks = pd.read_parquet(args.data_dir / "benchmarks.parquet")
    results = {}
    for model in ["baseline", "xgboost"]:
        stem = f"{args.prefix}_{model}"
        results[model] = SimpleNamespace(
            nav=pd.read_csv(args.artifacts / f"{stem}_nav.csv"),
            orders=pd.read_csv(
                args.artifacts / f"{stem}_orders.csv", dtype={"code": "string"}
            ),
            rebalance_summary=pd.read_csv(args.artifacts / f"{stem}_rebalances.csv"),
        )

    summary = write_research_diagnostics(
        predictions,
        results,
        args.artifacts,
        benchmarks=benchmarks,
        top_k=args.top_k,
        initial_cash=args.initial_cash,
        prefix=args.prefix,
    )
    print(pd.DataFrame(summary).T.to_string())


if __name__ == "__main__":
    main()
