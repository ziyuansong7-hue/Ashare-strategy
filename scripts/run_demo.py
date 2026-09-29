from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ashare_quant.config import AppConfig, ModelConfig, PortfolioConfig, ResearchConfig
from ashare_quant.data.schema import normalize_and_validate_bars
from ashare_quant.data.synthetic import generate_synthetic_bars
from ashare_quant.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deterministic synthetic Strategy V1 engineering demo."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo_v1"))
    parser.add_argument("--symbols", type=int, default=160)
    parser.add_argument("--days", type=int, default=520)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw = generate_synthetic_bars(symbols=args.symbols, days=args.days, seed=args.seed)
    bars, quality = normalize_and_validate_bars(raw)
    config = AppConfig(
        research=replace(
            ResearchConfig(),
            min_cross_section_size=100,
            train_fraction=0.55,
            validation_fraction=0.20,
        ),
        model=replace(
            ModelConfig(),
            n_estimators=120,
            max_depth=3,
            early_stopping_rounds=15,
        ),
        portfolio=replace(
            PortfolioConfig(),
            top_k=20,
            entry_rank=20,
            exit_rank=30,
            max_stock_weight=0.06,
            max_replacements_per_rebalance=4,
            min_replacement_rank_improvement=5,
            rebalance_weight_tolerance=0.01,
        ),
    )
    summary = run_pipeline(
        bars,
        quality,
        config,
        args.output,
        data_context={
            "input_kind": "deterministic_synthetic_demo",
            "evidence_grade": "ENGINEERING_ONLY",
            "seed": args.seed,
            "warning": "Synthetic data validates software behavior, not real investment alpha.",
        },
    )
    result = {
        "output": str(args.output.resolve()),
        "evidence_grade": "ENGINEERING_ONLY",
        "rows": quality.rows,
        "symbols": quality.symbols,
        "xgboost": summary["metrics"]["xgboost"],
        "warning": "Synthetic demo only; do not present its returns as strategy evidence.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
