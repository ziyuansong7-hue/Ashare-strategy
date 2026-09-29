from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ashare_quant import __version__
from ashare_quant.config import apply_portfolio_overrides, apply_research_overrides, load_config
from ashare_quant.data import (
    MarketDataSynchronizer,
    SyncConfig,
    audit_market_data,
    load_bars,
    load_corporate_actions,
)
from ashare_quant.data.providers import AKShareProvider, BaoStockStatusProvider
from ashare_quant.data.synthetic import generate_synthetic_bars
from ashare_quant.pipeline import run_pipeline, run_walk_forward_pipeline
from ashare_quant.portfolio import load_positions, plan_orders


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ashare-quant",
        description="Explainable A-share factor research and execution-aware backtesting",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("generate-demo", help="Generate deterministic synthetic OHLCV data")
    demo.add_argument("--output", default="data/demo_bars.csv")
    demo.add_argument("--symbols", type=int, default=300)
    demo.add_argument("--days", type=int, default=900)
    demo.add_argument("--seed", type=int, default=42)

    sync = subparsers.add_parser(
        "sync-data",
        help="Synchronize stock bars, security metadata, membership, and benchmark indices",
    )
    sync.add_argument(
        "--universe",
        choices=["all_a", "csi300", "csi500", "csi800", "csi1000"],
        default="csi800",
    )
    sync.add_argument(
        "--skip-historical-status",
        action="store_true",
        help="Skip BaoStock history; the resulting dataset will not pass research-ready gates",
    )
    sync.add_argument("--membership-file", help="Point-in-time CSV with effective dates")
    sync.add_argument(
        "--industry-file",
        help="Point-in-time CSV/Parquet with code,effective_from,industry_code,industry_name",
    )
    sync.add_argument("--start", default="2015-01-01")
    sync.add_argument("--end", default=datetime.now(UTC).date().isoformat())
    sync.add_argument("--data-dir", default="data/csi800_market")
    sync.add_argument("--workers", type=int, default=4)
    sync.add_argument("--retries", type=int, default=3)
    sync.add_argument("--retry-backoff", type=float, default=1.0)
    sync.add_argument("--timeout", type=float, default=20.0)
    sync.add_argument("--refresh-mode", choices=["missing", "full"], default="missing")
    sync.add_argument("--max-symbols", type=int, help="Development/testing limit")
    sync.add_argument(
        "--allow-current-universe-backfill",
        action="store_true",
        help="Acknowledge survivorship bias when current constituents are used for past dates",
    )

    audit = subparsers.add_parser("audit-data", help="Audit a synchronized market-data directory")
    audit.add_argument("--data-dir", default="data/csi800_market")
    audit.add_argument("--min-rows", type=int, default=120)

    run = subparsers.add_parser("run", help="Run factor research, model training, and backtesting")
    run.add_argument("--data", required=True, help="CSV/Parquet file or directory")
    run.add_argument("--config", default="configs/default.toml")
    run.add_argument("--output", default="artifacts/run")
    run.add_argument(
        "--benchmarks",
        help="Benchmark Parquet; defaults to <data>/benchmarks.parquet for a synchronized directory",
    )
    run.add_argument(
        "--allow-non-research-ready",
        action="store_true",
        help="Explicitly allow a labelled limited-research or engineering run",
    )
    run.add_argument(
        "--holdings",
        type=int,
        help="Target number of stocks, for example 3-5 for a concentrated personal portfolio",
    )
    run.add_argument(
        "--initial-cash",
        type=float,
        help="Override starting capital while preserving the configured transaction-cost model",
    )
    run.add_argument(
        "--weighting",
        choices=["equal", "inverse_volatility", "score_tilt"],
        help="Portfolio weighting method",
    )
    run.add_argument("--cash-buffer", type=float, help="Target cash fraction, e.g. 0.02")
    run.add_argument("--min-trade-value", type=float, help="Ignore non-exit trades below this value")
    run.add_argument("--exit-rank", type=int, help="Rank buffer below which existing holdings exit")
    run.add_argument("--max-replacements", type=int, help="Maximum routine replacements per month")
    run.add_argument(
        "--min-rank-improvement",
        type=int,
        help="Minimum rank advantage required for a replacement",
    )
    run.add_argument(
        "--rebalance-band",
        type=float,
        help="Absolute weight tolerance that suppresses small rebalancing trades",
    )
    run.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run purged rolling/expanding walk-forward evaluation instead of a single split",
    )
    run.add_argument("--wf-window", choices=["expanding", "rolling"])
    run.add_argument("--wf-train-months", type=int)
    run.add_argument("--wf-validation-months", type=int)
    run.add_argument("--wf-test-months", type=int)
    run.add_argument("--wf-step-months", type=int)

    plan = subparsers.add_parser(
        "plan-orders", help="Convert latest model scores and current holdings into indicative orders"
    )
    plan.add_argument("--scores", required=True, help="latest_scores.csv from a research run")
    plan.add_argument("--data", required=True, help="Market-data file or synchronized directory")
    plan.add_argument("--positions", help="CSV with code,quantity; omit for an empty portfolio")
    plan.add_argument("--cash", type=float, required=True)
    plan.add_argument("--config", default="configs/default.toml")
    plan.add_argument("--output", default="artifacts/order_plan")
    plan.add_argument("--model", choices=["xgboost", "baseline"], default="xgboost")
    plan.add_argument("--holdings", type=int, help="Target stock count")
    plan.add_argument(
        "--weighting",
        choices=["equal", "inverse_volatility", "score_tilt"],
    )
    plan.add_argument("--cash-buffer", type=float)
    plan.add_argument("--min-trade-value", type=float)
    plan.add_argument("--exit-rank", type=int)
    plan.add_argument("--max-replacements", type=int)
    plan.add_argument("--min-rank-improvement", type=int)
    plan.add_argument("--rebalance-band", type=float)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args()

    if args.command == "generate-demo":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = generate_synthetic_bars(symbols=args.symbols, days=args.days, seed=args.seed)
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame):,} synthetic rows to {path}")
        print("Synthetic data is for software verification only, not strategy-performance evidence.")
        return

    if args.command == "sync-data":
        try:
            start_date = pd.Timestamp(args.start).normalize()
            end_date = pd.Timestamp(args.end).normalize()
        except ValueError as exc:
            raise SystemExit(f"Invalid --start or --end date: {exc}") from exc
        provider = AKShareProvider(timeout_seconds=args.timeout)
        sync_config = SyncConfig(
            data_dir=Path(args.data_dir),
            start_date=start_date,
            end_date=end_date,
            universe=args.universe,
            membership_file=args.membership_file,
            industry_file=args.industry_file,
            workers=args.workers,
            retries=args.retries,
            retry_backoff_seconds=args.retry_backoff,
            refresh_mode=args.refresh_mode,
            max_symbols=args.max_symbols,
            allow_current_universe_backfill=args.allow_current_universe_backfill,
            fetch_historical_status=not args.skip_historical_status,
        )
        # BaoStock also supplies historical CSI 300/500 membership.  Keep it available when
        # per-stock daily ST/status enrichment is intentionally skipped by the free-data profile.
        status_provider = BaoStockStatusProvider()
        try:
            report = MarketDataSynchronizer(
                provider,
                sync_config,
                status_provider=status_provider,
            ).run()
        finally:
            if status_provider is not None:
                status_provider.close()
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print("Quality gates:")
        print(json.dumps(report["quality_gates"], ensure_ascii=False, indent=2))
        print(f"Evidence grade: {report['evidence_grade']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        print(f"Download report: {(Path(args.data_dir) / 'metadata' / 'download_report.json').resolve()}")
        if report["summary"]["failed_symbols"]:
            raise SystemExit(2)
        return

    if args.command == "plan-orders":
        try:
            config = apply_portfolio_overrides(
                load_config(args.config),
                holdings=args.holdings,
                weighting_method=args.weighting,
                cash_buffer=args.cash_buffer,
                min_trade_value=args.min_trade_value,
                exit_rank=args.exit_rank,
                max_replacements_per_rebalance=args.max_replacements,
                min_replacement_rank_improvement=args.min_rank_improvement,
                rebalance_weight_tolerance=args.rebalance_band,
            )
            bars, _ = load_bars(
                args.data, default_lot_size=config.portfolio.default_lot_size
            )
            scores = pd.read_csv(args.scores, dtype={"code": "string"})
            scores["code"] = scores["code"].astype(str).str.zfill(6)
            positions = load_positions(args.positions)
            orders, plan_summary = plan_orders(
                scores,
                bars,
                positions,
                cash=args.cash,
                portfolio_config=config.portfolio,
                cost_config=config.costs,
                score_column=f"{args.model}_score",
            )
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"Order planning rejected: {exc}") from exc
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        orders.to_csv(output / "planned_orders.csv", index=False)
        (output / "order_plan_summary.json").write_text(
            json.dumps(plan_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(plan_summary, ensure_ascii=False, indent=2))
        print(f"Planned orders: {(output / 'planned_orders.csv').resolve()}")
        return

    if args.command == "audit-data":
        report = audit_market_data(args.data_dir, min_rows_per_symbol=args.min_rows)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        if not report["passed"]:
            raise SystemExit(2)
        return

    try:
        config = apply_portfolio_overrides(
            load_config(args.config),
            holdings=args.holdings,
            initial_cash=args.initial_cash,
            weighting_method=args.weighting,
            cash_buffer=args.cash_buffer,
            min_trade_value=args.min_trade_value,
            exit_rank=args.exit_rank,
            max_replacements_per_rebalance=args.max_replacements,
            min_replacement_rank_improvement=args.min_rank_improvement,
            rebalance_weight_tolerance=args.rebalance_band,
        )
        config = apply_research_overrides(
            config,
            walk_forward_window=args.wf_window,
            train_months=args.wf_train_months,
            validation_months=args.wf_validation_months,
            test_months=args.wf_test_months,
            step_months=args.wf_step_months,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid portfolio configuration: {exc}") from exc
    bars, quality_report = load_bars(
        args.data,
        default_lot_size=config.portfolio.default_lot_size,
    )
    benchmark_path = Path(args.benchmarks) if args.benchmarks else None
    data_path = Path(args.data)
    data_context: dict[str, object] = {}
    download_report_path = data_path / "metadata" / "download_report.json"
    if data_path.is_dir() and download_report_path.exists():
        download_report = json.loads(download_report_path.read_text(encoding="utf-8"))
        data_context = {
            "download_run_id": download_report.get("run_id"),
            "provider": download_report.get("provider"),
            "evidence_grade": download_report.get("evidence_grade", "ENGINEERING_ONLY"),
            "request": download_report.get("request"),
            "quality_gates": download_report.get("quality_gates", {}),
            "warnings": download_report.get("warnings", []),
        }
        research_ready = bool(data_context["quality_gates"].get("research_ready", False))
        if not research_ready and not args.allow_non_research_ready:
            raise SystemExit(
                "Synchronized data is not research-ready. Review metadata/download_report.json, "
                "then use --allow-non-research-ready only for a labelled limited or engineering run."
            )
    else:
        data_context = {
            "input_kind": "standalone_or_unmanifested",
            "quality_gates": {"research_ready": False},
            "warnings": [
                (
                    "No synchronized download_report.json was found; provenance and point-in-time "
                    "coverage are unverified."
                )
            ],
        }
        if not args.allow_non_research_ready:
            raise SystemExit(
                "Input has no synchronized quality manifest. Use a synchronized data directory for "
                "formal research, or --allow-non-research-ready for a labelled engineering/demo run."
            )
    if benchmark_path is None and data_path.is_dir() and (data_path / "benchmarks.parquet").exists():
        benchmark_path = data_path / "benchmarks.parquet"
    benchmarks = pd.read_parquet(benchmark_path) if benchmark_path is not None else None
    corporate_actions = load_corporate_actions(data_path) if data_path.is_dir() else None
    try:
        pipeline = run_walk_forward_pipeline if args.walk_forward else run_pipeline
        summary = pipeline(
            bars,
            quality_report,
            config,
            args.output,
            benchmarks=benchmarks,
            data_context=data_context,
            corporate_actions=corporate_actions,
        )
    except ValueError as exc:
        raise SystemExit(f"Research run rejected: {exc}") from exc
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"Evidence grade: {data_context.get('evidence_grade', 'ENGINEERING_ONLY')}")
    print(f"Full artifacts: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
