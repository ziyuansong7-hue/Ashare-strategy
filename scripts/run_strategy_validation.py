from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.analytics import (
    annual_return_table,
    attach_benchmark_nav,
    flatten_backtest_metrics,
    maximum_drawdown_episode,
    parameter_robustness_summary,
    portfolio_regime_attribution,
    rank_ic_summary,
    scale_costs,
    selected_return_summary,
    size_exposure_attribution,
    summarize_backtest,
)
from ashare_quant.analytics.diagnostics import (
    factor_year_report,
    market_regime_report,
    segment_signal_attribution,
)
from ashare_quant.backtest import BacktestMarketContext, prepare_backtest_market, run_backtest
from ashare_quant.config import MODEL_FEATURES, AppConfig, load_config
from ashare_quant.data import load_bars, load_corporate_actions
from ashare_quant.features import compute_raw_factors
from ashare_quant.models import XGBoostCrossSectionalRanker
from ashare_quant.research import build_research_dataset, build_walk_forward_splits

LOGGER = logging.getLogger(__name__)

FACTOR_GROUPS = {
    "momentum": ["momentum_20_5_z", "momentum_60_5_z"],
    "trend": ["trend_quality_60_z", "high_position_120_z"],
    "reversal": ["reversal_5_z"],
    "risk": ["idio_vol_60_z", "downside_vol_60_z"],
    "price_volume": ["volume_z20_z", "price_volume_confirmation_5_z", "clv_5_z"],
}


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _signals(predictions: pd.DataFrame, score_column: str = "xgboost_score") -> pd.DataFrame:
    columns = [
        column
        for column in [
            "date",
            "code",
            "industry",
            "industry_is_point_in_time",
            "size_bucket",
            "tradable",
            "idio_vol_60",
        ]
        if column in predictions
    ]
    signals = predictions[columns].copy()
    signals["score"] = pd.to_numeric(predictions[score_column], errors="raise")
    return signals


def _run(
    factor_frame: pd.DataFrame,
    signals: pd.DataFrame,
    config: AppConfig,
    *,
    end_date: pd.Timestamp,
    corporate_actions: pd.DataFrame,
    market_context: BacktestMarketContext,
    capture_positions: bool = False,
) -> Any:
    return run_backtest(
        factor_frame,
        signals,
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=end_date,
        corporate_actions=corporate_actions,
        market_context=market_context,
        capture_positions=capture_positions,
    )


def _scenario_row(
    profile: str,
    scenario: str,
    config: AppConfig,
    result: Any,
    benchmarks: pd.DataFrame,
) -> dict[str, Any]:
    metrics = summarize_backtest(
        result,
        portfolio=config.portfolio,
        benchmarks=benchmarks,
    )
    return {
        "profile": profile,
        "scenario": scenario,
        "initial_cash": config.portfolio.initial_cash,
        "top_k": config.portfolio.top_k,
        "entry_rank": config.portfolio.entry_rank,
        "exit_rank": config.portfolio.exit_rank,
        "max_replacements": config.portfolio.max_replacements_per_rebalance,
        "min_rank_improvement": config.portfolio.min_replacement_rank_improvement,
        "rebalance_weight_tolerance": config.portfolio.rebalance_weight_tolerance,
        **flatten_backtest_metrics(metrics),
    }


def _write_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def run_parameter_stability(
    factor_frame: pd.DataFrame,
    signals: pd.DataFrame,
    configs: dict[str, AppConfig],
    *,
    end_date: pd.Timestamp,
    corporate_actions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    output: Path,
    market_context: BacktestMarketContext,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    definitions = {
        "institution": {
            "exit_rank": [100, 80, 120],
            "max_replacements": [10, 5, 15],
            "rebalance_band": [0.005, 0.003, 0.010],
        },
        "personal": {
            "exit_rank": [12, 10, 15],
            "max_replacements": [1],
            "rebalance_band": [0.030, 0.020, 0.040],
        },
    }
    rows: list[dict[str, Any]] = []
    base_results: dict[str, Any] = {}
    total = sum(
        len(values["exit_rank"])
        * len(values["max_replacements"])
        * len(values["rebalance_band"])
        for values in definitions.values()
    )
    completed = 0
    for profile, values in definitions.items():
        base = configs[profile]
        for exit_rank, replacements, band in itertools.product(
            values["exit_rank"],
            values["max_replacements"],
            values["rebalance_band"],
        ):
            completed += 1
            scenario = f"exit{exit_rank}_replace{replacements}_band{band:.3f}"
            LOGGER.info("Parameter stability %s/%s: %s %s", completed, total, profile, scenario)
            portfolio = replace(
                base.portfolio,
                exit_rank=exit_rank,
                max_replacements_per_rebalance=replacements,
                rebalance_weight_tolerance=band,
            )
            config = replace(base, portfolio=portfolio)
            result = _run(
                factor_frame,
                signals,
                config,
                end_date=end_date,
                corporate_actions=corporate_actions,
                market_context=market_context,
                capture_positions=(config.portfolio == base.portfolio),
            )
            rows.append(_scenario_row(profile, scenario, config, result, benchmarks))
            if config.portfolio == base.portfolio:
                base_results[profile] = result
            _write_checkpoint(rows, output / "parameter_stability_results.csv")
    results = pd.DataFrame(rows)
    parameter_robustness_summary(results).to_csv(
        output / "parameter_stability_summary.csv", index=False
    )
    return results, base_results


def run_cost_stress(
    factor_frame: pd.DataFrame,
    signals: pd.DataFrame,
    configs: dict[str, AppConfig],
    base_results: dict[str, Any],
    *,
    end_date: pd.Timestamp,
    corporate_actions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    output: Path,
    market_context: BacktestMarketContext,
) -> pd.DataFrame:
    rows = []
    for profile, base in configs.items():
        for multiplier in [0.0, 1.0, 2.0, 3.0]:
            LOGGER.info("Cost stress: %s %.0fx", profile, multiplier)
            config = replace(base, costs=scale_costs(base.costs, multiplier))
            result = base_results[profile] if multiplier == 1.0 else _run(
                factor_frame,
                signals,
                config,
                end_date=end_date,
                corporate_actions=corporate_actions,
                market_context=market_context,
            )
            row = _scenario_row(profile, f"cost_{multiplier:.0f}x", config, result, benchmarks)
            row["cost_multiplier"] = multiplier
            rows.append(row)
            _write_checkpoint(rows, output / "cost_stress_results.csv")
    return pd.DataFrame(rows)


def _train_ablation_predictions(
    dataset: pd.DataFrame,
    config: AppConfig,
    *,
    features: list[str],
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_walk_forward_splits(
        dataset,
        train_months=config.research.walk_forward_train_months,
        validation_months=config.research.walk_forward_validation_months,
        test_months=config.research.walk_forward_test_months,
        step_months=config.research.walk_forward_step_months,
        window=config.research.walk_forward_window,
    )
    predictions = []
    fold_rows = []
    for fold in folds:
        LOGGER.info("Ablation %s: training fold %s/%s", scenario, fold.fold_id, len(folds))
        model = XGBoostCrossSectionalRanker(config.model, features=features).fit(
            fold.train, fold.validation
        )
        test = fold.test.copy()
        test["xgboost_score"] = model.predict(test)
        test["fold_id"] = fold.fold_id
        predictions.append(test)
        fold_rows.append(
            {
                "scenario": scenario,
                "fold_id": fold.fold_id,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "test_rows": len(test),
                "features": "|".join(features),
                "best_iteration": int(model.model.best_iteration),
            }
        )
    combined = pd.concat(predictions, ignore_index=True).sort_values(
        ["date", "code"], kind="stable"
    )
    return combined, pd.DataFrame(fold_rows)


def run_factor_ablation(
    factor_frame: pd.DataFrame,
    full_predictions: pd.DataFrame,
    configs: dict[str, AppConfig],
    base_results: dict[str, Any],
    *,
    end_date: pd.Timestamp,
    corporate_actions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    output: Path,
    market_context: BacktestMarketContext,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = configs["institution"]
    LOGGER.info("Building the shared purged walk-forward dataset for factor ablation")
    dataset = build_research_dataset(
        factor_frame,
        horizon_days=config.research.horizon_days,
        winsor_lower=config.research.winsor_lower,
        winsor_upper=config.research.winsor_upper,
    )
    model_rows = []
    portfolio_rows = []
    fold_frames = []
    prediction_sets: dict[str, pd.DataFrame] = {"full_model": full_predictions}
    for group, removed in FACTOR_GROUPS.items():
        scenario = f"remove_{group}"
        features = [feature for feature in MODEL_FEATURES if feature not in removed]
        predictions, folds = _train_ablation_predictions(
            dataset,
            config,
            features=features,
            scenario=scenario,
        )
        prediction_sets[scenario] = predictions
        fold_frames.append(folds)
        reduced_columns = [
            column
            for column in [
                "date",
                "code",
                "label_end_date",
                "forward_return",
                "target_excess_return",
                "industry",
                "industry_is_point_in_time",
                "size_bucket",
                "tradable",
                "xgboost_score",
                "fold_id",
            ]
            if column in predictions
        ]
        predictions[reduced_columns].to_parquet(
            output / f"ablation_predictions_{scenario}.parquet", index=False
        )

    for scenario, predictions in prediction_sets.items():
        removed_group = scenario.removeprefix("remove_") if scenario != "full_model" else "none"
        model_summary = {
            "scenario": scenario,
            "removed_group": removed_group,
            "remaining_features": len(MODEL_FEATURES)
            if scenario == "full_model"
            else len(MODEL_FEATURES) - len(FACTOR_GROUPS[removed_group]),
            **rank_ic_summary(predictions, "xgboost_score"),
        }
        for profile, profile_config in configs.items():
            model_summary.update(
                {
                    f"{profile}_{key}": value
                    for key, value in selected_return_summary(
                        predictions,
                        "xgboost_score",
                        top_k=profile_config.portfolio.top_k,
                    ).items()
                }
            )
        model_rows.append(model_summary)

        signals = _signals(predictions)
        for profile, profile_config in configs.items():
            LOGGER.info("Ablation portfolio backtest: %s %s", scenario, profile)
            if scenario == "full_model":
                result = base_results[profile]
            else:
                result = _run(
                    factor_frame,
                    signals,
                    profile_config,
                    end_date=end_date,
                    corporate_actions=corporate_actions,
                    market_context=market_context,
                )
            row = _scenario_row(profile, scenario, profile_config, result, benchmarks)
            row["removed_group"] = removed_group
            portfolio_rows.append(row)
            _write_checkpoint(portfolio_rows, output / "factor_ablation_portfolio_results.csv")

    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            output / "factor_ablation_folds.csv", index=False
        )
    model_results = pd.DataFrame(model_rows)
    portfolio_results = pd.DataFrame(portfolio_rows)
    model_results.to_csv(output / "factor_ablation_model_results.csv", index=False)
    portfolio_results.to_csv(output / "factor_ablation_portfolio_results.csv", index=False)
    return model_results, portfolio_results


def write_risk_return_attribution(
    full_predictions: pd.DataFrame,
    configs: dict[str, AppConfig],
    base_results: dict[str, Any],
    benchmarks: pd.DataFrame,
    output: Path,
) -> None:
    annual_frames = []
    regime_frames = []
    size_frames = []
    drawdown_rows = []
    execution_rows = []
    rejection_rows = []
    for profile, config in configs.items():
        result = base_results[profile]
        nav = attach_benchmark_nav(
            result.nav, benchmarks, initial_nav=config.portfolio.initial_cash
        )
        annual = annual_return_table(nav, initial_nav=config.portfolio.initial_cash)
        annual.insert(0, "profile", profile)
        annual_frames.append(annual)
        regimes = portfolio_regime_attribution(
            result.nav,
            benchmarks,
            initial_cash=config.portfolio.initial_cash,
        )
        regimes.insert(0, "profile", profile)
        regime_frames.append(regimes)
        sizes = size_exposure_attribution(result)
        sizes.insert(0, "profile", profile)
        size_frames.append(sizes)
        drawdown_rows.append(
            {"profile": profile, **maximum_drawdown_episode(
                result.nav, initial_cash=config.portfolio.initial_cash
            )}
        )
        metrics = summarize_backtest(
            result,
            portfolio=config.portfolio,
            benchmarks=benchmarks,
        )
        execution_rows.append(
            {
                "profile": profile,
                **{
                    key: metrics[key]
                    for key in [
                        "average_turnover",
                        "median_turnover",
                        "total_fees",
                        "fees_as_initial_cash",
                        "filled_orders",
                        "rejected_orders",
                        "average_entries_after_initial",
                        "average_exits_after_initial",
                        "average_cash_weight",
                        "maximum_cash_weight",
                    ]
                },
            }
        )
        if not result.orders.empty:
            rejected = result.orders.loc[result.orders["filled_quantity"] == 0]
            for reason, count in rejected["reason"].fillna("UNKNOWN").value_counts().items():
                rejection_rows.append(
                    {"profile": profile, "reason": str(reason), "count": int(count)}
                )

    pd.concat(annual_frames, ignore_index=True).to_csv(
        output / "attribution_annual_returns.csv", index=False
    )
    pd.concat(regime_frames, ignore_index=True).to_csv(
        output / "attribution_market_regimes.csv", index=False
    )
    pd.concat(size_frames, ignore_index=True).to_csv(
        output / "attribution_size_exposure.csv", index=False
    )
    pd.DataFrame(drawdown_rows).to_csv(output / "attribution_drawdowns.csv", index=False)
    pd.DataFrame(execution_rows).to_csv(output / "attribution_execution.csv", index=False)
    pd.DataFrame(rejection_rows).to_csv(output / "attribution_rejections.csv", index=False)

    signal_regimes = []
    signal_segments = []
    for profile, config in configs.items():
        regimes = market_regime_report(
            full_predictions,
            benchmarks,
            ["xgboost_score"],
            top_k=config.portfolio.top_k,
        )
        regimes.insert(0, "profile", profile)
        signal_regimes.append(regimes)
        segments = segment_signal_attribution(
            full_predictions,
            ["xgboost_score"],
            top_k=config.portfolio.top_k,
        )
        segments.insert(0, "profile", profile)
        signal_segments.append(segments)
    pd.concat(signal_regimes, ignore_index=True).to_csv(
        output / "attribution_signal_regimes.csv", index=False
    )
    pd.concat(signal_segments, ignore_index=True).to_csv(
        output / "attribution_signal_size_segments.csv", index=False
    )
    factor_year_report(full_predictions).to_csv(
        output / "attribution_factor_year_rank_ic.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run formal low-turnover robustness, cost, ablation, and attribution tests."
    )
    parser.add_argument("--data", type=Path, default=Path("data/csi800_pit_market"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/csi800_pit_walk_forward/walk_forward_predictions.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/strategy_validation")
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    configs = {
        "institution": load_config("configs/low_turnover_institution.toml"),
        "personal": load_config("configs/low_turnover_personal.toml"),
    }
    full_predictions = pd.read_csv(args.predictions, dtype={"code": "string"})
    full_predictions["code"] = full_predictions["code"].astype(str).str.zfill(6)
    for column in ["date", "label_end_date"]:
        full_predictions[column] = pd.to_datetime(full_predictions[column]).dt.normalize()
    end_date = pd.Timestamp(full_predictions["label_end_date"].max())

    bars, quality = load_bars(args.data)
    LOGGER.info("Computing the shared causal factor and execution frame")
    factor_frame = compute_raw_factors(
        bars,
        min_history_days=configs["institution"].research.min_history_days,
        min_liquidity_percentile=configs["institution"].research.min_liquidity_percentile,
    )
    benchmarks = pd.read_parquet(args.data / "benchmarks.parquet")
    corporate_actions = load_corporate_actions(args.data)
    signals = _signals(full_predictions)
    LOGGER.info("Preparing one shared indexed market context for all backtests")
    market_context = prepare_backtest_market(factor_frame)

    parameter_results, base_results = run_parameter_stability(
        factor_frame,
        signals,
        configs,
        end_date=end_date,
        corporate_actions=corporate_actions,
        benchmarks=benchmarks,
        output=args.output,
        market_context=market_context,
    )
    cost_results = run_cost_stress(
        factor_frame,
        signals,
        configs,
        base_results,
        end_date=end_date,
        corporate_actions=corporate_actions,
        benchmarks=benchmarks,
        output=args.output,
        market_context=market_context,
    )
    ablation_models, ablation_portfolios = run_factor_ablation(
        factor_frame,
        full_predictions,
        configs,
        base_results,
        end_date=end_date,
        corporate_actions=corporate_actions,
        benchmarks=benchmarks,
        output=args.output,
        market_context=market_context,
    )
    write_risk_return_attribution(
        full_predictions,
        configs,
        base_results,
        benchmarks,
        args.output,
    )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "fixed_oos_portfolio_robustness_plus_purged_walk_forward_ablation",
        "data_quality": quality.to_dict(),
        "evidence_grade": "LIMITED_RESEARCH",
        "predictions": str(args.predictions),
        "factor_groups": FACTOR_GROUPS,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "counts": {
            "parameter_scenarios": len(parameter_results),
            "cost_scenarios": len(cost_results),
            "ablation_model_scenarios": len(ablation_models),
            "ablation_portfolio_scenarios": len(ablation_portfolios),
        },
        "warning": (
            "Free data does not provide complete daily historical ST/status or point-in-time "
            "industry classifications; conclusions remain LIMITED_RESEARCH."
        ),
    }
    (args.output / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
