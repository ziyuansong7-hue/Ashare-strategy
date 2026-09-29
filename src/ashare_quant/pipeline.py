from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.analytics import (
    annual_return_table,
    attach_benchmark_nav,
    factor_report,
    performance_metrics,
    segment_model_report,
    write_research_diagnostics,
)
from ashare_quant.backtest import run_backtest
from ashare_quant.config import BASELINE_WEIGHTS, MODEL_FEATURES, RAW_FEATURES, AppConfig
from ashare_quant.data.schema import DataQualityReport
from ashare_quant.features import compute_raw_factors, cross_sectional_preprocess
from ashare_quant.models import LinearFactorRanker, XGBoostCrossSectionalRanker
from ashare_quant.portfolio import validate_portfolio_universe, validate_research_universe
from ashare_quant.research import build_research_dataset, build_walk_forward_splits, split_dataset

LOGGER = logging.getLogger(__name__)


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _save_backtest(
    name: str,
    result,
    output: Path,
    *,
    initial_cash: float,
    benchmarks: pd.DataFrame | None,
) -> dict[str, object]:
    nav = attach_benchmark_nav(result.nav, benchmarks, initial_nav=initial_cash)
    nav.to_csv(output / f"{name}_nav.csv", index=False)
    annual_return_table(nav, initial_nav=initial_cash).to_csv(
        output / f"{name}_annual_returns.csv", index=False
    )
    result.orders.to_csv(output / f"{name}_orders.csv", index=False)
    result.positions.to_csv(output / f"{name}_positions.csv", index=False)
    result.rebalance_summary.to_csv(output / f"{name}_rebalances.csv", index=False)
    result.corporate_actions.to_csv(output / f"{name}_corporate_actions.csv", index=False)
    metrics = performance_metrics(nav, initial_nav=initial_cash)
    if not result.orders.empty:
        metrics["filled_orders"] = int((result.orders["filled_quantity"] > 0).sum())
        metrics["rejected_orders"] = int((result.orders["filled_quantity"] == 0).sum())
        metrics["total_fees"] = float(result.orders["fees"].sum())
    if not result.rebalance_summary.empty:
        metrics["average_rebalance_turnover"] = float(result.rebalance_summary["turnover"].mean())
    return metrics


def run_saved_prediction_backtest(
    factor_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    config: AppConfig,
    output_dir: str | Path,
    *,
    benchmarks: pd.DataFrame | None = None,
    corporate_actions: pd.DataFrame | None = None,
    score_column: str = "xgboost_score",
    source_predictions: str | None = None,
) -> dict[str, object]:
    """Re-run portfolio construction without retraining an already frozen OOS model.

    This is the correct comparison path for portfolio-policy experiments: every scenario receives
    the exact same point-in-time test rows and model scores, so performance changes can only come
    from portfolio construction, capital, lot sizes, or execution costs.
    """
    if score_column not in predictions:
        raise ValueError(f"Predictions are missing {score_column!r}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    for column in ["date", "label_end_date"]:
        if column in predictions:
            predictions[column] = pd.to_datetime(predictions[column]).dt.normalize()
    validate_portfolio_universe(predictions, config.portfolio)
    final_label_date = pd.Timestamp(predictions["label_end_date"].max())
    result = run_backtest(
        factor_frame,
        predictions.rename(columns={score_column: "score"}),
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=final_label_date,
        corporate_actions=corporate_actions,
    )
    model_name = score_column.removesuffix("_score")
    metrics = _save_backtest(
        model_name,
        result,
        output,
        initial_cash=config.portfolio.initial_cash,
        benchmarks=benchmarks,
    )
    comparison_rows = [
        {"model": model_name, "benchmark": benchmark, **values}
        for benchmark, values in metrics.get("benchmarks", {}).items()
    ]
    pd.DataFrame(comparison_rows).to_csv(output / "benchmark_comparison.csv", index=False)
    diagnostics = write_research_diagnostics(
        predictions,
        {model_name: result},
        output,
        benchmarks=benchmarks,
        top_k=config.portfolio.top_k,
        initial_cash=config.portfolio.initial_cash,
        prefix="low_turnover",
    )
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "saved_out_of_sample_predictions_portfolio_rerun",
        "score_column": score_column,
        "source_predictions": source_predictions,
        "config": asdict(config),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "warning": "LIMITED_RESEARCH: free data lacks complete daily historical ST/status fields.",
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return metadata


def run_pipeline(
    bars: pd.DataFrame,
    quality_report: DataQualityReport,
    config: AppConfig,
    output_dir: str | Path,
    benchmarks: pd.DataFrame | None = None,
    data_context: dict[str, object] | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Computing causal price-volume factors")
    factor_frame = compute_raw_factors(
        bars,
        min_history_days=config.research.min_history_days,
        min_liquidity_percentile=config.research.min_liquidity_percentile,
    )
    LOGGER.info("Building point-in-time monthly research dataset")
    dataset = build_research_dataset(
        factor_frame,
        horizon_days=config.research.horizon_days,
        winsor_lower=config.research.winsor_lower,
        winsor_upper=config.research.winsor_upper,
    )
    split = split_dataset(
        dataset,
        train_fraction=config.research.train_fraction,
        validation_fraction=config.research.validation_fraction,
    )
    validate_research_universe(dataset, config.research.min_cross_section_size)
    validate_portfolio_universe(split.test, config.portfolio)

    LOGGER.info("Training transparent linear-factor benchmark")
    baseline = LinearFactorRanker().fit(split.train)
    LOGGER.info("Training XGBoost cross-sectional ranker")
    xgboost_model = XGBoostCrossSectionalRanker(config.model).fit(split.train, split.validation)

    test = split.test.copy()
    test["baseline_score"] = baseline.predict(test)
    test["xgboost_score"] = xgboost_model.predict(test)
    test.to_csv(output / "test_predictions.csv", index=False)
    segment_model_report(test, ["baseline_score", "xgboost_score"]).to_csv(
        output / "segment_model_report.csv", index=False
    )

    report = factor_report(pd.concat([split.train, split.validation], ignore_index=True))
    report.to_csv(output / "factor_report.csv", index=False)
    baseline.save(output / "baseline_weights.json")
    xgboost_model.save(output / "xgboost_ranker.json")

    latest_date = pd.Timestamp(factor_frame["date"].max())
    latest = factor_frame.loc[(factor_frame["date"] == latest_date) & factor_frame["tradable"]].copy()
    latest = cross_sectional_preprocess(
        latest,
        lower_quantile=config.research.winsor_lower,
        upper_quantile=config.research.winsor_upper,
    ).dropna(subset=MODEL_FEATURES)
    latest["baseline_score"] = baseline.predict(latest)
    latest["xgboost_score"] = xgboost_model.predict(latest)
    latest["baseline_rank"] = latest["baseline_score"].rank(ascending=False, method="first")
    latest["xgboost_rank"] = latest["xgboost_score"].rank(ascending=False, method="first")
    contribution_columns = []
    for feature, weight in BASELINE_WEIGHTS.items():
        contribution = f"{feature}_contribution"
        latest[contribution] = latest[feature] * weight
        contribution_columns.append(contribution)
    reason_labels = {f"{feature}_contribution": feature.removesuffix("_z") for feature in BASELINE_WEIGHTS}
    latest["baseline_reason_1"] = latest[contribution_columns].idxmax(axis=1).map(reason_labels)
    latest["baseline_reason_2"] = latest[contribution_columns].apply(
        lambda row: reason_labels[row.nlargest(2).index[-1]], axis=1
    )
    xgb_contributions = xgboost_model.feature_contributions(latest)
    xgb_labels = {feature: feature.removesuffix("_z") for feature in MODEL_FEATURES}
    latest["xgboost_reason_1"] = xgb_contributions.idxmax(axis=1).map(xgb_labels)
    latest["xgboost_reason_2"] = xgb_contributions.apply(
        lambda row: xgb_labels[row.nlargest(2).index[-1]], axis=1
    )
    identity_columns = [
        column for column in ["security_name", "size_bucket"] if column in latest
    ]
    latest[
        [
            "date",
            "code",
            *identity_columns,
            "industry",
            *RAW_FEATURES,
            "baseline_score",
            "baseline_rank",
            "xgboost_score",
            "xgboost_rank",
            "baseline_reason_1",
            "baseline_reason_2",
            "xgboost_reason_1",
            "xgboost_reason_2",
        ]
    ].sort_values("xgboost_rank").to_csv(output / "latest_scores.csv", index=False)

    final_label_date = pd.Timestamp(test["label_end_date"].max())
    baseline_signals = test.rename(columns={"baseline_score": "score"})
    xgboost_signals = test.rename(columns={"xgboost_score": "score"})

    LOGGER.info("Backtesting baseline and XGBoost strategies with execution constraints")
    baseline_result = run_backtest(
        factor_frame,
        baseline_signals,
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=final_label_date,
        corporate_actions=corporate_actions,
    )
    xgboost_result = run_backtest(
        factor_frame,
        xgboost_signals,
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=final_label_date,
        corporate_actions=corporate_actions,
    )

    metrics = {
        "baseline": _save_backtest(
            "baseline",
            baseline_result,
            output,
            initial_cash=config.portfolio.initial_cash,
            benchmarks=benchmarks,
        ),
        "xgboost": _save_backtest(
            "xgboost",
            xgboost_result,
            output,
            initial_cash=config.portfolio.initial_cash,
            benchmarks=benchmarks,
        ),
    }
    comparison_rows = []
    for model_name, model_metrics in metrics.items():
        for benchmark_name, benchmark_metrics in model_metrics["benchmarks"].items():
            comparison_rows.append(
                {"model": model_name, "benchmark": benchmark_name, **benchmark_metrics}
            )
    pd.DataFrame(comparison_rows).to_csv(output / "benchmark_comparison.csv", index=False)
    diagnostics = write_research_diagnostics(
        test,
        {"baseline": baseline_result, "xgboost": xgboost_result},
        output,
        benchmarks=benchmarks,
        top_k=config.portfolio.top_k,
        initial_cash=config.portfolio.initial_cash,
        prefix="diagnostic",
    )
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_quality": quality_report.to_dict(),
        "config": asdict(config),
        "dataset": {
            "rows": len(dataset),
            "rebalance_dates": int(dataset["date"].nunique()),
            "train_rows": len(split.train),
            "validation_rows": len(split.validation),
            "test_rows": len(split.test),
            "validation_start": split.validation_start,
            "test_start": split.test_start,
            "latest_scored_date": latest_date,
            "latest_scored_symbols": len(latest),
            "latest_size_bucket_counts": (
                latest["size_bucket"].value_counts().to_dict()
                if "size_bucket" in latest
                else {}
            ),
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "external_benchmarks": sorted(benchmarks["benchmark_key"].unique().tolist())
        if benchmarks is not None and not benchmarks.empty
        else [],
        "data_context": data_context or {},
        "warning": "Performance is a research result, not an investment guarantee.",
    }
    (output / "run_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return metadata


def _rank_ic_summary(frame: pd.DataFrame, score_column: str) -> dict[str, float]:
    daily_values = []
    for _, group in frame.groupby("date", sort=True):
        value = group[score_column].corr(group["target_excess_return"], method="spearman")
        if pd.notna(value):
            daily_values.append(float(value))
    if not daily_values:
        return {"rank_ic": float("nan"), "rank_ic_ir": float("nan"), "positive_ic_rate": 0.0}
    values = pd.Series(daily_values, dtype=float)
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "rank_ic": float(values.mean()),
        "rank_ic_ir": float(values.mean() / standard_deviation)
        if standard_deviation > 0
        else float("nan"),
        "positive_ic_rate": float((values > 0).mean()),
    }


def run_walk_forward_pipeline(
    bars: pd.DataFrame,
    quality_report: DataQualityReport,
    config: AppConfig,
    output_dir: str | Path,
    benchmarks: pd.DataFrame | None = None,
    data_context: dict[str, object] | None = None,
    corporate_actions: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Retrain on every fold and backtest only stitched out-of-sample predictions."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "fold_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Computing causal factors for walk-forward research")
    factor_frame = compute_raw_factors(
        bars,
        min_history_days=config.research.min_history_days,
        min_liquidity_percentile=config.research.min_liquidity_percentile,
    )
    dataset = build_research_dataset(
        factor_frame,
        horizon_days=config.research.horizon_days,
        winsor_lower=config.research.winsor_lower,
        winsor_upper=config.research.winsor_upper,
    )
    validate_research_universe(dataset, config.research.min_cross_section_size)
    folds = build_walk_forward_splits(
        dataset,
        train_months=config.research.walk_forward_train_months,
        validation_months=config.research.walk_forward_validation_months,
        test_months=config.research.walk_forward_test_months,
        step_months=config.research.walk_forward_step_months,
        window=config.research.walk_forward_window,
    )

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for fold in folds:
        LOGGER.info(
            "Walk-forward fold %s: train %s..%s, validate %s..%s, test %s..%s",
            fold.fold_id,
            fold.train_start.date(),
            fold.train_end.date(),
            fold.validation_start.date(),
            fold.validation_end.date(),
            fold.test_start.date(),
            fold.test_end.date(),
        )
        baseline = LinearFactorRanker().fit(fold.train)
        xgboost_model = XGBoostCrossSectionalRanker(config.model).fit(
            fold.train, fold.validation
        )
        predictions = fold.test.copy()
        predictions["fold_id"] = fold.fold_id
        predictions["baseline_score"] = baseline.predict(predictions)
        predictions["xgboost_score"] = xgboost_model.predict(predictions)
        prediction_frames.append(predictions)
        xgboost_model.save(model_dir / f"xgboost_fold_{fold.fold_id:02d}.json")

        baseline_ic = _rank_ic_summary(predictions, "baseline_score")
        xgboost_ic = _rank_ic_summary(predictions, "xgboost_score")
        fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "window": config.research.walk_forward_window,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "effective_train_end_after_purge": pd.Timestamp(fold.train["date"].max()),
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "effective_validation_end_after_purge": pd.Timestamp(
                    fold.validation["date"].max()
                ),
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_rows": len(fold.train),
                "validation_rows": len(fold.validation),
                "test_rows": len(fold.test),
                "xgboost_best_iteration": int(xgboost_model.model.best_iteration),
                **{f"baseline_{key}": value for key, value in baseline_ic.items()},
                **{f"xgboost_{key}": value for key, value in xgboost_ic.items()},
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["date", "code"], kind="stable"
    )
    if predictions.duplicated(["date", "code"]).any():
        raise ValueError("Walk-forward test folds produced overlapping date/code predictions")
    validate_portfolio_universe(predictions, config.portfolio)
    predictions.to_csv(output / "walk_forward_predictions.csv", index=False)
    segment_model_report(predictions, ["baseline_score", "xgboost_score"]).to_csv(
        output / "walk_forward_segment_model_report.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(output / "walk_forward_folds.csv", index=False)

    final_label_date = pd.Timestamp(predictions["label_end_date"].max())
    baseline_result = run_backtest(
        factor_frame,
        predictions.rename(columns={"baseline_score": "score"}),
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=final_label_date,
        corporate_actions=corporate_actions,
    )
    xgboost_result = run_backtest(
        factor_frame,
        predictions.rename(columns={"xgboost_score": "score"}),
        portfolio_config=config.portfolio,
        cost_config=config.costs,
        end_date=final_label_date,
        corporate_actions=corporate_actions,
    )
    metrics = {
        "baseline": _save_backtest(
            "walk_forward_baseline",
            baseline_result,
            output,
            initial_cash=config.portfolio.initial_cash,
            benchmarks=benchmarks,
        ),
        "xgboost": _save_backtest(
            "walk_forward_xgboost",
            xgboost_result,
            output,
            initial_cash=config.portfolio.initial_cash,
            benchmarks=benchmarks,
        ),
    }
    comparison_rows = []
    for model_name, model_metrics in metrics.items():
        for benchmark_name, benchmark_metrics in model_metrics["benchmarks"].items():
            comparison_rows.append(
                {"model": model_name, "benchmark": benchmark_name, **benchmark_metrics}
            )
    pd.DataFrame(comparison_rows).to_csv(
        output / "walk_forward_benchmark_comparison.csv", index=False
    )
    diagnostics = write_research_diagnostics(
        predictions,
        {"baseline": baseline_result, "xgboost": xgboost_result},
        output,
        benchmarks=benchmarks,
        top_k=config.portfolio.top_k,
        initial_cash=config.portfolio.initial_cash,
        prefix="walk_forward",
    )
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "purged_walk_forward",
        "data_quality": quality_report.to_dict(),
        "config": asdict(config),
        "dataset": {
            "rows": len(dataset),
            "rebalance_dates": int(dataset["date"].nunique()),
            "folds": len(folds),
            "out_of_sample_rows": len(predictions),
            "out_of_sample_start": pd.Timestamp(predictions["date"].min()),
            "out_of_sample_end": pd.Timestamp(predictions["date"].max()),
        },
        "folds": fold_rows,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "external_benchmarks": sorted(benchmarks["benchmark_key"].unique().tolist())
        if benchmarks is not None and not benchmarks.empty
        else [],
        "data_context": data_context or {},
        "warning": "Only stitched fold-test predictions are used in reported performance.",
    }
    (output / "walk_forward_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return metadata
