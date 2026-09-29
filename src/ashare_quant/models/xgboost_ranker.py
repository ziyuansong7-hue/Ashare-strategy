from __future__ import annotations

from pathlib import Path

import pandas as pd
from xgboost import DMatrix, XGBRanker

from ashare_quant.config import MODEL_FEATURES, ModelConfig


def _sorted_xy(
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ordered = frame.sort_values(["date", "code"], kind="stable")
    qid = pd.Series(pd.factorize(ordered["date"], sort=True)[0], index=ordered.index)
    # XGBoost ranking objectives expect non-negative integer relevance grades.
    # The continuous target rank is retained separately for research evaluation.
    return ordered[features], ordered["relevance"].astype(int), qid


class XGBoostCrossSectionalRanker:
    def __init__(self, config: ModelConfig, features: list[str] | None = None) -> None:
        self.features = list(features or MODEL_FEATURES)
        if not self.features:
            raise ValueError("At least one model feature is required")
        self.model = XGBRanker(
            objective="rank:pairwise",
            eval_metric="ndcg@10",
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            tree_method="hist",
            random_state=config.random_state,
            early_stopping_rounds=config.early_stopping_rounds,
            n_jobs=-1,
        )

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> XGBoostCrossSectionalRanker:
        x_train, y_train, qid_train = _sorted_xy(train, self.features)
        x_validation, y_validation, qid_validation = _sorted_xy(validation, self.features)
        self.model.fit(
            x_train,
            y_train,
            qid=qid_train,
            eval_set=[(x_validation, y_validation)],
            eval_qid=[qid_validation],
            verbose=False,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        scores = self.model.predict(frame[self.features])
        return pd.Series(scores, index=frame.index, dtype=float)

    def feature_contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return per-row TreeSHAP contributions for local score explanations."""
        matrix = DMatrix(frame[self.features], feature_names=self.features)
        best_iteration = getattr(self.model, "best_iteration", None)
        iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)
        values = self.model.get_booster().predict(
            matrix,
            pred_contribs=True,
            iteration_range=iteration_range,
        )
        return pd.DataFrame(values[:, :-1], index=frame.index, columns=self.features)

    def save(self, path: str | Path) -> None:
        self.model.save_model(Path(path))
