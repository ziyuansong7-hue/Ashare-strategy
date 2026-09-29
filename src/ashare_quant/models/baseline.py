from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ashare_quant.config import BASELINE_WEIGHTS


class LinearFactorRanker:
    """Transparent benchmark that combines standardized factors with fixed signs."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or BASELINE_WEIGHTS)
        scale = sum(abs(value) for value in self.weights.values())
        if scale <= 0:
            raise ValueError("At least one non-zero baseline weight is required")
        self.weights = {name: value / scale for name, value in self.weights.items()}

    def fit(self, _: pd.DataFrame) -> LinearFactorRanker:
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        missing = set(self.weights).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing baseline features: {sorted(missing)}")
        score = pd.Series(0.0, index=frame.index, dtype=float)
        for feature, weight in self.weights.items():
            score = score + frame[feature] * weight
        return score

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.weights, indent=2), encoding="utf-8")

