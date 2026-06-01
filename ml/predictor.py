import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .features import FeatureEngine
from .model import MLModel
from utils import logger
from config.settings import Settings


@dataclass
class PredictionResult:
    signal: int
    confidence: float
    features_used: int
    model_ready: bool

    @property
    def is_buy(self) -> bool:
        return self.signal == 1 and self.confidence >= 0.75

    @property
    def is_sell(self) -> bool:
        return self.signal == -1 and self.confidence >= 0.75

    @property
    def is_valid(self) -> bool:
        return self.model_ready and self.confidence >= 0.70


class MLPredictor:
    def __init__(self, settings: Settings):
        self.log = logger
        self.settings = settings
        self.feature_engine = FeatureEngine(window=settings.ml.feature_window)
        self.model = MLModel(
            model_path=settings.ml.model_path,
            use_gpu=settings.ml.use_gpu,
        )
        self._initialized = False

    async def initialize(self):
        if not self.settings.ml.enabled:
            self.log.info("ML module disabled")
            return

        loaded = self.model.load()
        if not loaded:
            self.log.info("No pre-trained model found. Will need training.")
        self._initialized = True

    async def predict(self, df: pd.DataFrame, smc_analysis=None) -> PredictionResult:
        if not self._initialized or not self.settings.ml.enabled:
            return PredictionResult(0, 0.0, 0, False)

        features = self.feature_engine.compute_features(df, smc_analysis)
        if features.empty:
            return PredictionResult(0, 0.0, 0, False)

        if not self.model.is_trained:
            return PredictionResult(0, 0.0, features.shape[1], False)

        X = features.iloc[-1:].values
        pred_class, confidence = self.model.predict_confidence(X)

        signal_map = {0: -1, 1: 1}
        signal = signal_map.get(pred_class, 0)

        return PredictionResult(
            signal=signal,
            confidence=confidence,
            features_used=features.shape[1],
            model_ready=True,
        )

    async def train(self, df: pd.DataFrame):
        if not self.settings.ml.enabled:
            return

        self.log.info("Starting ML model training...")
        features = self.feature_engine.compute_features(df)

        if features.empty or len(features) < 100:
            self.log.warning(f"Insufficient data for training: {len(features)} rows")
            return

        labels = self.feature_engine.create_labels(df)
        valid_mask = labels != 0
        X = features[valid_mask].values
        y = labels[valid_mask].values + 1

        if len(X) < 50:
            self.log.warning(f"Insufficient labeled samples: {len(X)}")
            return

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        self.model.build(X.shape[1])
        self.model.train(X_train, y_train, X_val, y_val)
        self.model.save()
        self.log.info(f"ML model trained on {len(X_train)} samples")

    async def retrain_if_needed(self, df: pd.DataFrame, hours_since_train: float):
        if not self._initialized or not self.settings.ml.enabled:
            return

        if hours_since_train >= self.settings.ml.retrain_interval_hours:
            self.log.info("Retraining ML model...")
            await self.train(df)

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model.is_trained
