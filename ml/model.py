import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from utils import logger


class MLModel:
    def __init__(self, model_path: str = "models/xgboost_model.json", use_gpu: bool = False):
        self.log = logger
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self._model = None
        self._nn_model = None
        self._is_trained = False
        self._feature_importance: Optional[Dict[str, float]] = None

    def _import_xgboost(self):
        try:
            import xgboost as xgb
            return xgb
        except ImportError:
            self.log.warning("XGBoost not installed. Install with: pip install xgboost")
            return None

    def build(self, n_features: int):
        xgb = self._import_xgboost()
        if xgb is None:
            return

        params = {
            "objective": "multi:softprob",
            "num_class": 2,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": ["mlogloss", "merror"],
            "seed": 42,
        }

        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        self._model = xgb.XGBClassifier(**params)
        self.log.info(f"XGBoost model built with {n_features} features")

        # Build MLP Classifier (Neural Network with Input -> 16 -> 8 -> Output)
        from sklearn.neural_network import MLPClassifier
        self._nn_model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )
        self.log.info("MLP Neural Network (16, 8 neurons) built successfully")

    def train(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        if self._model is None:
            self.build(X.shape[1])

        # Train XGBoost
        eval_set = [(X, y)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self._model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )

        # Train MLP Neural Network
        try:
            self.log.info("Training MLP Neural Network...")
            self._nn_model.fit(X, y)
            self.log.info("MLP Neural Network training completed.")
        except Exception as e:
            self.log.error(f"Failed to train MLP Neural Network: {e}")

        self._is_trained = True
        self._compute_feature_importance()
        self.log.info(f"Model trained. Best iteration: {self._model.best_iteration if hasattr(self._model, 'best_iteration') else 'N/A'}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.array([])
        
        xgb_proba = self._model.predict_proba(X)
        try:
            if self._nn_model is not None:
                nn_proba = self._nn_model.predict_proba(X)
                # Ensemble: Average predictions between XGBoost & Neural Network
                return (xgb_proba + nn_proba) / 2.0
            return xgb_proba
        except Exception as e:
            self.log.error(f"Failed prediction with NN, falling back to XGBoost only: {e}")
            return xgb_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._is_trained:
            return np.array([])
        proba = self.predict_proba(X)
        if proba.size == 0:
            return np.array([])
        return np.argmax(proba, axis=1)

    def predict_confidence(self, X: np.ndarray) -> Tuple[int, float]:
        if self._model is None or X.size == 0 or not self._is_trained:
            return 0, 0.0

        proba = self.predict_proba(X.reshape(1, -1))
        if proba.size == 0:
            return 0, 0.0

        pred_class = int(np.argmax(proba[0]))
        confidence = float(np.max(proba[0]))
        return pred_class, confidence

    def save(self):
        if self._model is None:
            self.log.warning("No model to save")
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self.model_path))
        self.log.info(f"XGBoost model saved to {self.model_path}")

        # Save MLP model
        import pickle
        mlp_path = self.model_path.with_name("mlp_xaut.pkl")
        try:
            with open(mlp_path, "wb") as f:
                pickle.dump(self._nn_model, f)
            self.log.info(f"MLP Neural Network model saved to {mlp_path}")
        except Exception as e:
            self.log.error(f"Failed to save MLP model: {e}")

    def load(self) -> bool:
        xgb = self._import_xgboost()
        if xgb is None:
            return False

        if not self.model_path.exists():
            self.log.warning(f"Model not found at {self.model_path}")
            return False

        try:
            self._model = xgb.XGBClassifier()
            self._model.load_model(str(self.model_path))
            
            # Load MLP model
            import pickle
            mlp_path = self.model_path.with_name("mlp_xaut.pkl")
            if mlp_path.exists():
                with open(mlp_path, "rb") as f:
                    self._nn_model = pickle.load(f)
                self.log.info("MLP Neural Network model loaded successfully")
            else:
                self.log.warning(f"MLP Neural Network file not found at {mlp_path}. Rebuilding template...")
                from sklearn.neural_network import MLPClassifier
                self._nn_model = MLPClassifier(
                    hidden_layer_sizes=(16, 8),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=42,
                )
                self._is_trained = False
                return False

            self._is_trained = True
            self.log.info(f"Ensemble models successfully loaded from {self.model_path.parent}")
            return True
        except Exception as e:
            self.log.error(f"Failed to load models: {e}")
            self._is_trained = False
            return False

    def _compute_feature_importance(self):
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return
        importances = self._model.feature_importances_
        self._feature_importance = {f"f{i}": float(v) for i, v in enumerate(importances)}

    def get_feature_importance(self) -> Dict[str, float]:
        return self._feature_importance or {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def model(self):
        return self._model

    @property
    def nn_model(self):
        return self._nn_model
