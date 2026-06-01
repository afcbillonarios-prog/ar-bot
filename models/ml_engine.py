import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

from config.settings import config

logger = logging.getLogger(__name__)


class MLEngine:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns = config.ml.feature_columns
        self.model_path = config.ml.model_path
        self.last_train_time: Optional[datetime] = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Modelo ML cargado desde {self.model_path}")
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                self.model = None
        else:
            logger.warning(f"No se encontró modelo en {self.model_path}. Entrena el modelo primero.")

    def predict(self, features: pd.DataFrame) -> dict:
        if self.model is None:
            return {"prediction": "NO_TRADE", "confidence": 0.0, "probability": 0.0}

        try:
            available = [f for f in self.feature_columns if f in features.columns]
            X = features[available].iloc[[-1]]

            if X.isnull().any().any():
                X = X.fillna(0)

            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]

            prediction = "BUY" if pred == 1 else "NO_TRADE"
            confidence = float(max(proba))

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probability": float(proba[1]) if len(proba) > 1 else float(proba[0]),
            }

        except Exception as e:
            logger.error(f"Error en predicción ML: {e}", exc_info=True)
            return {"prediction": "NO_TRADE", "confidence": 0.0, "probability": 0.0}

    def train(self, df: pd.DataFrame, force: bool = False) -> dict:
        if not force and self.last_train_time:
            hours_since = (datetime.now(timezone.utc) - self.last_train_time).total_seconds() / 3600
            if hours_since < config.ml.retrain_interval_hours:
                return {"status": "skipped", "reason": f"Último entrenamiento hace {hours_since:.1f}h"}

        try:
            from indicators.technical import TechnicalIndicators
            ti = TechnicalIndicators(df)
            df_ind = ti.compute_all()

            ml_data = ti.prepare_ml_features()
            if len(ml_data) < 500:
                return {"status": "error", "reason": f"Datos insuficientes: {len(ml_data)} filas"}

            available = [f for f in self.feature_columns if f in ml_data.columns]
            X = ml_data[available].fillna(0)
            y = ml_data["target"]

            tscv = TimeSeriesSplit(n_splits=config.ml.walk_forward_windows)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )

                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                scores.append(acc)

            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )

            split = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            y_pred_final = self.model.predict(X_test)
            final_acc = accuracy_score(y_test, y_pred_final)

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save_model(self.model_path)
            self.last_train_time = datetime.now(timezone.utc)

            report = classification_report(y_test, y_pred_final, output_dict=True)

            result = {
                "status": "trained",
                "cv_accuracy_mean": np.mean(scores),
                "cv_accuracy_std": np.std(scores),
                "test_accuracy": final_acc,
                "precision_buy": report.get("1", {}).get("precision", 0),
                "recall_buy": report.get("1", {}).get("recall", 0),
                "f1_buy": report.get("1", {}).get("f1-score", 0),
                "samples": len(X),
                "features": available,
                "model_path": self.model_path,
            }

            logger.info(f"Modelo entrenado: accuracy={final_acc:.4f} CV_mean={np.mean(scores):.4f}")
            return result

        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def should_retrain(self) -> bool:
        if self.last_train_time is None:
            return True
        hours_since = (datetime.now(timezone.utc) - self.last_train_time).total_seconds() / 3600
        return hours_since >= config.ml.retrain_interval_hours

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        importance = self.model.feature_importances_
        available = [f for f in self.feature_columns if f in self.model.get_booster().feature_names or True]
        fi = pd.DataFrame({
            "feature": available[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return fi
