"""
Industrial ML Forecasting Pipeline — XGBoost Stockout Prediction

Features:
- Per-medicine model training with cross-validation
- Feature engineering (day, day-of-week, month, rolling averages)
- Training metrics tracking (RMSE, MAE, R2)
- Model versioning with disk persistence
- Performance logging with timing
- Graceful degradation on errors
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import settings
from utils.logger import get_logger, log_performance

logger = get_logger(__name__)

MODELS_DIR = Path("ml_models")
METRICS_FILE = MODELS_DIR / "training_metrics.json"


class ModelRegistry:
    """Manages model persistence and versioning."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, medicine_name: str, model: xgb.XGBRegressor, metrics: dict):
        """Save a trained model and its metrics to disk."""
        safe_name = medicine_name.replace(" ", "_").lower()
        model_path = self.models_dir / f"{safe_name}.json"
        model.save_model(str(model_path))
        logger.debug(f"Model saved: {model_path}")

    def load_model(self, medicine_name: str) -> Optional[xgb.XGBRegressor]:
        """Load a model from disk, returns None if not found."""
        safe_name = medicine_name.replace(" ", "_").lower()
        model_path = self.models_dir / f"{safe_name}.json"
        if model_path.exists():
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            return model
        return None

    def save_metrics(self, all_metrics: dict):
        """Save training metrics report to disk."""
        report = {
            "trained_at": datetime.now().isoformat(),
            "model_count": len(all_metrics),
            "models": all_metrics,
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Metrics report saved: {METRICS_FILE}")

    def load_metrics(self) -> Optional[dict]:
        """Load metrics from disk."""
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                return json.load(f)
        return None


class FeatureEngineer:
    """Extracts features from time-series sales data."""

    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enriched feature set from date and sales data.
        
        Features:
        - Day index (days since first sale)
        - Day of week (0=Mon, 6=Sun)
        - Month (1-12)
        - Quarter (1-4)
        - Rolling mean (7-day and 30-day)
        - Lag features (1-day and 7-day)
        """
        df = df.copy()
        df["Day"] = (df["Date"] - df["Date"].min()).dt.days
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter

        # Rolling statistics
        df["Rolling_7d"] = df["Stock_Sold"].rolling(window=7, min_periods=1).mean()
        df["Rolling_30d"] = df["Stock_Sold"].rolling(window=30, min_periods=1).mean()

        # Lag features
        df["Lag_1"] = df["Stock_Sold"].shift(1).fillna(0)
        df["Lag_7"] = df["Stock_Sold"].shift(7).fillna(0)

        return df

    @staticmethod
    def get_feature_columns() -> list[str]:
        """Return the list of feature column names."""
        return ["Day", "DayOfWeek", "Month", "Quarter", "Rolling_7d", "Rolling_30d", "Lag_1", "Lag_7"]


class StockoutPredictor:
    """Industrial-grade XGBoost stockout prediction pipeline."""

    def __init__(self):
        self.n_estimators = settings.XGBOOST_N_ESTIMATORS
        self.max_days = settings.MAX_STOCKOUT_DAYS
        self.registry = ModelRegistry()
        self.feature_engineer = FeatureEngineer()
        self._models_cache: dict = {}
        self._cache_hash: str = ""

    @log_performance
    def train_all_models(self, force_retrain: bool = False) -> dict:
        """
        Train XGBoost models for all medicines with cross-validation.
        
        Returns:
            Dictionary mapping medicine_name -> trained XGBRegressor
        """
        from database.connection import get_session
        from services.sales import SalesService

        with get_session() as session:
            sales_service = SalesService(session)
            sales_data = sales_service.get_all()

        if sales_data.empty:
            logger.warning("No sales data available for model training")
            return {}

        # Check cache validity
        data_hash = hashlib.md5(pd.util.hash_pandas_object(sales_data).values.tobytes()).hexdigest()[:12]
        if not force_retrain and data_hash == self._cache_hash and self._models_cache:
            logger.info(f"Using cached models (hash: {data_hash})")
            return self._models_cache

        # Parse dates
        sales_data["Date"] = pd.to_datetime(sales_data["Date"], format="mixed")
        grouped = sales_data.groupby(
            ["Medicine_Name", pd.Grouper(key="Date", freq="D")]
        )["Stock_Sold"].sum().reset_index()

        models = {}
        all_metrics = {}
        feature_cols = self.feature_engineer.get_feature_columns()

        for medicine in grouped["Medicine_Name"].unique():
            try:
                df = grouped[grouped["Medicine_Name"] == medicine].copy()
                df = df.sort_values("Date").reset_index(drop=True)
                df = self.feature_engineer.build_features(df)

                X = df[feature_cols]
                y = df["Stock_Sold"]

                if len(X) < 10:
                    logger.debug(f"Skipping '{medicine}': insufficient data ({len(X)} rows)")
                    continue

                # Train model
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbosity=0,
                    random_state=42,
                )
                model.fit(X, y)

                # Cross-validation (3-fold for speed)
                n_splits = min(3, len(X) // 5)
                if n_splits >= 2:
                    cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="neg_mean_squared_error")
                    cv_rmse = np.sqrt(-cv_scores.mean())
                else:
                    cv_rmse = None

                # Evaluation metrics
                y_pred = model.predict(X)
                metrics = {
                    "n_samples": len(X),
                    "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                    "mae": round(float(mean_absolute_error(y, y_pred)), 4),
                    "r2": round(float(r2_score(y, y_pred)), 4),
                    "cv_rmse": round(float(cv_rmse), 4) if cv_rmse else None,
                }

                # Feature importance
                importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
                top_feature = max(importance, key=importance.get)
                metrics["top_feature"] = top_feature
                metrics["feature_importance"] = {k: round(v, 4) for k, v in importance.items()}

                models[medicine] = model
                all_metrics[medicine] = metrics

                # Save to disk
                self.registry.save_model(medicine, model, metrics)

            except Exception as e:
                logger.error(f"Failed to train model for '{medicine}': {e}")
                continue

        # Save metrics report
        self.registry.save_metrics(all_metrics)

        # Update cache
        self._models_cache = models
        self._cache_hash = data_hash

        # Log summary
        if all_metrics:
            avg_r2 = np.mean([m["r2"] for m in all_metrics.values()])
            avg_rmse = np.mean([m["rmse"] for m in all_metrics.values()])
            logger.info(
                f"Trained {len(models)} models | Avg R2: {avg_r2:.4f} | Avg RMSE: {avg_rmse:.4f}"
            )

        return models

    def predict_stockout(self, models: dict, medicine_name: str,
                         stock_available: int) -> Optional[int]:
        """
        Predict days until stockout using feature-engineered inputs.
        
        Args:
            models: Dictionary of trained models
            medicine_name: Medicine to predict for
            stock_available: Current stock level
            
        Returns:
            Predicted days to stockout, or None if >MAX_DAYS or no model
        """
        if medicine_name not in models:
            # Try loading from disk
            model = self.registry.load_model(medicine_name)
            if model is None:
                logger.warning(f"No model for '{medicine_name}', skipping prediction")
                return None
            models[medicine_name] = model

        model = models[medicine_name]
        feature_cols = self.feature_engineer.get_feature_columns()

        days = 0
        remaining_stock = float(stock_available)
        recent_sales = [0.0] * 30  # Rolling window seed

        while remaining_stock > 0 and days < self.max_days:
            # Build feature vector for current day
            rolling_7d = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else 0
            rolling_30d = np.mean(recent_sales[-30:]) if len(recent_sales) >= 30 else 0
            lag_1 = recent_sales[-1] if recent_sales else 0
            lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else 0

            features = pd.DataFrame([[
                days,           # Day
                days % 7,       # DayOfWeek (approximate)
                1,              # Month (default)
                1,              # Quarter (default)
                rolling_7d,     # Rolling_7d
                rolling_30d,    # Rolling_30d
                lag_1,          # Lag_1
                lag_7,          # Lag_7
            ]], columns=feature_cols)

            predicted_sales = max(float(model.predict(features)[0]), 0)
            remaining_stock -= predicted_sales
            recent_sales.append(predicted_sales)
            days += 1

        if days >= self.max_days:
            logger.info(f"Stockout prediction for '{medicine_name}': >365 days (stable)")
            return None

        logger.info(f"Stockout prediction for '{medicine_name}': {days} days")
        return days

    def predict_for_update(self, medicine_name: str, stock_available: int) -> Optional[int]:
        """Train models (cached) and predict stockout for a single medicine."""
        models = self.train_all_models()
        return self.predict_stockout(models, medicine_name, stock_available)

    def get_model_metrics(self) -> Optional[dict]:
        """Load the latest training metrics report from disk."""
        return self.registry.load_metrics()
