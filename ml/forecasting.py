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
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import settings
from utils.logger import get_logger, log_performance

logger = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "ml_models"
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

    def _new_model(self) -> xgb.XGBRegressor:
        """Build a freshly configured, unfitted XGBRegressor with the pipeline's hyperparameters."""
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
            random_state=42,
        )

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

                result = self._train_and_evaluate(df, feature_cols)
                if result is None:
                    logger.debug(f"Skipping '{medicine}': insufficient data ({len(df)} rows)")
                    continue

                final_model, metrics = result
                models[medicine] = final_model
                all_metrics[medicine] = metrics
                self.registry.save_model(medicine, final_model, metrics)

            except Exception as e:
                logger.error(f"Failed to train model for '{medicine}': {e}")
                continue

        # Save metrics report
        self.registry.save_metrics(all_metrics)

        # Update cache
        self._models_cache = models
        self._cache_hash = data_hash

        # Log summary (test-set metrics only — the honest, out-of-sample numbers)
        if all_metrics:
            avg_r2 = np.mean([m["test_r2"] for m in all_metrics.values()])
            avg_rmse = np.mean([m["test_rmse"] for m in all_metrics.values()])
            avg_improvement = np.mean([
                m["improvement_over_baseline_pct"] for m in all_metrics.values()
                if m["improvement_over_baseline_pct"] is not None
            ])
            logger.info(
                f"Trained {len(models)} models | Avg test R2: {avg_r2:.4f} | "
                f"Avg test RMSE: {avg_rmse:.4f} | Avg improvement over baseline: {avg_improvement:.1f}%"
            )

        return models

    def _train_and_evaluate(self, df: pd.DataFrame, feature_cols: list[str]) -> Optional[tuple]:
        """
        Train and honestly evaluate a single medicine's model.

        Uses a chronological (not random) train/test split, since shuffling
        time-ordered sales data would leak future information into training.
        A naive "same as yesterday" baseline is evaluated on the same held-out
        window so the model's value is quantified, not assumed. The returned
        model is refit on the full dataset (train + test) for best real-world
        predictions, matching standard forecasting practice: hold out data to
        validate, then retrain on everything for production.

        Returns:
            (final_model, metrics) or None if there isn't enough data to split.
        """
        X, y = df[feature_cols], df["Stock_Sold"]
        if len(X) < 20:
            return None

        test_size = max(5, int(len(X) * 0.2))
        split_idx = len(X) - test_size
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        eval_model = self._new_model()
        eval_model.fit(X_train, y_train)

        # Time-series cross-validation on the training window only: each fold
        # trains on the past and validates on the future, unlike a random
        # K-fold split which would let later rows leak into earlier folds.
        cv_rmse = None
        n_splits = min(3, len(X_train) // 5)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(eval_model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
            cv_rmse = float(np.sqrt(-cv_scores.mean()))

        train_pred = eval_model.predict(X_train)
        test_pred = eval_model.predict(X_test)

        # Naive baseline: "tomorrow's sales = yesterday's sales" (the Lag_1 feature),
        # evaluated on the exact same held-out window as the model.
        baseline_pred = X_test["Lag_1"].values
        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        improvement_pct = (
            round((baseline_rmse - test_rmse) / baseline_rmse * 100, 2) if baseline_rmse > 0 else None
        )

        # Refit on all available data for the model actually used to predict stockouts.
        final_model = self._new_model()
        final_model.fit(X, y)
        importance = dict(zip(feature_cols, final_model.feature_importances_.tolist()))

        metrics = {
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_rmse": round(test_rmse, 4),
            "test_mae": round(float(mean_absolute_error(y_test, test_pred)), 4),
            "test_r2": round(float(r2_score(y_test, test_pred)), 4),
            "train_rmse": round(float(np.sqrt(mean_squared_error(y_train, train_pred))), 4),
            "train_mae": round(float(mean_absolute_error(y_train, train_pred)), 4),
            "train_r2": round(float(r2_score(y_train, train_pred)), 4),
            "cv_rmse": round(cv_rmse, 4) if cv_rmse is not None else None,
            "baseline_rmse": round(baseline_rmse, 4),
            "baseline_mae": round(float(mean_absolute_error(y_test, baseline_pred)), 4),
            "improvement_over_baseline_pct": improvement_pct,
            "top_feature": max(importance, key=importance.get),
            "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        }
        return final_model, metrics

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
        start_date = datetime.now()

        while remaining_stock > 0 and days < self.max_days:
            # Build feature vector for current day
            rolling_7d = np.mean(recent_sales[-7:]) if len(recent_sales) >= 7 else 0
            rolling_30d = np.mean(recent_sales[-30:]) if len(recent_sales) >= 30 else 0
            lag_1 = recent_sales[-1] if recent_sales else 0
            lag_7 = recent_sales[-7] if len(recent_sales) >= 7 else 0

            current_date = start_date + timedelta(days=days)

            features = pd.DataFrame([[
                days,                             # Day
                current_date.weekday(),           # DayOfWeek
                current_date.month,                # Month
                (current_date.month - 1) // 3 + 1,  # Quarter
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
