"""
Forecasting Tests — XGBoost stockout prediction and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from ml.forecasting import StockoutPredictor, FeatureEngineer, ModelRegistry


class TestFeatureEngineer:
    """Tests for the feature engineering pipeline."""

    def test_build_features_columns(self):
        """Should create all expected feature columns."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "Stock_Sold": np.random.randint(1, 50, 30),
        })
        result = FeatureEngineer.build_features(df)

        expected_cols = ["Day", "DayOfWeek", "Month", "Quarter", "Rolling_7d", "Rolling_30d", "Lag_1", "Lag_7"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_day_index_starts_at_zero(self):
        """Day index should start at 0."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Stock_Sold": range(10),
        })
        result = FeatureEngineer.build_features(df)
        assert result["Day"].iloc[0] == 0
        assert result["Day"].iloc[9] == 9

    def test_rolling_mean_calculation(self):
        """Rolling means should be computed correctly."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=14, freq="D"),
            "Stock_Sold": [10] * 14,  # Constant sales
        })
        result = FeatureEngineer.build_features(df)
        # After 7+ days, 7-day rolling mean of constant 10 should be 10
        assert result["Rolling_7d"].iloc[-1] == 10.0

    def test_feature_columns_list(self):
        """Should return correct feature column names."""
        cols = FeatureEngineer.get_feature_columns()
        assert len(cols) == 8
        assert "Day" in cols
        assert "Rolling_7d" in cols


class TestModelRegistry:
    """Tests for model persistence."""

    def test_save_and_load_model(self, tmp_path):
        """Should round-trip save/load a model."""
        import xgboost as xgb

        registry = ModelRegistry(models_dir=tmp_path)

        # Train a simple model
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([10, 20, 30, 40, 50])
        model = xgb.XGBRegressor(n_estimators=10, verbosity=0)
        model.fit(X, y)

        registry.save_model("TestMedicine", model, {"rmse": 0.5})
        loaded = registry.load_model("TestMedicine")

        assert loaded is not None
        pred_original = model.predict(X)
        pred_loaded = loaded.predict(X)
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

    def test_load_nonexistent_model(self, tmp_path):
        """Should return None for missing model."""
        registry = ModelRegistry(models_dir=tmp_path)
        result = registry.load_model("NonExistent")
        assert result is None

    def test_save_and_load_metrics(self, tmp_path):
        """Should persist metrics report."""
        registry = ModelRegistry(models_dir=tmp_path)
        from ml.forecasting import METRICS_FILE
        # Override the global path for testing
        import ml.forecasting as mf
        original = mf.METRICS_FILE
        mf.METRICS_FILE = tmp_path / "training_metrics.json"

        metrics = {"Aspirin": {"rmse": 1.5, "r2": 0.95}}
        registry.save_metrics(metrics)
        loaded = registry.load_metrics()

        assert loaded is not None
        assert loaded["model_count"] == 1
        assert "Aspirin" in loaded["models"]

        mf.METRICS_FILE = original  # Restore


class TestStockoutPredictor:
    """Tests for the stockout prediction pipeline."""

    def test_predict_with_no_model(self):
        """Should return None when no model exists."""
        predictor = StockoutPredictor()
        # Override registry to empty dir
        from pathlib import Path
        predictor.registry = ModelRegistry(models_dir=Path(__import__("tempfile").mkdtemp()))
        result = predictor.predict_stockout({}, "UnknownMedicine", 100)
        assert result is None

    def test_predict_respects_max_days(self):
        """Should return None when prediction exceeds max_days."""
        predictor = StockoutPredictor()
        predictor.max_days = 10

        class MockModel:
            def predict(self, X):
                return np.array([0.0])

        models = {"TestMedicine": MockModel()}
        result = predictor.predict_stockout(models, "TestMedicine", 100)
        assert result is None

    def test_predict_normal_stockout(self):
        """Should return correct days for consistent sales."""
        predictor = StockoutPredictor()
        predictor.max_days = 365

        class MockModel:
            def predict(self, X):
                return np.array([10.0])

        models = {"TestMedicine": MockModel()}
        result = predictor.predict_stockout(models, "TestMedicine", 50)
        assert result == 5  # 50 / 10 = 5 days

    def test_predict_with_negative_sales(self):
        """Should clamp negative predictions to 0."""
        predictor = StockoutPredictor()
        predictor.max_days = 100

        class MockModel:
            def predict(self, X):
                return np.array([-5.0])

        models = {"TestMedicine": MockModel()}
        result = predictor.predict_stockout(models, "TestMedicine", 10)
        assert result is None  # Never reaches stockout
