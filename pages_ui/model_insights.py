"""
Model Insights Page — Transparency into the stockout forecasting pipeline.

Surfaces the same held-out test metrics, baseline comparison, and feature
importance that live in ml_models/training_metrics.json, so the ML pipeline's
results are visible in the app itself rather than only in log files.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from ml.forecasting import StockoutPredictor


def render():
    """Render the model insights page."""
    st.title("🧠 Model Insights")
    st.caption("Held-out test performance and feature importance for the stockout forecasting models.")

    predictor = StockoutPredictor()
    report = predictor.get_model_metrics()

    if report is None:
        st.info(
            "No models have been trained yet. Update any medicine's inventory "
            "(Inventory → Manage Inventory → Update Inventory) to trigger training, "
            "or use the button below."
        )
        if st.button("Train Models Now"):
            with st.spinner("Training XGBoost models for all medicines — this takes about a minute..."):
                predictor.train_all_models(force_retrain=True)
            st.rerun()
        return

    metrics = report["models"]
    if not metrics:
        st.warning("Training ran but produced no models — check that sales data is available.")
        return

    results_df = pd.DataFrame(metrics).T
    for col in [
        "test_rmse", "test_mae", "test_r2", "train_rmse", "train_r2",
        "cv_rmse", "baseline_rmse", "improvement_over_baseline_pct",
    ]:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    st.caption(f"Last trained: {report.get('trained_at', 'unknown')} · {report.get('model_count', 0)} models")

    # --- Summary KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models Trained", report.get("model_count", 0))
    col2.metric("Avg Test RMSE", f"{results_df['test_rmse'].mean():.2f}")
    col3.metric("Avg Test R²", f"{results_df['test_r2'].mean():.3f}")
    col4.metric("Avg Improvement vs Baseline", f"{results_df['improvement_over_baseline_pct'].mean():.1f}%")

    st.info(
        "Test R² is low across the board — day-to-day medicine demand is inherently noisy "
        "at this granularity. What matters for reorder decisions is that the model beats a "
        "naive \"tomorrow = yesterday\" baseline, which it does by the improvement % above."
    )

    st.divider()

    # --- Per-medicine results table ---
    st.subheader("Per-Medicine Test Results")
    display_df = results_df[
        ["test_rmse", "test_mae", "test_r2", "baseline_rmse", "improvement_over_baseline_pct", "top_feature"]
    ].sort_values("improvement_over_baseline_pct", ascending=False)
    display_df.columns = ["Test RMSE", "Test MAE", "Test R²", "Baseline RMSE", "Improvement vs Baseline (%)", "Top Feature"]
    st.dataframe(display_df, use_container_width=True)

    st.divider()

    # --- Feature importance ---
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({m: d["feature_importance"] for m, d in metrics.items()}).T
    avg_importance = importance_df.mean().sort_values(ascending=False).reset_index()
    avg_importance.columns = ["Feature", "Average Importance"]

    fig = px.bar(
        avg_importance, x="Average Importance", y="Feature", orientation="h",
        color="Average Importance", color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Per-medicine drill-down ---
    st.subheader("Inspect a Single Medicine")
    selected = st.selectbox("Select Medicine", sorted(metrics.keys()))
    med = metrics[selected]

    c1, c2, c3 = st.columns(3)
    c1.metric("Train R²", f"{float(med['train_r2']):.3f}")
    c2.metric("Test R²", f"{float(med['test_r2']):.3f}")
    c3.metric("Improvement vs Baseline", f"{float(med['improvement_over_baseline_pct']):.1f}%")

    med_importance = pd.Series(med["feature_importance"]).sort_values(ascending=False).reset_index()
    med_importance.columns = ["Feature", "Importance"]
    fig2 = px.bar(med_importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#0F766E"])
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, title=f"Feature Importance — {selected}")
    st.plotly_chart(fig2, use_container_width=True)

    if st.button("Retrain All Models"):
        with st.spinner("Retraining XGBoost models for all medicines — this takes about a minute..."):
            predictor.train_all_models(force_retrain=True)
        st.rerun()
