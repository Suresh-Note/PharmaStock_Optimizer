"""
Dashboard Page — Main overview after login.
Uses Streamlit caching to avoid redundant DB queries on every rerun.
"""

import streamlit as st
import plotly.express as px

from database.connection import get_session
from services.inventory import InventoryService


@st.cache_data(ttl=30)
def _load_dashboard_data():
    """Load inventory data with 30-second cache to avoid repeated queries."""
    with get_session() as session:
        inv_service = InventoryService(session)
        stock_data = inv_service.get_dataframe()
        summary = inv_service.get_summary()
    return stock_data, summary


def render():
    """Render the dashboard page."""
    st.title(f"Welcome, {st.session_state.username}! 👋")
    st.caption("Here's the current state of your pharmacy inventory.")

    stock_data, summary = _load_dashboard_data()

    # KPI cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Medicines", summary["total_medicines"])
    col2.metric("Low Stock", summary["low_stock"])
    col3.metric("Expired", summary["expired"])

    st.divider()

    # Data table
    st.subheader("Inventory Overview")
    st.dataframe(stock_data, use_container_width=True)

    # Separate stock levels
    low_stock = stock_data[stock_data["Stock_Available"] <= stock_data["Reorder_Level"]]
    sufficient_stock = stock_data[stock_data["Stock_Available"] > stock_data["Reorder_Level"]]

    # Plot for low stock (Red)
    fig_low = px.bar(
        low_stock,
        x="Medicine_Name", y="Stock_Available",
        labels={"Medicine_Name": "Medicine", "Stock_Available": "Quantity"},
        color_discrete_sequence=["#d62728"],
    )

    # Plot for sufficient stock (Blues gradient)
    fig_sufficient = px.bar(
        sufficient_stock,
        x="Medicine_Name", y="Stock_Available",
        labels={"Medicine_Name": "Medicine", "Stock_Available": "Quantity"},
        color="Stock_Available",
        color_continuous_scale="Blues",
    )

    # Combine both plots
    fig_low.add_traces(fig_sufficient.data)

    st.divider()
    st.subheader("Stock Levels")
    st.plotly_chart(fig_low, use_container_width=True)
