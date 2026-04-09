"""
Supplier Management Page — View medicines by supplier with stock charts.
Uses caching to avoid repeated DB queries.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from database.connection import get_session
from services.suppliers import SupplierService


@st.cache_data(ttl=30)
def _load_supplier_data():
    """Load supplier mapping with 30-second cache."""
    with get_session() as session:
        supplier_service = SupplierService(session)
        return supplier_service.get_supplier_mapping()


def render():
    """Render the supplier management page."""
    sorted_suppliers, supplier_mapping = _load_supplier_data()

    st.sidebar.header("Select a Supplier")
    selected_supplier = st.sidebar.selectbox(
        "Select Supplier", 
        ["Select"] + sorted_suppliers
    )

    if selected_supplier and selected_supplier != "Select":
        medicine_stock_list = supplier_mapping.get(selected_supplier, [])

        if medicine_stock_list:
            stock_df = pd.DataFrame(
                medicine_stock_list, 
                columns=["Medicine", "Stock Available"]
            )

            st.subheader(f"Medicines Supplied by {selected_supplier}")
            st.dataframe(stock_df)

            fig = px.bar(
                stock_df, x="Medicine", y="Stock Available",
                title=f"Stock Levels of Medicines Supplied by {selected_supplier}",
                labels={"Stock Available": "Stock Level"},
                color="Stock Available",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig)
        else:
            st.warning(f"No stock data available for supplier: {selected_supplier}")
