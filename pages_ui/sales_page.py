"""
Sales Analytics Page — Filterable sales data with interactive charts.
Uses caching to avoid reloading 33K+ records on every interaction.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import calendar

from database.connection import get_session
from services.sales import SalesService


@st.cache_data(ttl=60)
def _load_all_sales():
    """Load all sales data with 60-second cache."""
    with get_session() as session:
        sales_service = SalesService(session)
        return sales_service.get_all(), sales_service.get_medicine_names()


@st.cache_data(ttl=60)
def _load_sales_by_date(date_str):
    with get_session() as session:
        return SalesService(session).get_by_date(date_str)


@st.cache_data(ttl=60)
def _load_sales_by_month_year(month, year):
    with get_session() as session:
        return SalesService(session).get_by_month_year(month, year)


@st.cache_data(ttl=60)
def _load_sales_by_year(year):
    with get_session() as session:
        return SalesService(session).get_by_year(year)


def render():
    """Render the sales analytics page."""
    sales_data, medicine_names = _load_all_sales()

    st.title("Sales Data Filtering")

    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("Sales Filters")
        date_option = st.radio("Filter by Date:", ["None", "Select a Date"])
        
        if date_option == "Select a Date":
            date_f = st.date_input("Select a Date:", value=None)
            date_filter = date_f.strftime("%Y-%b-%d") if date_f else None
        else:
            date_filter = None

        month_filter = st.selectbox(
            "Select Month:", 
            ["None", "All"] + list(calendar.month_name[1:])
        )
        year_filter = st.selectbox(
            "Select Year:", 
            ["None"] + [str(y) for y in range(2020, 2031)]
        )

    # --- Filtering Logic (cached) ---
    if date_filter:
        st.write(f"Filtering by Date: {date_filter}")
        df = _load_sales_by_date(date_filter)
        st.write(df if not df.empty else "No data found for this date.")

    elif month_filter not in ("None", "All"):
        if year_filter == "None":
            st.warning("Please select a year for month filtering.")
            df = pd.DataFrame()
        else:
            st.write(f"Filtering by Month: {month_filter} and Year: {year_filter}")
            df = _load_sales_by_month_year(month_filter, int(year_filter))
            st.write(df if not df.empty else "No data found for this month and year.")

    elif year_filter != "None":
        st.write(f"Filtering by Year: {year_filter}")
        df = _load_sales_by_year(int(year_filter))
        st.write(df if not df.empty else "No data found for this year.")

    else:
        st.write("Displaying all sales data (Default View).")
        df = sales_data
        st.write(df if not df.empty else "No sales data available.")

    # --- Visualization ---
    _render_charts(sales_data, medicine_names)


def _render_charts(sales_data: pd.DataFrame, medicine_names: list[str]):
    """Render interactive sales charts with filters."""
    if sales_data.empty:
        return

    sales_data_viz = sales_data.copy()
    sales_data_viz["Date"] = pd.to_datetime(sales_data_viz["Date"], format="%Y-%b-%d")

    st.sidebar.header("Sales Visualization")

    medicine_filter = st.sidebar.selectbox(
        "Select Medicine", ["None"] + medicine_names, key="medicine_select",
    )

    from_date = st.sidebar.date_input("From Date", value=sales_data_viz["Date"].min())
    to_date = st.sidebar.date_input("To Date", value=sales_data_viz["Date"].max())

    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)

    graph_data = sales_data_viz[
        (sales_data_viz["Date"] >= from_date) & (sales_data_viz["Date"] <= to_date)
    ]
    if medicine_filter != "None":
        graph_data = graph_data[graph_data["Medicine_Name"] == medicine_filter]

    fig_bar = px.bar(
        graph_data, x="Date", y="Stock_Sold", color="Medicine_Name",
        title="Medicine Sales Over Time",
        labels={"Date": "Date", "Stock_Sold": "Units Sold", "Medicine_Name": "Medicine"},
        barmode="group",
    )
    fig_bar.update_layout(width=1000, height=600, dragmode="zoom", hovermode="x")

    fig_line = px.line(
        graph_data, x="Date", y="Stock_Sold", color="Medicine_Name",
        title="Medicine Sales Over Time",
        labels={"Date": "Date", "Stock_Sold": "Units Sold", "Medicine_Name": "Medicine"},
        markers=True,
    )
    fig_line.update_layout(width=1000, height=600)

    if st.toggle("Bar Chart", disabled=False):
        st.plotly_chart(fig_bar, use_container_width=True)
    if st.toggle("Line Chart", disabled=False):
        st.plotly_chart(fig_line, use_container_width=True)
