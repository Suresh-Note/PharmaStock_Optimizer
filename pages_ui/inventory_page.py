"""
Inventory Management Page — View, update stock, and place orders.
"""

import streamlit as st
import pandas as pd

from database.connection import get_session
from services.inventory import InventoryService
from services.orders import OrderService
from ml.forecasting import StockoutPredictor
from utils.exceptions import InsufficientStockError


def render():
    """Render the inventory management page."""
    st.title("📦 Inventory Management")
    option = st.radio("Select an Option:", ["View Inventory", "Manage Inventory", "Orders and Payments"])

    if option == "View Inventory":
        _view_inventory()
    elif option == "Manage Inventory":
        _manage_inventory()
    elif option == "Orders and Payments":
        _orders_and_payments()


def _view_inventory():
    """Display all inventory records."""
    with get_session() as session:
        inv_service = InventoryService(session)
        df = inv_service.get_dataframe()
    st.dataframe(df, use_container_width=True)


def _manage_inventory():
    """Interface for updating inventory records."""
    with get_session() as session:
        inv_service = InventoryService(session)
        df = inv_service.get_dataframe()

    medicine_selected = st.selectbox("Select Medicine", df["Medicine_Name"].unique())
    row = df[df["Medicine_Name"] == medicine_selected].iloc[0]

    stock_available = st.number_input("Stock Available", min_value=0, value=int(row["Stock_Available"]))
    price_per_unit = st.number_input("Price per Unit", min_value=0.0, value=float(row["Price_per_Unit"]))
    reorder_level = st.number_input("Reorder Level", min_value=0, value=int(row["Reorder_Level"]))
    expiry_date = st.date_input("Select Expiry Date", value=pd.to_datetime(row["Expiry_Date"]))

    if st.button("Update Inventory"):
        predictor = StockoutPredictor()
        days_to_stockout = predictor.predict_for_update(medicine_selected, stock_available)
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        with get_session() as session:
            inv_service = InventoryService(session)
            inv_service.update(
                medicine_name=medicine_selected,
                stock_available=stock_available,
                price_per_unit=price_per_unit,
                reorder_level=reorder_level,
                expiry_date=expiry_str,
                days_to_stockout=days_to_stockout,
            )

        stockout_msg = f"Days to Stockout: {days_to_stockout}" if days_to_stockout else "Days to Stockout: >365 (stable)"
        st.success(f"Updated {medicine_selected} successfully! {stockout_msg}, Expiry Date: {expiry_str}")


def _orders_and_payments():
    """Shopping cart and order placement interface."""
    st.subheader("🛒 Orders and Payments")

    with get_session() as session:
        inv_service = InventoryService(session)
        df = inv_service.get_dataframe()

    medicine_selected = st.selectbox("Select Medicine", df["Medicine_Name"].unique())
    quantity = st.number_input("Quantity", min_value=1, value=1)
    row = df[df["Medicine_Name"] == medicine_selected].iloc[0]
    price_per_unit = float(row["Price_per_Unit"])
    stock_available = int(row["Stock_Available"])

    if st.button("Add to Cart"):
        try:
            with get_session() as session:
                order_service = OrderService(session)
                order_service.add_to_cart(medicine_selected, quantity, price_per_unit, stock_available)
            st.success(f"Added {quantity} units of {medicine_selected} to cart.")
        except InsufficientStockError as e:
            st.error(str(e))

    # Display cart
    with get_session() as session:
        order_service = OrderService(session)
        cart_items, total_cost = order_service.get_cart_summary()

    if cart_items:
        st.subheader("Cart Items")
        st.table(cart_items)
        st.write(f"**Total Cost: ₹{total_cost:.2f}**")

        if st.button("Place Order"):
            with get_session() as session:
                order_service = OrderService(session)
                order_service.place_order()
            st.success("Order placed successfully! Inventory has been updated.")
