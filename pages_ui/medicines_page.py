"""
Medicine Reference Page — Drug information with stock availability.
"""

import json
import streamlit as st
from pathlib import Path

from database.connection import get_session
from services.inventory import InventoryService


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "medicines_info.json"


def _load_medicines_info() -> list[dict]:
    """Load medicine reference data from JSON file."""
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Medicine data file not found: {DATA_FILE}")
        return []


def render():
    """Render the medicine reference page."""
    medicines_info = _load_medicines_info()
    if not medicines_info:
        return

    # Fetch stock data
    with get_session() as session:
        inv_service = InventoryService(session)
        stock_data = inv_service.get_stock_data()

    # Build lookup
    medicine_dict = {m["Medicine_Name"]: m for m in medicines_info}

    st.title("💊 Medicine Information")

    # Selection
    medicine_names = list(medicine_dict.keys())
    selected_medicine = st.selectbox("Select a medicine:", medicine_names)

    if selected_medicine:
        medicine = medicine_dict[selected_medicine]
        
        st.subheader(medicine["Medicine_Name"])
        st.write(f"**ATC Code:** {medicine['ATC_Code']}")
        st.write(f"**Description:** {medicine['Description']}")
        st.write(f"**Uses:** {medicine['Uses']}")
        st.write(f"**Cautions:** {medicine['Cautions']}")
        st.write(f"**Dosage:** {medicine['Dosage']}")
        st.write("**Alternatives:**")
        for alt in medicine["Alternatives"]:
            st.write(f"- {alt}")

        # Stock availability
        stock_info = stock_data[stock_data["Medicine_Name"] == selected_medicine]
        if not stock_info.empty:
            stock_available = int(stock_info["Stock_Available"].values[0])
            if stock_available > 0:
                st.success(f"**Stock Available:** {stock_available} units")
            else:
                st.error("**Stock Available:** Out of stock")
        else:
            st.warning("**Stock Available:** Information not available")
