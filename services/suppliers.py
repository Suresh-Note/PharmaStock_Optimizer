"""
Supplier Service — Supplier-Medicine Mapping and Stock Analysis

Handles supplier data aggregation from the Medicine and Inventory tables.
"""

import pandas as pd
from sqlalchemy.orm import Session

from database.models import Medicine, Inventory
from utils.logger import get_logger

logger = get_logger(__name__)


class SupplierService:
    """Handles supplier-related data operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_supplier_stock(self) -> pd.DataFrame:
        """Fetch supplier-medicine-stock joined data."""
        results = self.session.query(
            Medicine.Medicine_Name,
            Medicine.Supplier_ID,
            Inventory.Stock_Available,
        ).join(
            Inventory, Medicine.Medicine_Name == Inventory.Medicine_Name
        ).all()

        return pd.DataFrame(results, columns=["Medicine_Name", "Supplier_ID", "Stock_Available"])

    def get_supplier_mapping(self) -> tuple[list[str], dict]:
        """
        Build a mapping of suppliers to their medicines and stock levels.
        
        Returns:
            Tuple of (sorted_supplier_list, supplier_mapping_dict)
            where supplier_mapping_dict maps supplier_id -> [(medicine, stock), ...]
        """
        df = self.get_supplier_stock()
        
        supplier_mapping = {}
        unique_suppliers = set()

        for _, row in df.iterrows():
            medicine = row["Medicine_Name"]
            suppliers = [s.strip() for s in str(row["Supplier_ID"]).split(",")]
            stock = row.get("Stock_Available", 0)

            for supplier in suppliers:
                unique_suppliers.add(supplier)
                if supplier not in supplier_mapping:
                    supplier_mapping[supplier] = []
                supplier_mapping[supplier].append((medicine, stock))

        sorted_suppliers = sorted(unique_suppliers)
        logger.info(f"Supplier mapping built: {len(sorted_suppliers)} suppliers")
        return sorted_suppliers, supplier_mapping
