"""
Inventory Service — Stock Management CRUD Operations

Handles all inventory queries and updates using SQLAlchemy ORM.
"""

from typing import Optional
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models import Inventory
from utils.logger import get_logger
from utils.exceptions import MedicineNotFoundError

logger = get_logger(__name__)


class InventoryService:
    """Handles inventory data operations."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self) -> list[Inventory]:
        """Fetch all inventory records."""
        return self.session.query(Inventory).all()

    def get_dataframe(self) -> pd.DataFrame:
        """Fetch all inventory records as a pandas DataFrame."""
        records = self.get_all()
        return pd.DataFrame([{
            "Medicine_Name": r.Medicine_Name,
            "ATC_Code": r.ATC_Code,
            "Stock_Available": r.Stock_Available,
            "Price_per_Unit": r.Price_per_Unit,
            "Reorder_Level": r.Reorder_Level,
            "Days_to_Stockout": r.Days_to_Stockout,
            "Expiry_Date": r.Expiry_Date,
        } for r in records])

    def get_by_name(self, medicine_name: str) -> Inventory:
        """Fetch a single inventory record by medicine name."""
        item = self.session.query(Inventory).filter_by(
            Medicine_Name=medicine_name
        ).first()
        if not item:
            raise MedicineNotFoundError(medicine_name)
        return item

    def update(
        self,
        medicine_name: str,
        stock_available: int,
        price_per_unit: float,
        reorder_level: int,
        expiry_date: str,
        days_to_stockout: Optional[int] = None,
    ) -> Inventory:
        """
        Update an inventory record.
        
        Args:
            medicine_name: Medicine to update
            stock_available: New stock quantity
            price_per_unit: New price per unit
            reorder_level: New reorder threshold
            expiry_date: New expiry date (YYYY-MM-DD string)
            days_to_stockout: ML-predicted days to stockout
            
        Returns:
            Updated Inventory object
        """
        item = self.get_by_name(medicine_name)
        item.Stock_Available = stock_available
        item.Price_per_Unit = price_per_unit
        item.Reorder_Level = reorder_level
        item.Expiry_Date = expiry_date
        if days_to_stockout is not None:
            item.Days_to_Stockout = days_to_stockout
        
        self.session.flush()
        logger.info(
            f"Inventory updated: {medicine_name} | "
            f"stock={stock_available}, price={price_per_unit}, "
            f"reorder={reorder_level}, stockout_days={days_to_stockout}"
        )
        return item

    def deduct_stock(self, medicine_name: str, quantity: int) -> Inventory:
        """Deduct stock after an order is placed."""
        item = self.get_by_name(medicine_name)
        item.Stock_Available -= quantity
        self.session.flush()
        logger.info(f"Stock deducted: {medicine_name} -{quantity} (remaining: {item.Stock_Available})")
        return item

    def get_summary(self) -> dict:
        """Get inventory summary statistics."""
        total = self.session.query(func.count(Inventory.Medicine_Name)).scalar()
        low_stock = self.session.query(func.count(Inventory.Medicine_Name)).filter(
            Inventory.Stock_Available <= Inventory.Reorder_Level
        ).scalar()
        
        # Count expired medicines by comparing date strings
        all_items = self.get_all()
        expired = sum(1 for item in all_items if item.is_expired)
        
        return {
            "total_medicines": total or 0,
            "low_stock": low_stock or 0,
            "expired": expired,
        }

    def get_stock_data(self) -> pd.DataFrame:
        """Fetch medicine names and stock levels for display."""
        records = self.session.query(
            Inventory.Medicine_Name, 
            Inventory.Stock_Available
        ).all()
        return pd.DataFrame(records, columns=["Medicine_Name", "Stock_Available"])
