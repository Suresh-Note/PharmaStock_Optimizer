"""
Sales Service — Sales Data Queries and Filtering

Handles all sales data retrieval and filtering operations.
"""

from typing import Optional
import pandas as pd
from sqlalchemy.orm import Session

from database.models import Sale
from utils.logger import get_logger

logger = get_logger(__name__)


class SalesService:
    """Handles sales data queries and filtering."""

    def __init__(self, session: Session):
        self.session = session

    def get_all(self) -> pd.DataFrame:
        """Fetch all sales records as a DataFrame."""
        records = self.session.query(Sale).all()
        return self._to_dataframe(records)

    def get_by_date(self, date_str: str) -> pd.DataFrame:
        """Fetch sales for a specific date."""
        records = self.session.query(Sale).filter(Sale.Date == date_str).all()
        logger.info(f"Sales query by date: {date_str} ({len(records)} records)")
        return self._to_dataframe(records)

    def get_by_month_year(self, month_name: str, year: int) -> pd.DataFrame:
        """Fetch sales for a specific month and year."""
        records = self.session.query(Sale).filter(
            Sale.Year == year,
            Sale.Month == month_name,
        ).all()
        logger.info(f"Sales query by month/year: {month_name} {year} ({len(records)} records)")
        return self._to_dataframe(records)

    def get_by_year(self, year: int) -> pd.DataFrame:
        """Fetch all sales for a specific year."""
        records = self.session.query(Sale).filter(Sale.Year == year).all()
        logger.info(f"Sales query by year: {year} ({len(records)} records)")
        return self._to_dataframe(records)

    def get_medicine_names(self) -> list[str]:
        """Get sorted list of unique medicine names from sales data."""
        results = self.session.query(Sale.Medicine_Name).distinct().all()
        return sorted([r[0] for r in results if r[0]])

    def _to_dataframe(self, records: list[Sale]) -> pd.DataFrame:
        """Convert a list of Sale objects to a pandas DataFrame."""
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([{
            "Date": r.Date,
            "Month": r.Month,
            "Year": r.Year,
            "Medicine_Name": r.Medicine_Name,
            "ATC_Code": r.ATC_Code,
            "Stock_Available": r.Stock_Available,
            "Stock_Sold": r.Stock_Sold,
            "Stockout_Flag": r.Stockout_Flag,
            "External_Factor": r.External_Factor,
            "Price_per_Unit": r.Price_per_Unit,
            "Supplier_ID": r.Supplier_ID,
            "Reorder_Level": r.Reorder_Level,
            "Days_to_Stockout": r.Days_to_Stockout,
        } for r in records])
