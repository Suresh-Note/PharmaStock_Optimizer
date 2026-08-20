"""
Inventory API Routes — CRUD operations and stock management.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from api.dependencies import get_db
from services.inventory import InventoryService
from ml.forecasting import StockoutPredictor
from utils.exceptions import MedicineNotFoundError

router = APIRouter()


# --- Schemas ---

class InventoryItem(BaseModel):
    Medicine_Name: str
    ATC_Code: Optional[str] = None
    Stock_Available: int
    Price_per_Unit: float
    Reorder_Level: int
    Days_to_Stockout: Optional[int] = None
    Expiry_Date: Optional[str] = None

    class Config:
        from_attributes = True


class InventoryUpdateRequest(BaseModel):
    stock_available: int = Field(ge=0, examples=[100])
    price_per_unit: float = Field(gt=0, examples=[9.99])
    reorder_level: int = Field(ge=0, examples=[20])
    expiry_date: str = Field(examples=["2027-12-31"])


class InventorySummary(BaseModel):
    total_medicines: int
    low_stock: int
    expired: int


# --- Endpoints ---

@router.get(
    "/",
    response_model=list[InventoryItem],
    summary="List all inventory",
)
def list_inventory(db: Session = Depends(get_db)):
    """Fetch all inventory records."""
    service = InventoryService(db)
    items = service.get_all()
    return [InventoryItem(
        Medicine_Name=i.Medicine_Name,
        ATC_Code=i.ATC_Code,
        Stock_Available=i.Stock_Available,
        Price_per_Unit=i.Price_per_Unit,
        Reorder_Level=i.Reorder_Level,
        Days_to_Stockout=i.Days_to_Stockout,
        Expiry_Date=i.Expiry_Date,
    ) for i in items]


@router.get(
    "/summary",
    response_model=InventorySummary,
    summary="Get inventory statistics",
)
def inventory_summary(db: Session = Depends(get_db)):
    """Get summary: total medicines, low stock count, expired count."""
    service = InventoryService(db)
    summary = service.get_summary()
    return InventorySummary(**summary)


@router.get(
    "/{medicine_name}",
    response_model=InventoryItem,
    summary="Get single medicine",
)
def get_medicine(medicine_name: str, db: Session = Depends(get_db)):
    """Fetch a single inventory record by medicine name."""
    try:
        service = InventoryService(db)
        item = service.get_by_name(medicine_name)
        return InventoryItem(
            Medicine_Name=item.Medicine_Name,
            ATC_Code=item.ATC_Code,
            Stock_Available=item.Stock_Available,
            Price_per_Unit=item.Price_per_Unit,
            Reorder_Level=item.Reorder_Level,
            Days_to_Stockout=item.Days_to_Stockout,
            Expiry_Date=item.Expiry_Date,
        )
    except MedicineNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put(
    "/{medicine_name}",
    response_model=InventoryItem,
    summary="Update inventory",
)
def update_inventory(
    medicine_name: str,
    request: InventoryUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update stock, price, reorder level, and expiry. Triggers ML stockout prediction."""
    try:
        service = InventoryService(db)

        # ML prediction
        try:
            predictor = StockoutPredictor()
            days_to_stockout = predictor.predict_for_update(medicine_name, request.stock_available)
        except Exception:
            days_to_stockout = None

        item = service.update(
            medicine_name=medicine_name,
            stock_available=request.stock_available,
            price_per_unit=request.price_per_unit,
            reorder_level=request.reorder_level,
            expiry_date=request.expiry_date,
            days_to_stockout=days_to_stockout,
        )
        return InventoryItem(
            Medicine_Name=item.Medicine_Name,
            ATC_Code=item.ATC_Code,
            Stock_Available=item.Stock_Available,
            Price_per_Unit=item.Price_per_Unit,
            Reorder_Level=item.Reorder_Level,
            Days_to_Stockout=item.Days_to_Stockout,
            Expiry_Date=item.Expiry_Date,
        )
    except MedicineNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
