"""
Suppliers API Routes — Supplier-medicine stock data.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from api.dependencies import get_db
from services.suppliers import SupplierService

router = APIRouter()


class SupplierMedicine(BaseModel):
    medicine_name: str
    stock_available: int


class SupplierDetail(BaseModel):
    supplier_id: str
    medicines: list[SupplierMedicine]
    total_stock: int


@router.get(
    "/",
    response_model=list[SupplierDetail],
    summary="List all suppliers with medicines",
)
def list_suppliers(db: Session = Depends(get_db)):
    """Get all suppliers with their medicines and stock levels."""
    service = SupplierService(db)
    sorted_suppliers, supplier_mapping = service.get_supplier_mapping()

    result = []
    for supplier_id in sorted_suppliers:
        medicines = supplier_mapping.get(supplier_id, [])
        total_stock = sum(stock for _, stock in medicines)
        result.append(SupplierDetail(
            supplier_id=supplier_id,
            medicines=[
                SupplierMedicine(medicine_name=name, stock_available=stock)
                for name, stock in medicines
            ],
            total_stock=total_stock,
        ))

    return result


@router.get(
    "/{supplier_id}",
    response_model=SupplierDetail,
    summary="Get supplier details",
)
def get_supplier(supplier_id: str, db: Session = Depends(get_db)):
    """Get medicines and stock for a specific supplier."""
    service = SupplierService(db)
    _, supplier_mapping = service.get_supplier_mapping()

    medicines = supplier_mapping.get(supplier_id, [])
    if not medicines:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )

    total_stock = sum(stock for _, stock in medicines)
    return SupplierDetail(
        supplier_id=supplier_id,
        medicines=[
            SupplierMedicine(medicine_name=name, stock_available=stock)
            for name, stock in medicines
        ],
        total_stock=total_stock,
    )
