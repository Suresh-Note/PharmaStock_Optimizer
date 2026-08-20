"""
Sales API Routes — Sales data querying with filters.
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from services.sales import SalesService

router = APIRouter()


@router.get(
    "/",
    summary="Get sales data with optional filters",
)
def get_sales(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-Mon-DD)", examples=["2024-Jan-15"]),
    month: Optional[str] = Query(None, description="Filter by month name", examples=["January"]),
    year: Optional[int] = Query(None, description="Filter by year", examples=[2024]),
    db: Session = Depends(get_db),
):
    """
    Fetch sales data with optional date/month/year filters.
    
    - No filters: returns all sales
    - `date`: returns sales for that specific date
    - `month` + `year`: returns sales for that month/year combo
    - `year` only: returns all sales for that year
    """
    service = SalesService(db)

    if date:
        df = service.get_by_date(date)
    elif month and year:
        df = service.get_by_month_year(month, year)
    elif year:
        df = service.get_by_year(year)
    else:
        df = service.get_all()

    if df.empty:
        return {"count": 0, "data": []}

    records = df.to_dict(orient="records")
    return {"count": len(records), "data": records}


@router.get(
    "/medicines",
    response_model=list[str],
    summary="Get unique medicine names from sales",
)
def get_medicine_names(db: Session = Depends(get_db)):
    """Returns sorted list of unique medicine names in sales data."""
    service = SalesService(db)
    return service.get_medicine_names()
