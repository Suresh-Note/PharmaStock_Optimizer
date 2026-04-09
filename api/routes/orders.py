"""
Orders API Routes — Cart management and order placement.

Note: Cart state is stored server-side in a simple dict for API usage.
For production, this should use Redis or database-backed sessions.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from api.dependencies import get_db
from services.inventory import InventoryService
from utils.exceptions import MedicineNotFoundError, InsufficientStockError
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Server-side cart storage (in production, use Redis)
_carts: dict[str, dict] = {}


# --- Schemas ---

class AddToCartRequest(BaseModel):
    user: str = Field(examples=["johndoe"])
    medicine_name: str = Field(examples=["Aspirin"])
    quantity: int = Field(gt=0, examples=[5])


class PlaceOrderRequest(BaseModel):
    user: str = Field(examples=["johndoe"])


class CartItem(BaseModel):
    medicine_name: str
    quantity: int
    price_per_unit: float
    total: float


class CartResponse(BaseModel):
    items: list[CartItem]
    total_cost: float
    item_count: int


# --- Endpoints ---

@router.post(
    "/cart",
    response_model=CartResponse,
    summary="Add item to cart",
)
def add_to_cart(request: AddToCartRequest, db: Session = Depends(get_db)):
    """Add a medicine to the user's cart with stock validation."""
    try:
        inv_service = InventoryService(db)
        item = inv_service.get_by_name(request.medicine_name)
    except MedicineNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    cart = _carts.setdefault(request.user, {})
    existing_qty = cart.get(request.medicine_name, {}).get("quantity", 0)
    total_requested = existing_qty + request.quantity

    if total_requested > item.Stock_Available:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient stock for '{request.medicine_name}': "
                   f"requested {total_requested}, available {item.Stock_Available}",
        )

    if request.medicine_name in cart:
        cart[request.medicine_name]["quantity"] += request.quantity
    else:
        cart[request.medicine_name] = {
            "quantity": request.quantity,
            "price_per_unit": item.Price_per_Unit,
        }

    logger.info(f"Cart[{request.user}]: +{request.quantity} {request.medicine_name}")
    return _build_cart_response(request.user)


@router.get(
    "/cart/{user}",
    response_model=CartResponse,
    summary="View cart",
)
def view_cart(user: str):
    """Get current cart contents for a user."""
    return _build_cart_response(user)


@router.delete(
    "/cart/{user}",
    summary="Clear cart",
)
def clear_cart(user: str):
    """Clear all items from a user's cart."""
    _carts.pop(user, None)
    return {"message": "Cart cleared"}


@router.post(
    "/place",
    summary="Place order",
)
def place_order(request: PlaceOrderRequest, db: Session = Depends(get_db)):
    """Process the order: deduct stock for all cart items."""
    cart = _carts.get(request.user, {})
    if not cart:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cart is empty",
        )

    inv_service = InventoryService(db)
    total_cost = 0.0

    for medicine_name, details in cart.items():
        try:
            inv_service.deduct_stock(medicine_name, details["quantity"])
            total_cost += details["quantity"] * details["price_per_unit"]
        except MedicineNotFoundError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    item_count = len(cart)
    _carts.pop(request.user, None)
    logger.info(f"Order placed by {request.user}: {item_count} items, ₹{total_cost:.2f}")

    return {
        "message": "Order placed successfully",
        "items_ordered": item_count,
        "total_cost": round(total_cost, 2),
    }


def _build_cart_response(user: str) -> CartResponse:
    """Build a structured cart response."""
    cart = _carts.get(user, {})
    items = []
    total_cost = 0.0

    for medicine, details in cart.items():
        item_total = details["quantity"] * details["price_per_unit"]
        total_cost += item_total
        items.append(CartItem(
            medicine_name=medicine,
            quantity=details["quantity"],
            price_per_unit=details["price_per_unit"],
            total=round(item_total, 2),
        ))

    return CartResponse(
        items=items,
        total_cost=round(total_cost, 2),
        item_count=sum(i.quantity for i in items),
    )
