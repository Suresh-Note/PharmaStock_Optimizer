"""
Order Service — Cart Management and Order Processing

Handles shopping cart operations and inventory deduction on order placement.
"""

import streamlit as st
from sqlalchemy.orm import Session

from services.inventory import InventoryService
from utils.logger import get_logger
from utils.exceptions import InsufficientStockError

logger = get_logger(__name__)


class OrderService:
    """Handles cart operations and order processing."""

    def __init__(self, session: Session):
        self.session = session
        self.inventory_service = InventoryService(session)

    @staticmethod
    def get_cart() -> dict:
        """Get the current cart from session state."""
        if "cart" not in st.session_state:
            st.session_state.cart = {}
        return st.session_state.cart

    def add_to_cart(self, medicine_name: str, quantity: int, 
                    price_per_unit: float, stock_available: int):
        """
        Add an item to the cart with cumulative stock validation.
        
        Raises:
            InsufficientStockError: If total cart qty exceeds available stock
        """
        cart = self.get_cart()
        existing_qty = cart.get(medicine_name, {}).get("quantity", 0)
        total_requested = existing_qty + quantity

        if total_requested > stock_available:
            raise InsufficientStockError(medicine_name, total_requested, stock_available)

        if medicine_name in cart:
            cart[medicine_name]["quantity"] += quantity
        else:
            cart[medicine_name] = {
                "quantity": quantity,
                "price_per_unit": price_per_unit,
                "stock_available": stock_available,
            }
        
        logger.info(f"Cart: +{quantity} {medicine_name} (total: {cart[medicine_name]['quantity']})")

    def remove_from_cart(self, medicine_name: str):
        """Remove an item from the cart."""
        cart = self.get_cart()
        if medicine_name in cart:
            del cart[medicine_name]
            logger.info(f"Cart: removed {medicine_name}")

    def clear_cart(self):
        """Clear all items from the cart."""
        st.session_state.cart = {}
        logger.info("Cart cleared")

    def get_cart_summary(self) -> tuple[list[dict], float]:
        """
        Get formatted cart items and total cost.
        
        Returns:
            Tuple of (cart_items_list, total_cost)
        """
        cart = self.get_cart()
        cart_items = []
        total_cost = 0.0

        for medicine, details in cart.items():
            item_total = details["quantity"] * details["price_per_unit"]
            total_cost += item_total
            cart_items.append({
                "Medicine": medicine,
                "Quantity": details["quantity"],
                "Price per Unit": details["price_per_unit"],
                "Total": item_total,
            })

        return cart_items, total_cost

    def place_order(self) -> float:
        """
        Process the order: deduct stock for all cart items.
        
        Returns:
            Total order cost
            
        Raises:
            InsufficientStockError: If stock is insufficient at order time
        """
        cart = self.get_cart()
        if not cart:
            logger.warning("Attempted to place order with empty cart")
            return 0.0

        total_cost = 0.0
        for medicine_name, details in cart.items():
            self.inventory_service.deduct_stock(medicine_name, details["quantity"])
            total_cost += details["quantity"] * details["price_per_unit"]

        order_items = len(cart)
        self.clear_cart()
        logger.info(f"Order placed: {order_items} items, total ₹{total_cost:.2f}")
        return total_cost
