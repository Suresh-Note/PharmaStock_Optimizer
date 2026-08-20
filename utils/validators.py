"""
Pydantic Validation Schemas

Input validation for all user-facing operations.
Ensures data integrity before it reaches the database.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator, EmailStr


# --- Auth Schemas ---

class UserRegister(BaseModel):
    """Schema for user registration."""
    username: str = Field(min_length=3, max_length=50, description="Unique username")
    email: str = Field(description="Valid email address")
    password: str = Field(min_length=6, max_length=100, description="Account password")

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").isalnum():
            raise ValueError("Username must contain only letters, numbers, and underscores")
        return v


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


# --- Inventory Schemas ---

class InventoryUpdate(BaseModel):
    """Schema for inventory update operations."""
    medicine_name: str = Field(description="Medicine to update")
    stock_available: int = Field(ge=0, description="New stock quantity")
    price_per_unit: float = Field(gt=0, description="Price per unit in currency")
    reorder_level: int = Field(ge=0, description="Minimum stock threshold")
    expiry_date: date = Field(description="Medicine batch expiry date")

    @field_validator("expiry_date")
    @classmethod
    def expiry_not_in_past(cls, v):
        if v < date.today():
            raise ValueError("Expiry date cannot be in the past for new stock")
        return v


# --- Order Schemas ---

class OrderItem(BaseModel):
    """Schema for a single item in an order."""
    medicine_name: str = Field(description="Medicine name")
    quantity: int = Field(gt=0, description="Number of units to order")
    price_per_unit: float = Field(gt=0, description="Unit price at time of order")
    stock_available: int = Field(ge=0, description="Available stock at time of order")


class OrderCart(BaseModel):
    """Schema for the entire cart."""
    items: dict[str, OrderItem] = Field(default_factory=dict)
    
    @property
    def total_cost(self) -> float:
        return sum(item.quantity * item.price_per_unit for item in self.items.values())
    
    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items.values())


# --- Sales Filter Schemas ---

class SalesDateFilter(BaseModel):
    """Schema for sales date filtering."""
    date: Optional[str] = Field(default=None, description="Specific date filter (YYYY-Mon-DD)")
    month: Optional[str] = Field(default=None, description="Month name filter")
    year: Optional[int] = Field(default=None, ge=2020, le=2035, description="Year filter")
