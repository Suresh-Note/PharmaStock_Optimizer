"""
Database ORM Models

SQLAlchemy models mapped to the existing pharmastock_optimizer.db tables.
Column names match the existing database schema exactly.
"""

from datetime import datetime, date
from sqlalchemy import Column, String, Integer, Float, Text, Date, DateTime
from database.connection import Base


class User(Base):
    """Registered user account with role-based access."""
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)  # "admin" or "user"
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"

    @property
    def is_admin(self):
        return self.role == "admin"


class Medicine(Base):
    """Drug master data with supplier information."""
    __tablename__ = "Medicine"

    Medicine_Name = Column(String, primary_key=True, index=True)
    ATC_Code = Column(String)
    Price_per_Unit = Column(Float)
    Supplier_ID = Column(String)  # Comma-separated supplier IDs
    Expiry_Date = Column(String)  # Stored as text in existing DB

    def __repr__(self):
        return f"<Medicine(name='{self.Medicine_Name}', atc='{self.ATC_Code}')>"

    @property
    def suppliers_list(self):
        """Parse comma-separated supplier IDs into a list."""
        if self.Supplier_ID:
            return [s.strip() for s in self.Supplier_ID.split(",")]
        return []


class Inventory(Base):
    """Current stock levels and expiry tracking."""
    __tablename__ = "Inventory"

    Medicine_Name = Column(String, primary_key=True, index=True)
    ATC_Code = Column(String)
    Stock_Available = Column(Integer, default=0)
    Price_per_Unit = Column(Float)
    Reorder_Level = Column(Integer, default=0)
    Days_to_Stockout = Column(Integer)
    Expiry_Date = Column(String)  # Stored as text in existing DB

    def __repr__(self):
        return f"<Inventory(name='{self.Medicine_Name}', stock={self.Stock_Available})>"

    @property
    def is_low_stock(self):
        """Check if stock is below reorder level."""
        return self.Stock_Available <= self.Reorder_Level

    @property
    def is_expired(self):
        """Check if medicine has expired."""
        try:
            expiry = datetime.strptime(self.Expiry_Date, "%Y-%m-%d").date()
            return expiry < date.today()
        except (ValueError, TypeError):
            return False


class Sale(Base):
    """Historical sales record."""
    __tablename__ = "Sales"

    # Composite key workaround — SQLite doesn't enforce PKs well,
    # so we use rowid as implicit PK via autoincrement
    rowid = Column(Integer, primary_key=True, autoincrement=True)
    Date = Column(String)
    Month = Column(String)  # Full month name (e.g., "January")
    Year = Column(Integer)
    Medicine_Name = Column(String, index=True)
    ATC_Code = Column(String)
    Stock_Available = Column(Integer)
    Stock_Sold = Column(Integer)
    Stockout_Flag = Column(Integer)
    External_Factor = Column(String)
    Price_per_Unit = Column(Float)
    Supplier_ID = Column(String)
    Reorder_Level = Column(Integer)
    Days_to_Stockout = Column(Integer)

    def __repr__(self):
        return f"<Sale(date='{self.Date}', medicine='{self.Medicine_Name}', sold={self.Stock_Sold})>"
