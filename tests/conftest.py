"""
Test Fixtures — Shared test configuration and database setup.
"""

import pytest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="function")
def test_engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    return engine


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a test database session with all tables."""
    from database.connection import Base
    from database.models import User, Medicine, Inventory, Sale  # noqa: F401
    
    Base.metadata.create_all(bind=test_engine)
    Session = sessionmaker(bind=test_engine)
    session = Session()
    
    yield session
    
    session.close()
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def sample_inventory(test_session):
    """Seed test database with sample inventory data."""
    from database.models import Inventory
    
    items = [
        Inventory(
            Medicine_Name="Aspirin", ATC_Code="N02BA01",
            Stock_Available=100, Price_per_Unit=5.99,
            Reorder_Level=20, Days_to_Stockout=30,
            Expiry_Date="2027-12-31",
        ),
        Inventory(
            Medicine_Name="Ibuprofen", ATC_Code="M01AE01",
            Stock_Available=15, Price_per_Unit=8.50,
            Reorder_Level=20, Days_to_Stockout=5,
            Expiry_Date="2025-01-01",  # Already expired
        ),
        Inventory(
            Medicine_Name="Paracetamol", ATC_Code="N02BE01",
            Stock_Available=200, Price_per_Unit=3.25,
            Reorder_Level=50, Days_to_Stockout=60,
            Expiry_Date="2028-06-15",
        ),
    ]
    for item in items:
        test_session.add(item)
    test_session.flush()
    return items
