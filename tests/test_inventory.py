"""
Inventory Service Tests — CRUD operations and summary statistics.
"""

import pytest
from services.inventory import InventoryService
from utils.exceptions import MedicineNotFoundError


class TestInventoryService:
    """Tests for inventory service operations."""

    def test_get_all(self, test_session, sample_inventory):
        """Should return all inventory items."""
        service = InventoryService(test_session)
        items = service.get_all()
        assert len(items) == 3

    def test_get_dataframe(self, test_session, sample_inventory):
        """Should return inventory as DataFrame with correct columns."""
        service = InventoryService(test_session)
        df = service.get_dataframe()
        assert len(df) == 3
        assert "Medicine_Name" in df.columns
        assert "Stock_Available" in df.columns

    def test_get_by_name(self, test_session, sample_inventory):
        """Should return specific medicine by name."""
        service = InventoryService(test_session)
        item = service.get_by_name("Aspirin")
        assert item.Stock_Available == 100
        assert item.Price_per_Unit == 5.99

    def test_get_by_name_not_found(self, test_session, sample_inventory):
        """Should raise error for nonexistent medicine."""
        service = InventoryService(test_session)
        with pytest.raises(MedicineNotFoundError):
            service.get_by_name("NonexistentDrug")

    def test_update(self, test_session, sample_inventory):
        """Should update inventory fields correctly."""
        service = InventoryService(test_session)
        item = service.update(
            medicine_name="Aspirin",
            stock_available=150,
            price_per_unit=6.99,
            reorder_level=25,
            expiry_date="2028-01-01",
            days_to_stockout=45,
        )
        assert item.Stock_Available == 150
        assert item.Price_per_Unit == 6.99
        assert item.Reorder_Level == 25
        assert item.Days_to_Stockout == 45

    def test_deduct_stock(self, test_session, sample_inventory):
        """Should correctly deduct stock quantity."""
        service = InventoryService(test_session)
        item = service.deduct_stock("Aspirin", 30)
        assert item.Stock_Available == 70

    def test_summary(self, test_session, sample_inventory):
        """Should calculate correct summary statistics."""
        service = InventoryService(test_session)
        summary = service.get_summary()
        assert summary["total_medicines"] == 3
        assert summary["low_stock"] == 1  # Ibuprofen: 15 <= 20
        assert summary["expired"] == 1    # Ibuprofen: 2025-01-01

    def test_is_low_stock_property(self, test_session, sample_inventory):
        """Should correctly identify low stock items."""
        service = InventoryService(test_session)
        aspirin = service.get_by_name("Aspirin")
        ibuprofen = service.get_by_name("Ibuprofen")
        assert aspirin.is_low_stock is False   # 100 > 20
        assert ibuprofen.is_low_stock is True  # 15 <= 20

    def test_is_expired_property(self, test_session, sample_inventory):
        """Should correctly identify expired items."""
        service = InventoryService(test_session)
        aspirin = service.get_by_name("Aspirin")
        ibuprofen = service.get_by_name("Ibuprofen")
        assert aspirin.is_expired is False      # 2027-12-31
        assert ibuprofen.is_expired is True     # 2025-01-01
