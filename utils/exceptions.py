"""
Custom Exception Hierarchy

Structured exceptions for clean error handling across the application.
All exceptions inherit from PharmaStockError for catch-all handling.
"""


class PharmaStockError(Exception):
    """Base exception for all PharmaStock application errors."""
    
    def __init__(self, message: str = "An unexpected error occurred"):
        self.message = message
        super().__init__(self.message)


# --- Authentication Errors ---

class AuthenticationError(PharmaStockError):
    """Raised when login credentials are invalid."""
    
    def __init__(self, message: str = "Invalid username or password"):
        super().__init__(message)


class UserAlreadyExistsError(PharmaStockError):
    """Raised when attempting to register a duplicate username."""
    
    def __init__(self, username: str):
        super().__init__(f"Username '{username}' already exists")


class UserNotFoundError(PharmaStockError):
    """Raised when a user lookup fails."""
    
    def __init__(self, identifier: str):
        super().__init__(f"User not found: {identifier}")


class InsufficientPermissionError(PharmaStockError):
    """Raised when a non-admin attempts an admin-only action."""
    
    def __init__(self, action: str = "this action"):
        super().__init__(f"Admin privileges required for {action}")


# --- Inventory Errors ---

class MedicineNotFoundError(PharmaStockError):
    """Raised when a medicine lookup fails."""
    
    def __init__(self, medicine_name: str):
        super().__init__(f"Medicine not found: {medicine_name}")


class InsufficientStockError(PharmaStockError):
    """Raised when order quantity exceeds available stock."""
    
    def __init__(self, medicine_name: str, requested: int, available: int):
        self.medicine_name = medicine_name
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient stock for '{medicine_name}': "
            f"requested {requested}, available {available}"
        )


class ExpiredMedicineError(PharmaStockError):
    """Raised when attempting to order an expired medicine."""
    
    def __init__(self, medicine_name: str, expiry_date: str):
        super().__init__(f"Medicine '{medicine_name}' expired on {expiry_date}")


# --- Email Errors ---

class EmailConfigurationError(PharmaStockError):
    """Raised when SMTP credentials are not configured."""
    
    def __init__(self):
        super().__init__("SMTP credentials not configured. Set SMTP_EMAIL and SMTP_PASSWORD in .env")


class EmailDeliveryError(PharmaStockError):
    """Raised when email sending fails."""
    
    def __init__(self, recipient: str, reason: str = "unknown error"):
        super().__init__(f"Failed to send email to {recipient}: {reason}")


# --- ML Errors ---

class ModelNotTrainedError(PharmaStockError):
    """Raised when prediction is attempted without a trained model."""
    
    def __init__(self, medicine_name: str):
        super().__init__(f"No trained model available for '{medicine_name}'")
