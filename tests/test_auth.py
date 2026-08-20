"""
Auth Service Tests — Registration, login, and account management.
"""

import pytest
from auth.service import AuthService
from auth.security import hash_password, verify_password
from utils.exceptions import AuthenticationError, UserAlreadyExistsError, UserNotFoundError


class TestPasswordSecurity:
    """Tests for bcrypt password hashing."""

    def test_hash_produces_different_hashes(self):
        """Same password should produce different hashes (salt)."""
        h1 = hash_password("test123")
        h2 = hash_password("test123")
        assert h1 != h2  # Different salts

    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        hashed = hash_password("mypassword")
        assert verify_password("mypassword", hashed) is True

    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hashed = hash_password("mypassword")
        assert verify_password("wrongpassword", hashed) is False


class TestAuthService:
    """Tests for authentication service."""

    def test_register_user(self, test_session):
        """Should register a new user successfully."""
        auth = AuthService(test_session)
        user = auth.register("testuser", "test@example.com", "pass123")
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == "user"

    def test_register_duplicate_raises(self, test_session):
        """Should raise error for duplicate username."""
        auth = AuthService(test_session)
        auth.register("testuser", "test1@example.com", "pass123")
        with pytest.raises(UserAlreadyExistsError):
            auth.register("testuser", "test2@example.com", "pass456")

    def test_login_success(self, test_session):
        """Should login with correct credentials."""
        auth = AuthService(test_session)
        auth.register("testuser", "test@example.com", "pass123")
        user = auth.login("testuser", "pass123")
        assert user.username == "testuser"

    def test_login_wrong_password(self, test_session):
        """Should raise error for wrong password."""
        auth = AuthService(test_session)
        auth.register("testuser", "test@example.com", "pass123")
        with pytest.raises(AuthenticationError):
            auth.login("testuser", "wrongpass")

    def test_login_nonexistent_user(self, test_session):
        """Should raise error for nonexistent user."""
        auth = AuthService(test_session)
        with pytest.raises(AuthenticationError):
            auth.login("ghostuser", "pass123")

    def test_delete_account(self, test_session):
        """Should delete account with correct credentials."""
        auth = AuthService(test_session)
        auth.register("testuser", "test@example.com", "pass123")
        result = auth.delete_account("testuser", "pass123")
        assert result is True

    def test_delete_account_wrong_password(self, test_session):
        """Should raise error when deleting with wrong password."""
        auth = AuthService(test_session)
        auth.register("testuser", "test@example.com", "pass123")
        with pytest.raises(AuthenticationError):
            auth.delete_account("testuser", "wrongpass")

    def test_register_admin(self, test_session):
        """Should register admin user with correct role."""
        auth = AuthService(test_session)
        user = auth.register("admin", "admin@example.com", "adminpass", role="admin")
        assert user.is_admin is True
