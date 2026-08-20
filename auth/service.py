"""
Authentication Service — User Registration, Login, and Account Management

Orchestrates all authentication operations using SQLAlchemy ORM.
"""

from sqlalchemy.orm import Session

from database.models import User
from auth.security import hash_password, verify_password
from auth.email import EmailService
from utils.logger import get_logger
from utils.exceptions import (
    AuthenticationError,
    UserAlreadyExistsError,
    UserNotFoundError,
)

logger = get_logger(__name__)


class AuthService:
    """Handles user authentication, registration, and account management."""

    def __init__(self, session: Session):
        self.session = session
        self.email_service = EmailService()

    def register(self, username: str, email: str, password: str, role: str = "user") -> User:
        """
        Register a new user account.
        
        Args:
            username: Unique username
            email: User email address
            password: Plain text password (will be hashed)
            role: User role ('user' or 'admin')
            
        Returns:
            Created User object
            
        Raises:
            UserAlreadyExistsError: If username is taken
        """
        existing = self.session.query(User).filter_by(username=username).first()
        if existing:
            raise UserAlreadyExistsError(username)

        user = User(
            username=username,
            email=email,
            password=hash_password(password),
            role=role,
        )
        self.session.add(user)
        self.session.flush()
        logger.info(f"User registered: {username} (role: {role})")
        return user

    def login(self, username: str, password: str) -> User:
        """
        Authenticate a user by credentials.
        
        Args:
            username: Username to authenticate
            password: Plain text password
            
        Returns:
            Authenticated User object
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = self.session.query(User).filter_by(username=username).first()
        if not user or not verify_password(password, user.password):
            logger.warning(f"Failed login attempt for: {username}")
            raise AuthenticationError()
        
        logger.info(f"User logged in: {username}")
        return user

    def retrieve_username(self, email: str) -> str:
        """
        Retrieve and email the username for an email address.
        
        Args:
            email: Email to look up
            
        Returns:
            The username found
            
        Raises:
            UserNotFoundError: If email is not registered
        """
        user = self.session.query(User).filter_by(email=email).first()
        if not user:
            raise UserNotFoundError(email)
        
        self.email_service.send_username_recovery(email, user.username)
        logger.info(f"Username recovery sent to: {email}")
        return user.username

    def delete_account(self, username: str, password: str) -> bool:
        """
        Delete a user account after credential verification.
        
        Args:
            username: Account to delete
            password: Password for verification
            
        Returns:
            True if deleted successfully
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = self.login(username, password)  # Verifies credentials
        self.session.delete(user)
        self.session.flush()
        logger.info(f"Account deleted: {username}")
        return True

    def get_user(self, username: str) -> User:
        """Fetch a user by username."""
        user = self.session.query(User).filter_by(username=username).first()
        if not user:
            raise UserNotFoundError(username)
        return user
