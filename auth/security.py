"""
Security Module — Password Hashing & Session Management

Provides bcrypt-based password hashing and Streamlit session management.
"""

import bcrypt
import streamlit as st
from datetime import datetime, timedelta

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt with automatic salting."""
    return bcrypt.hashpw(
        password.encode(), 
        bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
    ).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def init_session_state():
    """Initialize all required session state variables."""
    defaults = {
        "authenticated": False,
        "username": "",
        "role": "user",
        "login_time": None,
        "cart": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_authenticated(username: str, role: str = "user"):
    """Set the user as authenticated in session state."""
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.role = role
    st.session_state.login_time = datetime.now()
    logger.info(f"User '{username}' authenticated (role: {role})")


def clear_session():
    """Clear authentication from session state."""
    username = st.session_state.get("username", "unknown")
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = "user"
    st.session_state.login_time = None
    st.session_state.cart = {}
    logger.info(f"User '{username}' logged out")


def is_session_expired() -> bool:
    """Check if the current session has expired based on timeout setting."""
    login_time = st.session_state.get("login_time")
    if login_time is None:
        return True
    timeout = timedelta(minutes=settings.SESSION_TIMEOUT_MINUTES)
    return datetime.now() - login_time > timeout


def is_admin() -> bool:
    """Check if the current user has admin privileges."""
    return st.session_state.get("role") == "admin"
