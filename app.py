"""
PharmaStock Optimizer — Application Entry Point

Thin Streamlit entry point that routes to page modules.
All business logic lives in the service layer.
"""

import streamlit as st

from config import settings
from auth.security import init_session_state, set_authenticated, clear_session, is_session_expired
from auth.service import AuthService
from database.connection import get_session, init_db
from utils.logger import get_logger
from utils.exceptions import (
    AuthenticationError,
    UserAlreadyExistsError,
    UserNotFoundError,
    EmailConfigurationError,
    EmailDeliveryError,
    PharmaStockError,
)

logger = get_logger(__name__)

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(
    layout=settings.APP_LAYOUT,
    page_title=settings.APP_TITLE,
    page_icon="💊",
)

# --- Custom Styling (animations + polish; safe to delete this block to revert) ---
st.markdown("""
<style>
.main .block-container, [data-testid="stAppViewBlockContainer"] {
    animation: ps-fade-in 0.4s ease-out;
}
@keyframes ps-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid^="stBaseButton"] {
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid^="stBaseButton"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(15, 118, 110, 0.25);
}
[data-testid="stMetric"] {
    background: #F0F4F3;
    border-radius: 10px;
    padding: 12px 16px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(15, 118, 110, 0.15);
}
[data-testid="stSidebar"] label {
    transition: background-color 0.15s ease;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize ---
init_session_state()

# Only initialize DB once per session (not every rerun)
if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# --- Page Registry ---
PAGES = {
    "Dashboard": "pages_ui.dashboard",
    "Inventory": "pages_ui.inventory_page",
    "Sales": "pages_ui.sales_page",
    "Medicines": "pages_ui.medicines_page",
    "Suppliers": "pages_ui.suppliers_page",
    "Model Insights": "pages_ui.model_insights",
}


def _render_login_page():
    """Render the authentication page (login/register/manage)."""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style='display: flex; align-items: center; min-height: 70vh; padding: 20px;'>
            <h1 style='font-size: clamp(2.5em, 5vw, 5em); line-height: 1.1; font-weight: 800; 
                       color: #1a1a2e; white-space: nowrap;'>
                PharmaStock<br>Optimizer
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='margin-top: 100px;'></div>", unsafe_allow_html=True)

        option = st.selectbox("Choose an option:", ["Login", "Register", "Manage Account"])

        if option == "Login":
            _login_form()
        elif option == "Register":
            _register_form()
        elif option == "Manage Account":
            _manage_account_form()


def _login_form():
    """Render login form."""
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        try:
            with get_session() as session:
                auth_service = AuthService(session)
                user = auth_service.login(username, password)
                set_authenticated(user.username, user.role)
                st.rerun()
        except AuthenticationError:
            st.error("Invalid username or password.")
        except Exception as e:
            logger.error(f"Login error: {e}")
            st.error("An unexpected error occurred. Please try again.")


def _register_form():
    """Render registration form."""
    st.subheader("Register")
    username = st.text_input("Choose a Username", key="register_username")
    email = st.text_input("Enter Your Email", key="register_email")
    password = st.text_input("Choose a Password", type="password", key="register_password")

    if st.button("Register"):
        try:
            with get_session() as session:
                auth_service = AuthService(session)
                auth_service.register(username, email, password)
            st.success("Registration successful! Please log in.")
        except UserAlreadyExistsError:
            st.error("Username already exists. Please choose a different one.")
        except Exception as e:
            logger.error(f"Registration error: {e}")
            st.error("An unexpected error occurred. Please try again.")


def _manage_account_form():
    """Render account management form."""
    st.subheader("Manage Account")
    action = st.selectbox("Select Action", ["Retrieve Username", "Delete Account"])

    if action == "Retrieve Username":
        email = st.text_input("Enter Your Email", key="retrieve_email")
        if st.button("Retrieve Username"):
            try:
                with get_session() as session:
                    auth_service = AuthService(session)
                    auth_service.retrieve_username(email)
                st.success("Your username has been sent to your email address.")
            except UserNotFoundError:
                st.error("Email not found.")
            except EmailConfigurationError:
                st.error("Email credentials not configured. Contact administrator.")
            except EmailDeliveryError:
                st.error("Failed to send email. Please try again later.")

    elif action == "Delete Account":
        username = st.text_input("Enter Your Username", key="delete_username")
        password = st.text_input("Enter Your Password", type="password", key="delete_password")
        if st.button("Delete Account"):
            try:
                with get_session() as session:
                    auth_service = AuthService(session)
                    auth_service.delete_account(username, password)
                st.success("Account deleted successfully.")
            except AuthenticationError:
                st.error("Invalid username or password.")


def _render_authenticated_app():
    """Render the authenticated application with navigation."""
    # Check session timeout
    if is_session_expired():
        clear_session()
        st.warning("Your session has expired. Please log in again.")
        st.rerun()
        return

    # Sidebar
    st.sidebar.title(settings.APP_TITLE)
    st.sidebar.button("Logout", on_click=clear_session)

    selected_page = st.sidebar.radio("Go to", list(PAGES.keys()))

    # Dynamic page loading
    try:
        import importlib
        module = importlib.import_module(PAGES[selected_page])
        module.render()
    except Exception as e:
        logger.error(f"Error rendering page '{selected_page}': {e}")
        st.error(f"Error loading page: {e}")


# --- Main Routing ---
if st.session_state.authenticated:
    _render_authenticated_app()
else:
    _render_login_page()
