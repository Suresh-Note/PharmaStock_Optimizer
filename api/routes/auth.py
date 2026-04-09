"""
Auth API Routes — Registration, login, and account management.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from api.dependencies import get_db
from auth.service import AuthService
from utils.exceptions import (
    AuthenticationError,
    UserAlreadyExistsError,
    UserNotFoundError,
    EmailConfigurationError,
    EmailDeliveryError,
)

router = APIRouter()


# --- Request/Response Schemas ---

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, examples=["johndoe"])
    email: str = Field(examples=["john@pharmacy.com"])
    password: str = Field(min_length=6, examples=["SecurePass123"])
    role: str = Field(default="user", examples=["user"])


class LoginRequest(BaseModel):
    username: str = Field(examples=["johndoe"])
    password: str = Field(examples=["SecurePass123"])


class DeleteAccountRequest(BaseModel):
    username: str
    password: str


class RetrieveUsernameRequest(BaseModel):
    email: str


class UserResponse(BaseModel):
    username: str
    email: str
    role: str

    class Config:
        from_attributes = True


class AuthTokenResponse(BaseModel):
    username: str
    role: str
    message: str


# --- Endpoints ---

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account with bcrypt-hashed password."""
    try:
        auth_service = AuthService(db)
        user = auth_service.register(
            username=request.username,
            email=request.email,
            password=request.password,
            role=request.role,
        )
        return UserResponse(username=user.username, email=user.email, role=user.role)
    except UserAlreadyExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.post(
    "/login",
    response_model=AuthTokenResponse,
    summary="Authenticate user",
)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Verify credentials and return user info."""
    try:
        auth_service = AuthService(db)
        user = auth_service.login(request.username, request.password)
        return AuthTokenResponse(
            username=user.username,
            role=user.role,
            message="Login successful",
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )


@router.post(
    "/retrieve-username",
    summary="Recover username via email",
)
def retrieve_username(request: RetrieveUsernameRequest, db: Session = Depends(get_db)):
    """Send username to the registered email address."""
    try:
        auth_service = AuthService(db)
        auth_service.retrieve_username(request.email)
        return {"message": "Username sent to your email address"}
    except UserNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not found")
    except EmailConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Email service not configured",
        )
    except EmailDeliveryError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to send email",
        )


@router.delete(
    "/account",
    summary="Delete user account",
)
def delete_account(request: DeleteAccountRequest, db: Session = Depends(get_db)):
    """Delete account after credential verification."""
    try:
        auth_service = AuthService(db)
        auth_service.delete_account(request.username, request.password)
        return {"message": f"Account '{request.username}' deleted successfully"}
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
