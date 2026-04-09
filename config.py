"""
PharmaStock Optimizer — Centralized Configuration

All application settings are managed here via Pydantic BaseSettings.
Values are loaded from environment variables and .env file.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application-wide configuration loaded from .env and environment variables."""

    # --- Database ---
    DB_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'pharmastock_optimizer.db'}",
        description="SQLAlchemy database connection URL"
    )

    # --- SMTP Email ---
    SMTP_EMAIL: str = Field(default="", description="Gmail account for sending emails")
    SMTP_PASSWORD: str = Field(default="", description="Gmail app password")
    SMTP_HOST: str = Field(default="smtp.gmail.com", description="SMTP server hostname")
    SMTP_PORT: int = Field(default=587, description="SMTP server port")

    # --- Authentication ---
    SESSION_TIMEOUT_MINUTES: int = Field(default=30, description="Session timeout in minutes")
    BCRYPT_ROUNDS: int = Field(default=12, description="bcrypt hashing rounds")

    # --- ML Pipeline ---
    MODEL_CACHE_TTL_HOURS: int = Field(default=1, description="XGBoost model cache TTL in hours")
    MAX_STOCKOUT_DAYS: int = Field(default=365, description="Max days for stockout prediction loop")
    XGBOOST_N_ESTIMATORS: int = Field(default=100, description="Number of XGBoost estimators")

    # --- Logging ---
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_DIR: str = Field(
        default=str(BASE_DIR / "logs"),
        description="Directory for log files"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        description="Log output format"
    )

    # --- App ---
    APP_TITLE: str = Field(default="PharmaStock Optimizer", description="Application title")
    APP_LAYOUT: str = Field(default="wide", description="Streamlit layout mode")

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton settings instance
settings = Settings()
