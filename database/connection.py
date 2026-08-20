"""
Database Connection — SQLAlchemy Engine & Session Factory

Provides a centralized database connection layer using SQLAlchemy ORM.
All database access throughout the application goes through this module.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from contextlib import contextmanager

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


# Create engine with SQLite-specific optimizations
engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,
    pool_pre_ping=True,
)

# Enable WAL mode for better concurrent read performance on SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session() -> Session:
    """
    Context manager that provides a transactional database session.
    
    Usage:
        with get_session() as session:
            results = session.query(Model).all()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables defined in ORM models."""
    from database.models import User, Medicine, Inventory, Sale  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")
