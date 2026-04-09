"""
FastAPI Application — Main entry point for the REST API.

Run with: uvicorn api.main:app --reload --port 8000
OpenAPI docs at: http://localhost:8000/docs
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from database.connection import init_db
from utils.logger import get_logger, set_correlation_id

from api.routes import auth, inventory, sales, orders, suppliers

logger = get_logger(__name__)


# --- Correlation ID + Timing Middleware ---
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with correlation ID and response time."""

    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID
        cid = set_correlation_id()

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)",
            extra={
                "endpoint": str(request.url.path),
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 1),
            },
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("Starting PharmaStock API...")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down PharmaStock API...")


app = FastAPI(
    title="PharmaStock Optimizer API",
    description=(
        "Industrial-grade REST API for pharmaceutical inventory management, "
        "sales analytics, and AI-powered stockout prediction."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware (order matters: last added = first executed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# --- Register route modules ---
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])
app.include_router(sales.router, prefix="/api/sales", tags=["Sales"])
app.include_router(orders.router, prefix="/api/orders", tags=["Orders"])
app.include_router(suppliers.router, prefix="/api/suppliers", tags=["Suppliers"])


@app.get("/", tags=["Health"])
def root():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_TITLE,
        "version": "2.0.0",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    from sqlalchemy import text
    from database.connection import engine
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "healthy",
        "database": db_status,
        "version": "2.0.0",
    }


@app.get("/api/ml/metrics", tags=["Machine Learning"])
def get_ml_metrics():
    """Get latest ML model training metrics (RMSE, MAE, R2, feature importance)."""
    from ml.forecasting import StockoutPredictor
    predictor = StockoutPredictor()
    metrics = predictor.get_model_metrics()
    if metrics is None:
        return {"message": "No models trained yet. Trigger via inventory update."}
    return metrics


@app.post("/api/ml/train", tags=["Machine Learning"])
def trigger_model_training():
    """Trigger retraining of all ML models."""
    from ml.forecasting import StockoutPredictor
    predictor = StockoutPredictor()
    models = predictor.train_all_models(force_retrain=True)
    metrics = predictor.get_model_metrics()
    return {
        "message": f"Trained {len(models)} models successfully",
        "metrics": metrics,
    }
