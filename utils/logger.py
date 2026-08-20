"""
Industrial-Grade Logging System

Features:
- Console + rotating file output
- JSON structured logging for production
- Request correlation IDs
- Performance timing decorator
- Module-level and root logger configuration
"""

import logging
import logging.handlers
import json
import time
import uuid
import functools
from pathlib import Path
from contextvars import ContextVar
from typing import Callable, Any

from config import settings

# --- Correlation ID ---
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(cid: str | None = None) -> str:
    """Set a correlation ID for the current context (request tracing)."""
    cid = cid or uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    return _correlation_id.get("")


# --- JSON Formatter ---
class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation (ELK, CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        from datetime import datetime
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if present
        cid = get_correlation_id()
        if cid:
            log_data["correlation_id"] = cid

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        # Add extra fields
        for key in ("duration_ms", "medicine", "user", "endpoint", "status_code"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data, default=str)


# --- Console Formatter ---
class ColorFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        cid = get_correlation_id()
        cid_str = f" [{cid}]" if cid else ""

        record.msg = (
            f"{color}{record.levelname:<8}{self.RESET} | "
            f"{record.name:<24} |{cid_str} {record.msg}"
        )
        return super().format(record)


# --- Logger Setup ---
_configured = False


def _configure_root_logger():
    """Configure the root logger once with all handlers."""
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Remove default handlers
    root.handlers.clear()

    # 1. Console handler (colored for dev)
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    console.setLevel(logging.DEBUG)
    root.addHandler(console)

    # 2. Rotating file handler (JSON for production parsing)
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pharmastock.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    # 3. Error file handler (errors only, separate file)
    error_file = log_dir / "pharmastock_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        filename=error_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setFormatter(JSONFormatter())
    error_handler.setLevel(logging.ERROR)
    root.addHandler(error_handler)

    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger with console + file + error file handlers
    """
    _configure_root_logger()
    return logging.getLogger(name)


# --- Performance Decorator ---
def log_performance(func: Callable = None, *, logger_name: str = None) -> Callable:
    """
    Decorator that logs function execution time.
    
    Usage:
        @log_performance
        def my_function():
            ...
        
        @log_performance(logger_name="custom")
        def my_function():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        _logger = get_logger(logger_name or fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                duration = (time.perf_counter() - start) * 1000
                _logger.info(
                    f"{fn.__name__}() completed in {duration:.1f}ms",
                    extra={"duration_ms": round(duration, 1)},
                )
                return result
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                _logger.error(
                    f"{fn.__name__}() failed after {duration:.1f}ms: {e}",
                    extra={"duration_ms": round(duration, 1)},
                    exc_info=True,
                )
                raise

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
