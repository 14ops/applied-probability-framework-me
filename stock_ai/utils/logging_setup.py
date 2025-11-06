"""
Logging Setup with Rotation

Configures loguru logger with rotating file handlers for Stock AI modules.
Logs are automatically rotated by date and size to prevent disk space issues.
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days",
    date_format: bool = True
) -> None:
    """
    Configure loguru logger with rotating file handlers.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: When to rotate logs (e.g., "10 MB", "1 day", "midnight")
        retention: How long to keep old logs (e.g., "30 days", "10 files")
        date_format: Whether to include date in filename
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File handler with date-based filename
    if date_format:
        log_file = log_path / f"stock_ai_{datetime.now().strftime('%Y%m%d')}.log"
    else:
        log_file = log_path / "stock_ai.log"
    
    # Rotating file handler
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression="zip",  # Compress old logs
        backtrace=True,     # Include stack trace for errors
        diagnose=True       # Include variable values in stack traces
    )
    
    logger.info(f"Logging configured: {log_file} (rotation: {rotation}, retention: {retention})")


def setup_live_trading_logging(
    log_dir: str = "logs",
    log_level: str = "INFO"
) -> None:
    """
    Configure logging specifically for live trading with per-date logs.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Daily rotating file handler for live trading
    log_file = log_path / "live_trading_{time:YYYY-MM-DD}.log"
    
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="00:00",  # Rotate at midnight
        retention="90 days",  # Keep 90 days of logs
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Live trading logging configured: {log_file}")

