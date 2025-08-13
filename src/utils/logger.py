"""
Logging configuration for HFT Simulator

This module provides centralized logging configuration with support for
multiple log levels, file rotation, and performance monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time
import functools
from datetime import datetime

from config.settings import get_config
from .constants import LogLevel, LOG_FORMATS

# Global logger registry
_loggers: Dict[str, logging.Logger] = {}


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record):
        # Add timestamp in microseconds for high-precision timing
        record.timestamp_us = int(time.time() * 1_000_000)
        record.process_time = time.process_time()
        return True


class HFTFormatter(logging.Formatter):
    """Custom formatter for HFT simulator with enhanced information"""
    
    def __init__(self, fmt=None, datefmt=None, include_performance=False):
        super().__init__(fmt, datefmt)
        self.include_performance = include_performance
    
    def format(self, record):
        # Add custom fields
        if not hasattr(record, 'component'):
            record.component = record.name.split('.')[-1]
        
        # Add performance info if enabled
        if self.include_performance and hasattr(record, 'timestamp_us'):
            record.perf_info = f"[{record.timestamp_us}μs]"
        else:
            record.perf_info = ""
            
        # Manually format time to include microseconds
        record.asctime = self.formatTime(record, self.datefmt)
        if hasattr(record, 'created'):
            record.asctime += f".{int(record.created * 1000) % 1000:03d}"

        return super().format(record)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    include_performance: bool = False,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name (typically module name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        include_performance: Whether to include performance metrics
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger('order_book', level='DEBUG')
        >>> logger.info("Order book initialized")
        >>> logger.debug("Processing order: %s", order_id)
    """
    
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    # Get configuration
    config = get_config()
    
    # Set default values from config
    if level is None:
        level = config.logging.console_level
    if log_file is None and config.logging.log_file:
        log_file = config.logging.log_dir / config.logging.log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add performance filter if requested
    if include_performance:
        logger.addFilter(PerformanceFilter())
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        console_format = LOG_FORMATS['simple']
        if include_performance:
            console_format = '%(asctime)s %(perf_info)s - %(levelname)s - %(component)s - %(message)s'
        
        console_formatter = HFTFormatter(
            console_format,
            datefmt='%H:%M:%S',
            include_performance=include_performance
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Parse file size
        size_multipliers = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        size_str = max_file_size.upper()
        for suffix, multiplier in size_multipliers.items():
            if size_str.endswith(suffix):
                max_bytes = int(size_str[:-len(suffix)]) * multiplier
                break
        else:
            max_bytes = 10 * 1024**2  # Default 10MB
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.logging.file_level.upper()))
        
        file_format = LOG_FORMATS['detailed']
        if include_performance:
            file_format = '%(asctime)s %(perf_info)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        file_formatter = HFTFormatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S',
            include_performance=include_performance
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    else:
        return setup_logger(name)


def log_performance(func):
    """
    Decorator to log function execution time and performance metrics
    
    Example:
        >>> @log_performance
        ... def process_order_book_update(self, update):
        ...     # Process update
        ...     pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        start_time = time.perf_counter()
        start_process_time = time.process_time()
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate timing
            wall_time = (time.perf_counter() - start_time) * 1000  # ms
            cpu_time = (time.process_time() - start_process_time) * 1000  # ms
            
            logger.debug(
                f"{func.__name__} completed - Wall: {wall_time:.3f}ms, CPU: {cpu_time:.3f}ms"
            )
            
            return result
            
        except Exception as e:
            wall_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{func.__name__} failed after {wall_time:.3f}ms - Error: {str(e)}"
            )
            raise
    
    return wrapper


def log_memory_usage(func):
    """
    Decorator to log memory usage before and after function execution
    
    Requires psutil package for memory monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            # Memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            
            logger.debug(
                f"{func.__name__} memory usage - Before: {mem_before:.1f}MB, "
                f"After: {mem_after:.1f}MB, Delta: {mem_delta:+.1f}MB"
            )
            
            return result
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Memory monitoring failed for {func.__name__}: {str(e)}")
            return func(*args, **kwargs)
    
    return wrapper


class LogContext:
    """
    Context manager for adding context to log messages
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, order_id="12345", symbol="AAPL"):
        ...     logger.info("Processing order")  # Will include order_id and symbol
    """
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


class StructuredLogger:
    """
    Structured logger for consistent log formatting with key-value pairs
    
    Example:
        >>> slogger = StructuredLogger('trading')
        >>> slogger.info("Order executed", 
        ...               order_id="12345", 
        ...               symbol="AAPL", 
        ...               price=150.25, 
        ...               quantity=100)
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with structured key-value pairs"""
        if kwargs:
            kv_pairs = [f"{k}={v}" for k, v in kwargs.items()]
            return f"{message} | {' | '.join(kv_pairs)}"
        return message
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(self._format_message(message, **kwargs))


# Convenience function to set up the main application logger
def setup_main_logger() -> logging.Logger:
    """Set up the main application logger with default configuration"""
    config = get_config()
    
    return setup_logger(
        'hft_simulator',
        level=config.logging.console_level,
        log_file=config.logging.log_dir / config.logging.log_file,
        console_output=True,
        include_performance=config.logging.log_performance,
        max_file_size=config.logging.max_file_size,
        backup_count=config.logging.backup_count
    )


# Module-level convenience functions
def debug(message: str, *args, **kwargs):
    """Log debug message using main logger"""
    logger = get_logger('hft_simulator')
    logger.debug(message, *args, **kwargs)

def info(message: str, *args, **kwargs):
    """Log info message using main logger"""
    logger = get_logger('hft_simulator')
    logger.info(message, *args, **kwargs)

def warning(message: str, *args, **kwargs):
    """Log warning message using main logger"""
    logger = get_logger('hft_simulator')
    logger.warning(message, *args, **kwargs)

def error(message: str, *args, **kwargs):
    """Log error message using main logger"""
    logger = get_logger('hft_simulator')
    logger.error(message, *args, **kwargs)

def critical(message: str, *args, **kwargs):
    """Log critical message using main logger"""
    logger = get_logger('hft_simulator')
    logger.critical(message, *args, **kwargs)


# Initialize main logger on module import
_main_logger = setup_main_logger()