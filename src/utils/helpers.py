"""
Helper utilities for HFT Simulator

This module contains common utility functions used throughout the HFT simulator
for data processing, validation, timing, and mathematical operations.
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import functools
from pathlib import Path
import hashlib
import json

from .constants import (
    MICROSECONDS_PER_SECOND, MILLISECONDS_PER_SECOND,
    PRICE_PRECISION, VOLUME_PRECISION, PNL_PRECISION,
    EPSILON, validate_price, validate_volume, round_to_tick_size
)


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """
    High-precision timer for performance measurement
    
    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do some work ...
        >>> elapsed_ms = timer.stop()
        >>> print(f"Operation took {elapsed_ms:.3f} ms")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in milliseconds"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        return (self.end_time - self.start_time) * 1000  # Convert to ms
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        return (time.perf_counter() - self.start_time) * 1000
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def timestamp_to_microseconds(timestamp: Union[str, datetime, pd.Timestamp]) -> int:
    """
    Convert timestamp to microseconds since epoch
    
    Args:
        timestamp: Timestamp in various formats
    
    Returns:
        Microseconds since epoch
    
    Example:
        >>> ts = "2023-01-01 09:30:00.123456"
        >>> microseconds = timestamp_to_microseconds(ts)
    """
    if isinstance(timestamp, str):
        dt = pd.to_datetime(timestamp)
    elif isinstance(timestamp, datetime):
        dt = pd.Timestamp(timestamp)
    elif isinstance(timestamp, pd.Timestamp):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return int(dt.timestamp() * MICROSECONDS_PER_SECOND)


def microseconds_to_timestamp(microseconds: int) -> pd.Timestamp:
    """
    Convert microseconds since epoch to pandas Timestamp
    
    Args:
        microseconds: Microseconds since epoch
    
    Returns:
        Pandas Timestamp
    """
    return pd.Timestamp(microseconds / MICROSECONDS_PER_SECOND, unit='s')


def format_duration(microseconds: int) -> str:
    """
    Format duration in microseconds to human-readable string
    
    Args:
        microseconds: Duration in microseconds
    
    Returns:
        Formatted duration string
    
    Example:
        >>> format_duration(1500000)  # 1.5 seconds
        '1.500s'
        >>> format_duration(2500)     # 2.5 milliseconds
        '2.500ms'
    """
    if microseconds >= MICROSECONDS_PER_SECOND:
        return f"{microseconds / MICROSECONDS_PER_SECOND:.3f}s"
    elif microseconds >= 1000:
        return f"{microseconds / 1000:.3f}ms"
    else:
        return f"{microseconds}μs"


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics
    
    Returns:
        Dictionary with memory usage in MB
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024 / 1024,
            'total': psutil.virtual_memory().total / 1024 / 1024,
        }
    except Exception:
        return {'error': 'Unable to get memory info'}


def check_memory_limit(max_usage_pct: float = 80.0) -> bool:
    """
    Check if memory usage is below the specified limit
    
    Args:
        max_usage_pct: Maximum memory usage percentage
    
    Returns:
        True if memory usage is below limit
    """
    try:
        memory_info = get_memory_usage()
        return memory_info.get('percent', 100) < max_usage_pct
    except Exception:
        return True  # Assume OK if can't check


def estimate_dataframe_memory(df: pd.DataFrame) -> float:
    """
    Estimate memory usage of a DataFrame in MB
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        Estimated memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024


# =============================================================================
# DATA VALIDATION UTILITIES
# =============================================================================

def validate_order_data(order: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate order data for completeness and correctness
    
    Args:
        order: Order dictionary with required fields
    
    Returns:
        Tuple of (is_valid, error_messages)
    
    Example:
        >>> order = {'price': 100.0, 'volume': 100, 'side': 'bid'}
        >>> is_valid, errors = validate_order_data(order)
    """
    errors = []
    
    # Required fields
    required_fields = ['price', 'volume', 'side']
    for field in required_fields:
        if field not in order:
            errors.append(f"Missing required field: {field}")
    
    # Validate price
    if 'price' in order:
        if not isinstance(order['price'], (int, float)):
            errors.append("Price must be numeric")
        elif not validate_price(float(order['price'])):
            errors.append("Price out of valid range")
    
    # Validate volume
    if 'volume' in order:
        if not isinstance(order['volume'], (int, float)):
            errors.append("Volume must be numeric")
        elif not validate_volume(int(order['volume'])):
            errors.append("Volume out of valid range")
        elif order['volume'] <= 0:
            errors.append("Volume must be positive")
    
    # Validate side
    if 'side' in order:
        valid_sides = ['bid', 'ask', 'buy', 'sell']
        if order['side'].lower() not in valid_sides:
            errors.append(f"Invalid side: {order['side']}")
    
    return len(errors) == 0, errors


def clean_price_data(prices: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
    """
    Clean price data by removing invalid values and outliers
    
    Args:
        prices: Array-like of prices
    
    Returns:
        Cleaned numpy array of prices
    """
    prices = np.array(prices, dtype=float)
    
    # Remove NaN and infinite values
    prices = prices[np.isfinite(prices)]
    
    # Remove negative prices
    prices = prices[prices > 0]
    
    # Remove extreme outliers (beyond 5 standard deviations)
    if len(prices) > 10:
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        lower_bound = mean_price - 5 * std_price
        upper_bound = mean_price + 5 * std_price
        prices = prices[(prices >= lower_bound) & (prices <= upper_bound)]
    
    return prices


def detect_outliers(data: Union[List, np.ndarray, pd.Series], 
                   method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data using various methods
    
    Args:
        data: Array-like data
        method: Method to use ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    data = np.array(data, dtype=float)
    data = data[np.isfinite(data)]  # Remove NaN/inf
    
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    
    elif method == 'modified_zscore':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def safe_divide(numerator: Union[float, pd.Series], denominator: Union[float, pd.Series], default: float = 0.0) -> Union[float, pd.Series]:
    """
    Safely divide two numbers or Series, returning default if denominator is zero
    
    Args:
        numerator: Numerator (scalar or pandas Series)
        denominator: Denominator (scalar or pandas Series)
        default: Default value if division by zero
    
    Returns:
        Result of division or default value
    """
    # Handle pandas Series
    if isinstance(denominator, pd.Series):
        # Create a mask for values that are close to zero
        near_zero_mask = np.abs(denominator) < EPSILON
        result = numerator / denominator
        result[near_zero_mask] = default
        return result
    elif isinstance(numerator, pd.Series):
        # If numerator is Series but denominator is scalar
        if abs(denominator) < EPSILON:
            return pd.Series([default] * len(numerator), index=numerator.index)
        return numerator / denominator
    else:
        # Handle scalar case
        if abs(denominator) < EPSILON:
            return default
        return numerator / denominator


def calculate_returns(prices: Union[List, np.ndarray, pd.Series], 
                     method: str = 'simple') -> np.ndarray:
    """
    Calculate returns from price series
    
    Args:
        prices: Array-like of prices
        method: Return calculation method ('simple', 'log')
    
    Returns:
        Array of returns
    
    Example:
        >>> prices = [100, 101, 99, 102]
        >>> returns = calculate_returns(prices, method='simple')
    """
    prices = np.array(prices, dtype=float)
    
    if len(prices) < 2:
        return np.array([])
    
    if method == 'simple':
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
    elif method == 'log':
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns


def rolling_statistics(data: Union[List, np.ndarray, pd.Series], 
                      window: int, 
                      stats: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Calculate rolling statistics for time series data
    
    Args:
        data: Array-like data
        window: Rolling window size
        stats: List of statistics to calculate ('mean', 'std', 'min', 'max', 'median')
    
    Returns:
        Dictionary of rolling statistics
    """
    if stats is None:
        stats = ['mean', 'std']
    
    data = pd.Series(data)
    results = {}
    
    for stat in stats:
        if stat == 'mean':
            results[stat] = data.rolling(window).mean().values
        elif stat == 'std':
            results[stat] = data.rolling(window).std().values
        elif stat == 'min':
            results[stat] = data.rolling(window).min().values
        elif stat == 'max':
            results[stat] = data.rolling(window).max().values
        elif stat == 'median':
            results[stat] = data.rolling(window).median().values
        else:
            raise ValueError(f"Unknown statistic: {stat}")
    
    return results


def exponential_moving_average(data: Union[List, np.ndarray, pd.Series], 
                              alpha: float) -> np.ndarray:
    """
    Calculate exponential moving average
    
    Args:
        data: Array-like data
        alpha: Smoothing factor (0 < alpha <= 1)
    
    Returns:
        Exponential moving average
    """
    if not 0 < alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    data = np.array(data, dtype=float)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_price(price: float, precision: int = PRICE_PRECISION) -> str:
    """
    Format price with appropriate precision
    
    Args:
        price: Price value
        precision: Number of decimal places
    
    Returns:
        Formatted price string
    """
    return f"{price:.{precision}f}"


def format_volume(volume: int) -> str:
    """
    Format volume with thousands separators
    
    Args:
        volume: Volume value
    
    Returns:
        Formatted volume string
    """
    return f"{volume:,}"


def format_pnl(pnl: float, precision: int = PNL_PRECISION) -> str:
    """
    Format P&L with appropriate sign and precision
    
    Args:
        pnl: P&L value
        precision: Number of decimal places
    
    Returns:
        Formatted P&L string with sign
    """
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0.05 = 5%)
        precision: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def format_large_number(number: float, precision: int = 1) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B)
    
    Args:
        number: Number to format
        precision: Number of decimal places
    
    Returns:
        Formatted number string
    """
    abs_number = abs(number)
    sign = "-" if number < 0 else ""
    
    if abs_number >= 1e9:
        return f"{sign}{abs_number/1e9:.{precision}f}B"
    elif abs_number >= 1e6:
        return f"{sign}{abs_number/1e6:.{precision}f}M"
    elif abs_number >= 1e3:
        return f"{sign}{abs_number/1e3:.{precision}f}K"
    else:
        return f"{sign}{abs_number:.{precision}f}"


# =============================================================================
# FILE AND DATA UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(filepath: Union[str, Path]) -> str:
    """
    Calculate MD5 hash of a file
    
    Args:
        filepath: Path to file
    
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file with proper formatting
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def memoize(func):
    """
    Simple memoization decorator for caching function results
    
    Example:
        >>> @memoize
        ... def expensive_calculation(x, y):
        ...     return x ** y
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from arguments
        key = str(args) + str(sorted(kwargs.items()))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    wrapper.cache = cache
    wrapper.cache_clear = cache.clear
    return wrapper


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation
    
    Example:
        >>> cache = LRUCache(maxsize=100)
        >>> cache.put("key1", "value1")
        >>> value = cache.get("key1")
    """
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
    
    def get(self, key: Any) -> Any:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update existing key
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


# =============================================================================
# EXPORT ALL UTILITIES
# =============================================================================

__all__ = [
    # Timing utilities
    'Timer', 'timestamp_to_microseconds', 'microseconds_to_timestamp', 'format_duration',
    
    # Memory utilities
    'get_memory_usage', 'check_memory_limit', 'estimate_dataframe_memory',
    
    # Data validation
    'validate_order_data', 'clean_price_data', 'detect_outliers',
    
    # Mathematical utilities
    'safe_divide', 'calculate_returns', 'rolling_statistics', 'exponential_moving_average',
    
    # Formatting utilities
    'format_price', 'format_volume', 'format_pnl', 'format_percentage', 'format_large_number',
    
    # File utilities
    'ensure_directory', 'get_file_hash', 'save_json', 'load_json',
    
    # Caching utilities
    'memoize', 'LRUCache',
]