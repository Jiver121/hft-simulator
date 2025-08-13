"""
Constants and enumerations for HFT Simulator

This module defines all constants, enumerations, and fixed values used
throughout the HFT Order Book Simulator.
"""

from enum import Enum, IntEnum
from typing import Dict, Any
import numpy as np

# =============================================================================
# ORDER BOOK CONSTANTS
# =============================================================================

class OrderSide(Enum):
    """Order side enumeration"""
    BID = "bid"
    ASK = "ask"
    BUY = "buy"    # Alternative naming
    SELL = "sell"  # Alternative naming

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    CANCEL = "cancel"
    MODIFY = "modify"

class OrderStatus(Enum):
    """Order status enumeration (aligned with realtime components)"""
    # Core lifecycle
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    PARTIAL = "partially_filled"  # alias for compatibility
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

    # Pending transitions
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    REPLACED = "replaced"

    # Fallback
    UNKNOWN = "unknown"

class TradeDirection(IntEnum):
    """Trade direction for performance tracking"""
    LONG = 1
    FLAT = 0
    SHORT = -1

# =============================================================================
# MARKET DATA CONSTANTS
# =============================================================================

# Standard market hours (US Eastern Time)
MARKET_OPEN_TIME = "09:30:00"
MARKET_CLOSE_TIME = "16:00:00"
PRE_MARKET_OPEN = "04:00:00"
POST_MARKET_CLOSE = "20:00:00"

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_TRADING_DAY = 6.5
MINUTES_PER_TRADING_DAY = 390
SECONDS_PER_TRADING_DAY = 23400

# Time conversions
MICROSECONDS_PER_SECOND = 1_000_000
MILLISECONDS_PER_SECOND = 1_000
NANOSECONDS_PER_MICROSECOND = 1_000

# =============================================================================
# FINANCIAL CONSTANTS
# =============================================================================

# Risk-free rates (annualized)
DEFAULT_RISK_FREE_RATE = 0.02  # 2%
TREASURY_BILL_RATE = 0.025     # 2.5%

# Standard deviations for risk calculations
CONFIDENCE_LEVELS = {
    0.90: 1.645,  # 90% confidence
    0.95: 1.960,  # 95% confidence
    0.99: 2.576,  # 99% confidence
}

# Basis points
BASIS_POINT = 0.0001  # 1 basis point = 0.01%
HALF_BASIS_POINT = 0.00005

# Common tick sizes
TICK_SIZES = {
    'penny': 0.01,
    'half_penny': 0.005,
    'nickel': 0.05,
    'dime': 0.10,
    'quarter': 0.25,
}

# =============================================================================
# STRATEGY CONSTANTS
# =============================================================================

class StrategyType(Enum):
    """Strategy type enumeration"""
    MARKET_MAKING = "market_making"
    LIQUIDITY_TAKING = "liquidity_taking"
    ARBITRAGE = "arbitrage"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CUSTOM = "custom"

# Default strategy parameters
DEFAULT_STRATEGY_PARAMS = {
    'max_position_size': 1000,
    'max_order_size': 100,
    'min_order_size': 1,
    'risk_limit': 10000.0,
    'max_drawdown': 0.05,
}

# Market making specific
MM_DEFAULT_PARAMS = {
    'min_spread': 0.01,
    'target_spread': 0.02,
    'max_spread': 0.05,
    'quote_size': 100,
    'max_inventory': 500,
    'inventory_penalty': 0.001,
}

# Liquidity taking specific
LT_DEFAULT_PARAMS = {
    'signal_threshold': 0.001,
    'aggression_level': 0.5,
    'max_market_impact': 0.002,
    'participation_rate': 0.1,
}

# =============================================================================
# PERFORMANCE METRICS CONSTANTS
# =============================================================================

class MetricType(Enum):
    """Performance metric types"""
    RETURN = "return"
    RISK = "risk"
    RATIO = "ratio"
    DRAWDOWN = "drawdown"
    VOLUME = "volume"
    EXECUTION = "execution"

# Standard performance metrics
PERFORMANCE_METRICS = [
    'total_return',
    'annualized_return',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'average_trade',
]

# Risk metrics
RISK_METRICS = [
    'value_at_risk',
    'expected_shortfall',
    'beta',
    'tracking_error',
    'information_ratio',
]

# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

# Standard column names for HFT data
STANDARD_COLUMNS = {
    'timestamp': 'timestamp',
    'symbol': 'symbol',
    'price': 'price',
    'volume': 'volume',
    'side': 'side',
    'order_type': 'order_type',
    'order_id': 'order_id',
    'exchange': 'exchange',
}

# Data validation limits
DATA_LIMITS = {
    'min_price': 0.0001,      # Minimum valid price
    'max_price': 1000000.0,   # Maximum valid price
    'min_volume': 1,          # Minimum valid volume
    'max_volume': 1000000,    # Maximum valid volume
    'max_spread_pct': 0.1,    # Maximum spread as % of mid-price
}

# Memory management
DEFAULT_CHUNK_SIZE = 100_000
MAX_MEMORY_USAGE = 0.8  # 80% of available memory

# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

# Color schemes
COLORS = {
    'bid': '#2E8B57',      # Sea Green
    'ask': '#DC143C',      # Crimson
    'trade': '#4169E1',    # Royal Blue
    'pnl_positive': '#228B22',  # Forest Green
    'pnl_negative': '#B22222',  # Fire Brick
    'neutral': '#708090',   # Slate Gray
}

# Plot styles
PLOT_STYLES = {
    'line_width': 1.5,
    'marker_size': 4,
    'alpha': 0.7,
    'grid_alpha': 0.3,
}

# Figure sizes (width, height)
FIGURE_SIZES = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 10),
    'wide': (16, 6),
    'tall': (8, 12),
}

# =============================================================================
# ERROR CODES AND MESSAGES
# =============================================================================

class ErrorCode(IntEnum):
    """Error codes for the simulator"""
    SUCCESS = 0
    INVALID_DATA = 1001
    INVALID_ORDER = 1002
    INSUFFICIENT_LIQUIDITY = 1003
    POSITION_LIMIT_EXCEEDED = 1004
    RISK_LIMIT_EXCEEDED = 1005
    MARKET_CLOSED = 1006
    SYSTEM_ERROR = 9999

ERROR_MESSAGES = {
    ErrorCode.SUCCESS: "Operation completed successfully",
    ErrorCode.INVALID_DATA: "Invalid or corrupted data",
    ErrorCode.INVALID_ORDER: "Invalid order parameters",
    ErrorCode.INSUFFICIENT_LIQUIDITY: "Insufficient market liquidity",
    ErrorCode.POSITION_LIMIT_EXCEEDED: "Position limit exceeded",
    ErrorCode.RISK_LIMIT_EXCEEDED: "Risk limit exceeded",
    ErrorCode.MARKET_CLOSED: "Market is closed",
    ErrorCode.SYSTEM_ERROR: "System error occurred",
}

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

# Precision settings
PRICE_PRECISION = 4      # Decimal places for prices
VOLUME_PRECISION = 0     # Decimal places for volumes
PNL_PRECISION = 2        # Decimal places for P&L
RATIO_PRECISION = 4      # Decimal places for ratios

# Numerical limits
EPSILON = 1e-10          # Small number for comparisons
MAX_FLOAT = np.finfo(np.float64).max
MIN_FLOAT = np.finfo(np.float64).min

# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Log message formats
LOG_FORMATS = {
    'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'simple': '%(asctime)s - %(levelname)s - %(message)s',
    'minimal': '%(levelname)s: %(message)s',
}

# =============================================================================
# UTILITY FUNCTIONS FOR CONSTANTS
# =============================================================================

def get_side_multiplier(side: OrderSide) -> int:
    """Get multiplier for order side (+1 for buy/bid, -1 for sell/ask)"""
    if side in [OrderSide.BID, OrderSide.BUY]:
        return 1
    elif side in [OrderSide.ASK, OrderSide.SELL]:
        return -1
    else:
        raise ValueError(f"Invalid order side: {side}")

def get_opposite_side(side: OrderSide) -> OrderSide:
    """Get the opposite side of an order"""
    if side == OrderSide.BID:
        return OrderSide.ASK
    elif side == OrderSide.ASK:
        return OrderSide.BID
    elif side == OrderSide.BUY:
        return OrderSide.SELL
    elif side == OrderSide.SELL:
        return OrderSide.BUY
    else:
        raise ValueError(f"Invalid order side: {side}")

def validate_price(price: float) -> bool:
    """Validate if price is within acceptable limits"""
    return DATA_LIMITS['min_price'] <= price <= DATA_LIMITS['max_price']

def validate_volume(volume: int) -> bool:
    """Validate if volume is within acceptable limits"""
    return DATA_LIMITS['min_volume'] <= volume <= DATA_LIMITS['max_volume']

def round_to_tick_size(price: float, tick_size: float = TICK_SIZES['penny']) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size

# =============================================================================
# EXPORT ALL CONSTANTS
# =============================================================================

__all__ = [
    # Enums
    'OrderSide', 'OrderType', 'OrderStatus', 'TradeDirection',
    'StrategyType', 'MetricType', 'ErrorCode', 'LogLevel',
    
    # Time constants
    'MARKET_OPEN_TIME', 'MARKET_CLOSE_TIME', 'PRE_MARKET_OPEN', 'POST_MARKET_CLOSE',
    'TRADING_DAYS_PER_YEAR', 'HOURS_PER_TRADING_DAY', 'MINUTES_PER_TRADING_DAY',
    'SECONDS_PER_TRADING_DAY', 'MICROSECONDS_PER_SECOND', 'MILLISECONDS_PER_SECOND',
    'NANOSECONDS_PER_MICROSECOND',
    
    # Financial constants
    'DEFAULT_RISK_FREE_RATE', 'TREASURY_BILL_RATE', 'CONFIDENCE_LEVELS',
    'BASIS_POINT', 'HALF_BASIS_POINT', 'TICK_SIZES',
    
    # Strategy constants
    'DEFAULT_STRATEGY_PARAMS', 'MM_DEFAULT_PARAMS', 'LT_DEFAULT_PARAMS',
    
    # Performance constants
    'PERFORMANCE_METRICS', 'RISK_METRICS',
    
    # Data constants
    'STANDARD_COLUMNS', 'DATA_LIMITS', 'DEFAULT_CHUNK_SIZE', 'MAX_MEMORY_USAGE',
    
    # Visualization constants
    'COLORS', 'PLOT_STYLES', 'FIGURE_SIZES',
    
    # Error constants
    'ERROR_MESSAGES',
    
    # Mathematical constants
    'PRICE_PRECISION', 'VOLUME_PRECISION', 'PNL_PRECISION', 'RATIO_PRECISION',
    'EPSILON', 'MAX_FLOAT', 'MIN_FLOAT',
    
    # Logging constants
    'LOG_FORMATS',
    
    # Utility functions
    'get_side_multiplier', 'get_opposite_side', 'validate_price', 'validate_volume',
    'round_to_tick_size',
]