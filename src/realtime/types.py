"""
Shared Types for Real-Time Trading System

This module contains shared data structures used across the real-time trading
components to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from src.utils.constants import OrderSide, OrderType


class OrderPriority(Enum):
    """Order priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ExecutionAlgorithm(Enum):
    """Order execution algorithms"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    SNIPER = "sniper"


class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of risk violations"""
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    ORDER_SIZE_LIMIT = "order_size_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    CORRELATION_LIMIT = "correlation_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VELOCITY_LIMIT = "velocity_limit"


@dataclass
class OrderRequest:
    """Enhanced order request with execution parameters"""
    
    # Basic order information
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    
    # Execution parameters
    execution_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    priority: OrderPriority = OrderPriority.NORMAL
    time_in_force: str = "DAY"
    
    # Advanced parameters
    max_participation_rate: float = 0.1  # Max % of volume
    min_fill_size: int = 1
    max_show_size: Optional[int] = None  # For iceberg orders
    
    # Risk parameters
    max_slippage: float = 0.01  # 1%
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        import uuid
        if self.client_order_id is None:
            self.client_order_id = str(uuid.uuid4())


@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    average_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # 'long', 'short', 'flat'
    opened_at: datetime
    updated_at: datetime
    
    # Additional data
    cost_basis: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskViolation:
    """Record of a risk limit violation"""
    
    violation_type: ViolationType
    risk_level: RiskLevel
    
    # Details
    current_value: float
    limit_value: float
    excess_amount: float
    
    # Context
    symbol: Optional[str] = None
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Actions taken
    action_taken: str = ""
    auto_resolved: bool = False
    
    # Description
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskValidationResult:
    """Result of risk validation check"""
    
    approved: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[RiskViolation] = field(default_factory=list)
    
    # Additional info
    risk_score: float = 0.0  # 0-100 scale
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)