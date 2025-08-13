"""
Base Asset Class for Multi-Asset Trading System

This module defines the base asset class that provides a unified interface
for all asset types in the HFT simulator. It abstracts common functionality
while allowing asset-specific implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.engine.order_types import Order, Trade
from src.utils.constants import OrderSide, OrderType, PRICE_PRECISION


class AssetType(Enum):
    """Asset class enumeration"""
    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"
    FX = "fx"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    SYNTHETIC = "synthetic"


class TradingStatus(Enum):
    """Trading status for assets"""
    TRADING = "trading"
    HALTED = "halted"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    MAINTENANCE = "maintenance"


@dataclass
class AssetInfo:
    """Asset metadata and specifications"""
    symbol: str
    name: str
    asset_type: AssetType
    currency: str = "USD"
    tick_size: float = 0.01
    lot_size: int = 1
    min_trade_size: int = 1
    max_trade_size: Optional[int] = None
    
    # Market data
    trading_hours: Dict[str, Any] = field(default_factory=dict)
    exchanges: List[str] = field(default_factory=list)
    
    # Risk parameters
    margin_requirement: float = 0.0
    position_limit: Optional[int] = None
    daily_limit: Optional[float] = None
    
    # Asset-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAsset(ABC):
    """
    Base class for all tradeable assets
    
    Provides common functionality and interface for all asset types
    while allowing specialized implementations for each asset class.
    """
    
    def __init__(self, asset_info: AssetInfo):
        self.asset_info = asset_info
        self._trading_status = TradingStatus.TRADING
        self._current_price: Optional[float] = None
        self._last_trade_time: Optional[pd.Timestamp] = None
        self._daily_stats = self._initialize_daily_stats()
    
    def _initialize_daily_stats(self) -> Dict[str, Any]:
        """Initialize daily trading statistics"""
        return {
            'open_price': None,
            'high_price': None,
            'low_price': None,
            'volume': 0,
            'vwap': 0.0,
            'trade_count': 0,
            'last_price': None
        }
    
    @property
    def symbol(self) -> str:
        return self.asset_info.symbol
    
    @property
    def asset_type(self) -> AssetType:
        return self.asset_info.asset_type
    
    @property
    def current_price(self) -> Optional[float]:
        return self._current_price
    
    @property
    def trading_status(self) -> TradingStatus:
        return self._trading_status
    
    @property
    def daily_stats(self) -> Dict[str, Any]:
        return self._daily_stats.copy()
    
    # Abstract methods that must be implemented by asset-specific classes
    
    @abstractmethod
    def calculate_fair_value(self, **kwargs) -> float:
        """Calculate theoretical fair value of the asset"""
        pass
    
    @abstractmethod
    def get_risk_metrics(self, position_size: int = 0) -> Dict[str, float]:
        """Calculate risk metrics for given position size"""
        pass
    
    @abstractmethod
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate order against asset-specific rules"""
        pass
    
    # Common methods with default implementations
    
    def update_price(self, price: float, timestamp: pd.Timestamp = None) -> None:
        """Update current price and related statistics"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        if not self._is_valid_price(price):
            raise ValueError(f"Invalid price {price} for {self.symbol}")
        
        # Round to tick size
        price = self._round_to_tick_size(price)
        
        # Update current price
        self._current_price = price
        self._last_trade_time = timestamp
        
        # Update daily stats
        self._update_daily_stats(price)
    
    def process_trade(self, trade: Trade) -> None:
        """Process a completed trade and update statistics"""
        if trade.symbol != self.symbol:
            raise ValueError(f"Trade symbol {trade.symbol} doesn't match asset {self.symbol}")
        
        # Update price and stats
        self.update_price(trade.price, trade.timestamp)
        
        # Update volume-specific stats
        self._daily_stats['volume'] += trade.volume
        self._daily_stats['trade_count'] += 1
        
        # Update VWAP
        total_volume = self._daily_stats['volume']
        if total_volume > 0:
            current_vwap = self._daily_stats['vwap']
            self._daily_stats['vwap'] = (
                (current_vwap * (total_volume - trade.volume) + 
                 trade.price * trade.volume) / total_volume
            )
    
    def get_market_data_snapshot(self) -> Dict[str, Any]:
        """Get current market data snapshot"""
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type.value,
            'current_price': self._current_price,
            'timestamp': self._last_trade_time,
            'trading_status': self._trading_status.value,
            'daily_stats': self.daily_stats,
            'asset_info': {
                'tick_size': self.asset_info.tick_size,
                'lot_size': self.asset_info.lot_size,
                'currency': self.asset_info.currency
            }
        }
    
    def calculate_position_value(self, position_size: int, price: float = None) -> float:
        """Calculate total value of a position"""
        if price is None:
            price = self._current_price
        if price is None:
            raise ValueError(f"No price available for {self.symbol}")
        
        return abs(position_size) * price * self.asset_info.lot_size
    
    def calculate_margin_requirement(self, position_size: int) -> float:
        """Calculate margin requirement for position"""
        position_value = self.calculate_position_value(position_size)
        return position_value * self.asset_info.margin_requirement
    
    def is_tradeable(self, timestamp: pd.Timestamp = None) -> bool:
        """Check if asset is currently tradeable"""
        if self._trading_status != TradingStatus.TRADING:
            return False
        
        # Check trading hours if specified
        if timestamp and self.asset_info.trading_hours:
            return self._is_within_trading_hours(timestamp)
        
        return True
    
    def reset_daily_stats(self) -> None:
        """Reset daily trading statistics"""
        self._daily_stats = self._initialize_daily_stats()
    
    # Helper methods
    
    def _is_valid_price(self, price: float) -> bool:
        """Validate price against asset constraints"""
        return price > 0 and np.isfinite(price)
    
    def _round_to_tick_size(self, price: float) -> float:
        """Round price to nearest valid tick size"""
        tick_size = self.asset_info.tick_size
        return round(price / tick_size) * tick_size
    
    def _update_daily_stats(self, price: float) -> None:
        """Update daily price statistics"""
        stats = self._daily_stats
        
        # Set open price if not set
        if stats['open_price'] is None:
            stats['open_price'] = price
        
        # Update high/low
        if stats['high_price'] is None or price > stats['high_price']:
            stats['high_price'] = price
        if stats['low_price'] is None or price < stats['low_price']:
            stats['low_price'] = price
        
        # Update last price
        stats['last_price'] = price
    
    def _is_within_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours"""
        # This is a simplified implementation
        # Real implementation would handle timezone-aware trading hours
        trading_hours = self.asset_info.trading_hours
        if not trading_hours:
            return True
        
        current_time = timestamp.time()
        if 'start' in trading_hours and 'end' in trading_hours:
            start_time = pd.Timestamp(trading_hours['start']).time()
            end_time = pd.Timestamp(trading_hours['end']).time()
            return start_time <= current_time <= end_time
        
        return True
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.symbol})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(symbol='{self.symbol}', "
                f"type={self.asset_type.value}, price={self._current_price})")


class EquityAsset(BaseAsset):
    """Traditional equity/stock asset implementation"""
    
    def calculate_fair_value(self, **kwargs) -> float:
        """For equities, fair value is typically the current market price"""
        if self._current_price is None:
            raise ValueError(f"No current price available for {self.symbol}")
        return self._current_price
    
    def get_risk_metrics(self, position_size: int = 0) -> Dict[str, float]:
        """Calculate equity-specific risk metrics"""
        if self._current_price is None:
            return {}
        
        position_value = self.calculate_position_value(position_size)
        
        return {
            'position_value': position_value,
            'margin_requirement': self.calculate_margin_requirement(position_size),
            'daily_var': position_value * 0.02,  # Simplified 2% daily VaR
            'beta': self.asset_info.metadata.get('beta', 1.0)
        }
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate equity order"""
        if order.symbol != self.symbol:
            return False, f"Order symbol {order.symbol} doesn't match asset {self.symbol}"
        
        if order.volume <= 0:
            return False, "Order volume must be positive"
        
        if order.volume < self.asset_info.min_trade_size:
            return False, f"Order volume below minimum {self.asset_info.min_trade_size}"
        
        if (self.asset_info.max_trade_size and 
            order.volume > self.asset_info.max_trade_size):
            return False, f"Order volume exceeds maximum {self.asset_info.max_trade_size}"
        
        if order.order_type == OrderType.LIMIT and order.price:
            if not self._is_valid_price(order.price):
                return False, f"Invalid limit price {order.price}"
        
        return True, "Order validated successfully"
