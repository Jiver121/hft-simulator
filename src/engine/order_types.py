"""
Order Types and Data Structures for HFT Simulator

This module defines the fundamental data structures used in the order book engine,
including orders, trades, and market updates.

Educational Notes:
- Orders represent intentions to buy or sell at specific prices
- Trades represent completed transactions between buyers and sellers
- Order updates track changes to the order book state
- These structures form the foundation of all market activity
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
from enum import Enum
import uuid
import threading

from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.utils.helpers import format_price, format_volume

# Global order ID registry for uniqueness validation
_order_id_registry = set()
_registry_lock = threading.Lock()


def validate_order_id_uniqueness(order_id: str, allow_duplicates_in_tests: bool = False) -> bool:
    """
    Validate that an order ID is unique across the system
    
    Args:
        order_id: The order ID to validate
        allow_duplicates_in_tests: Whether to allow duplicates in test environment
        
    Returns:
        True if the ID is unique, False otherwise
        
    Raises:
        ValueError: If the order ID is not unique and duplicates are not allowed
    """
    with _registry_lock:
        if order_id in _order_id_registry:
            if not allow_duplicates_in_tests:
                raise ValueError(f"Duplicate order ID detected: {order_id}. "
                               "Order IDs must be unique across the system.")
            return False
        
        _order_id_registry.add(order_id)
        return True


def remove_order_id_from_registry(order_id: str) -> None:
    """
    Remove an order ID from the global registry (when order is cancelled or filled)
    
    Args:
        order_id: The order ID to remove
    """
    with _registry_lock:
        _order_id_registry.discard(order_id)


def clear_order_id_registry() -> None:
    """
    Clear the order ID registry (useful for testing)
    """
    with _registry_lock:
        _order_id_registry.clear()


def get_order_id_registry_size() -> int:
    """
    Get the current size of the order ID registry
    
    Returns:
        Number of order IDs currently in the registry
    """
    with _registry_lock:
        return len(_order_id_registry)


@dataclass(init=False)
class Order:
    """
    Represents a single order in the order book
    
    This is the fundamental unit of market activity. Each order represents
    an intention to buy or sell a specific quantity at a specific price.
    
    Educational Notes:
    - order_id: Unique identifier for tracking the order
    - symbol: The financial instrument being traded
    - side: Whether this is a buy (bid) or sell (ask) order
    - order_type: Market orders execute immediately, limit orders wait for price
    - price: The price at which the order should execute (None for market orders)
    - volume: The quantity to buy or sell
    - timestamp: When the order was created
    - status: Current state of the order (pending, filled, cancelled, etc.)
    """
    
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    volume: int
    timestamp: pd.Timestamp
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: int = 0
    remaining_volume: int = field(init=False)
    average_fill_price: float = 0.0
    
    # Metadata
    source: str = "simulator"  # Source of the order (strategy name, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: "OrderSide",
        order_type: "OrderType",
        volume: Optional[int] = None,
        timestamp: Optional[pd.Timestamp] = None,
        price: Optional[float] = None,
        status: "OrderStatus" = None,
        filled_volume: int = 0,
        *,
        # Aliases / optional extras
        quantity: Optional[int] = None,
        source: str = "simulator",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Custom initializer to support both volume and quantity parameters.
        
        Args:
            order_id: Unique identifier for the order
            symbol: Trading symbol
            side: Buy or sell side
            order_type: Market or limit order type
            volume: Primary volume/quantity parameter
            timestamp: Order creation timestamp (defaults to now)
            price: Order price (required for limit orders)
            status: Order status (defaults to PENDING)
            filled_volume: Already filled volume (defaults to 0)
            quantity: Alias for volume parameter (keyword-only)
            source: Source identifier (defaults to "simulator")
            metadata: Additional order metadata
            
        Note:
            Either volume or quantity must be provided. If both are provided,
            volume takes precedence. Both parameters are aliases for the same value.
        """
        # Handle volume/quantity parameter aliases
        if volume is not None and quantity is not None:
            # Both provided - use volume and ignore quantity with warning
            import warnings
            warnings.warn(
                "Both 'volume' and 'quantity' provided. Using 'volume' parameter. "
                "Consider using only one parameter to avoid confusion.",
                UserWarning
            )
            final_volume = int(volume)
        elif volume is not None:
            final_volume = int(volume)
        elif quantity is not None:
            final_volume = int(quantity)
        else:
            raise ValueError("Order requires either a 'volume' or 'quantity' parameter")

        # Validate parameters before assignment
        if final_volume <= 0:
            raise ValueError(f"Order volume/quantity must be positive, got {final_volume}")
        
        if price is not None and price <= 0:
            raise ValueError(f"Order price must be positive, got {price}")
        
        if filled_volume < 0:
            raise ValueError(f"Filled volume cannot be negative, got {filled_volume}")
        
        if filled_volume > final_volume:
            raise ValueError(f"Filled volume ({filled_volume}) cannot exceed total volume ({final_volume})")

        # Assign validated values
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.volume = final_volume
        self.timestamp = timestamp or pd.Timestamp.now()
        self.price = price
        self.status = status or OrderStatus.PENDING
        self.filled_volume = int(filled_volume)
        self.remaining_volume = 0  # set in __post_init__
        self.average_fill_price = 0.0
        self.source = source
        self.metadata = metadata or {}

        self.__post_init__()

    def __post_init__(self):
        """Initialize computed fields after object creation"""
        self.remaining_volume = self.volume - self.filled_volume
        
        # Validate order type constraints
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must have a price")
        
        if self.order_type == OrderType.MARKET and self.price is not None:
            # Market orders shouldn't have prices, but we'll allow it for flexibility
            pass
    
    @classmethod
    def create_market_order(cls, symbol: str, side: OrderSide, volume: int = None, 
                           timestamp: pd.Timestamp = None, *,
                           quantity: int = None, **kwargs) -> 'Order':
        """
        Create a market order (executes immediately at best available price)
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            volume: Quantity (use either volume or quantity)
            timestamp: Order timestamp (defaults to now)
            quantity: Alias for volume parameter (keyword-only)
            **kwargs: Additional order parameters
            
        Returns:
            Market order instance
            
        Note:
            Either volume or quantity must be provided.
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Handle volume/quantity parameter
        final_volume = volume if volume is not None else quantity
        if final_volume is None:
            raise ValueError("Either 'volume' or 'quantity' parameter must be provided")
        
        return cls(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=final_volume,
            timestamp=timestamp,
            price=None,
            **kwargs
        )
    
    @classmethod
    def create_limit_order(cls, symbol: str, side: OrderSide, volume: int = None, 
                          price: float = None, timestamp: pd.Timestamp = None, *,
                          quantity: int = None, **kwargs) -> 'Order':
        """
        Create a limit order (executes only at specified price or better)
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            volume: Quantity (use either volume or quantity)
            price: Limit price
            timestamp: Order timestamp (defaults to now)
            quantity: Alias for volume parameter (keyword-only)
            **kwargs: Additional order parameters
            
        Returns:
            Limit order instance
            
        Note:
            Either volume or quantity must be provided, and price is required.
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Handle volume/quantity parameter
        final_volume = volume if volume is not None else quantity
        if final_volume is None:
            raise ValueError("Either 'volume' or 'quantity' parameter must be provided")
        
        if price is None:
            raise ValueError("Limit orders require a 'price' parameter")
        
        return cls(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            volume=final_volume,
            price=price,
            timestamp=timestamp,
            **kwargs
        )
    
    def is_buy(self) -> bool:
        """Check if this is a buy order"""
        return self.side in [OrderSide.BID, OrderSide.BUY]
    
    def is_sell(self) -> bool:
        """Check if this is a sell order"""
        return self.side in [OrderSide.ASK, OrderSide.SELL]
    
    def is_market_order(self) -> bool:
        """Check if this is a market order"""
        return self.order_type == OrderType.MARKET
    
    def is_limit_order(self) -> bool:
        """Check if this is a limit order"""
        return self.order_type == OrderType.LIMIT
    
    def is_active(self) -> bool:
        """Check if order is still active (can be filled)"""
        # An order is active if it has a valid status AND has remaining volume to fill
        return (self.status in [OrderStatus.PENDING, OrderStatus.PARTIAL] and 
                self.remaining_volume > 0)
    
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED or self.remaining_volume == 0
    
    def can_match_price(self, price: float) -> bool:
        """
        Check if this order can match at the given price
        
        Args:
            price: Price to check against
            
        Returns:
            True if order can execute at this price
        """
        if self.is_market_order():
            return True
        
        if self.is_buy():
            # Buy orders can match at or below their limit price
            return price <= self.price
        else:
            # Sell orders can match at or above their limit price
            return price >= self.price
    
    def partial_fill(self, fill_volume: int, fill_price: float) -> 'Trade':
        """
        Partially fill this order
        
        Args:
            fill_volume: Volume to fill
            fill_price: Price of the fill
            
        Returns:
            Trade object representing the fill
        """
        if fill_volume <= 0:
            raise ValueError("Fill volume must be positive")
        
        if fill_volume > self.remaining_volume:
            raise ValueError("Fill volume exceeds remaining volume")
        
        # Update order state
        old_filled_volume = self.filled_volume
        self.filled_volume += fill_volume
        self.remaining_volume = self.volume - self.filled_volume
        
        # Update average fill price
        total_fill_value = (old_filled_volume * self.average_fill_price) + (fill_volume * fill_price)
        self.average_fill_price = total_fill_value / self.filled_volume
        
        # Update status
        if self.remaining_volume == 0:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIAL
        
        # Create trade record
        return Trade(
            trade_id=str(uuid.uuid4()),
            symbol=self.symbol,
            buy_order_id=self.order_id if self.is_buy() else None,
            sell_order_id=self.order_id if self.is_sell() else None,
            price=fill_price,
            volume=fill_volume,
            timestamp=pd.Timestamp.now(),
            aggressor_side=self.side
        )
    
    def cancel(self) -> None:
        """Cancel this order"""
        if self.is_active():
            self.status = OrderStatus.CANCELLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'volume': self.volume,
            'filled_volume': self.filled_volume,
            'remaining_volume': self.remaining_volume,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'average_fill_price': self.average_fill_price,
            'source': self.source,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the order"""
        price_str = format_price(self.price) if self.price else "MKT"
        return (f"Order({self.order_id[:8]}): {self.side.value.upper()} "
                f"{format_volume(self.volume)} {self.symbol} @ {price_str} "
                f"[{self.status.value}]")
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class Trade:
    """
    Represents a completed trade between two orders
    
    Educational Notes:
    - Trades occur when a buy order matches with a sell order
    - The trade price is determined by the order book matching rules
    - Aggressor side indicates which order initiated the trade
    - Trade records are essential for performance analysis
    """
    
    trade_id: str
    symbol: str
    price: float
    volume: int
    timestamp: pd.Timestamp
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    aggressor_side: Optional[OrderSide] = None
    
    # Trade metadata
    trade_type: str = "normal"  # normal, opening, closing, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trade parameters"""
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.volume <= 0:
            raise ValueError("Trade volume must be positive")
    
    @property
    def trade_value(self) -> float:
        """Calculate the total value of the trade"""
        return self.price * self.volume
    
    def is_buy_aggressor(self) -> bool:
        """Check if the buy side was the aggressor"""
        return self.aggressor_side in [OrderSide.BID, OrderSide.BUY]
    
    def is_sell_aggressor(self) -> bool:
        """Check if the sell side was the aggressor"""
        return self.aggressor_side in [OrderSide.ASK, OrderSide.SELL]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary representation"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'trade_value': self.trade_value,
            'timestamp': self.timestamp,
            'buy_order_id': self.buy_order_id,
            'sell_order_id': self.sell_order_id,
            'aggressor_side': self.aggressor_side.value if self.aggressor_side else None,
            'trade_type': self.trade_type,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the trade"""
        aggressor = f" ({self.aggressor_side.value} aggressor)" if self.aggressor_side else ""
        return (f"Trade({self.trade_id[:8]}): {format_volume(self.volume)} "
                f"{self.symbol} @ {format_price(self.price)}{aggressor}")
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class OrderUpdate:
    """
    Represents an update to the order book state
    
    Educational Notes:
    - Order updates track all changes to the order book
    - They can represent new orders, cancellations, modifications, or trades
    - These updates are used to reconstruct the order book state over time
    - In real systems, these would come from exchange feeds
    """
    
    update_id: str
    symbol: str
    timestamp: pd.Timestamp
    update_type: str  # 'new', 'cancel', 'modify', 'trade'
    
    # Order information
    order_id: Optional[str] = None
    side: Optional[OrderSide] = None
    price: Optional[float] = None
    volume: Optional[int] = None
    
    # Trade information (for trade updates)
    trade_id: Optional[str] = None
    trade_price: Optional[float] = None
    trade_volume: Optional[int] = None
    
    # Metadata
    source: str = "simulator"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def new_order(cls, order: Order) -> 'OrderUpdate':
        """Create an update for a new order"""
        return cls(
            update_id=str(uuid.uuid4()),
            symbol=order.symbol,
            timestamp=order.timestamp,
            update_type='new',
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            volume=order.volume,
            source=order.source
        )
    
    @classmethod
    def cancel_order(cls, order: Order, timestamp: pd.Timestamp = None) -> 'OrderUpdate':
        """Create an update for order cancellation"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        return cls(
            update_id=str(uuid.uuid4()),
            symbol=order.symbol,
            timestamp=timestamp,
            update_type='cancel',
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            volume=order.remaining_volume,
            source=order.source
        )
    
    @classmethod
    def trade_update(cls, trade: Trade) -> 'OrderUpdate':
        """Create an update for a trade"""
        return cls(
            update_id=str(uuid.uuid4()),
            symbol=trade.symbol,
            timestamp=trade.timestamp,
            update_type='trade',
            trade_id=trade.trade_id,
            trade_price=trade.price,
            trade_volume=trade.volume,
            source="trade_engine"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert update to dictionary representation"""
        return {
            'update_id': self.update_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'update_type': self.update_type,
            'order_id': self.order_id,
            'side': self.side.value if self.side else None,
            'price': self.price,
            'volume': self.volume,
            'trade_id': self.trade_id,
            'trade_price': self.trade_price,
            'trade_volume': self.trade_volume,
            'source': self.source,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation of the update"""
        if self.update_type == 'new':
            return f"OrderUpdate: NEW {self.side.value} {self.volume} @ {format_price(self.price)}"
        elif self.update_type == 'cancel':
            return f"OrderUpdate: CANCEL {self.order_id[:8]}"
        elif self.update_type == 'trade':
            return f"OrderUpdate: TRADE {self.trade_volume} @ {format_price(self.trade_price)}"
        else:
            return f"OrderUpdate: {self.update_type.upper()}"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class PriceLevel:
    """
    Represents a single price level in the order book
    
    Educational Notes:
    - Each price level aggregates all orders at that price
    - The order book is composed of multiple price levels
    - Price levels are sorted by price (bids descending, asks ascending)
    - This structure enables efficient order matching
    """
    
    price: float
    total_volume: int = 0
    order_count: int = 0
    orders: List[Order] = field(default_factory=list)
    
    def add_order(self, order: Order) -> None:
        """Add an order to this price level"""
        if order.price != self.price:
            raise ValueError(f"Order price {order.price} doesn't match level price {self.price}")
        
        self.orders.append(order)
        self.total_volume += order.remaining_volume
        self.order_count += 1
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove an order from this price level"""
        for i, order in enumerate(self.orders):
            if order.order_id == order_id:
                removed_order = self.orders.pop(i)
                self.total_volume -= removed_order.remaining_volume
                self.order_count -= 1
                return removed_order
        return None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID"""
        for order in self.orders:
            if order.order_id == order_id:
                return order
        return None
    
    def is_empty(self) -> bool:
        """Check if this price level has no orders"""
        return self.order_count == 0 or self.total_volume == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert price level to dictionary"""
        return {
            'price': self.price,
            'total_volume': self.total_volume,
            'order_count': self.order_count,
            'orders': [order.order_id for order in self.orders]
        }
    
    def __str__(self) -> str:
        """String representation of the price level"""
        return f"PriceLevel({format_price(self.price)}): {format_volume(self.total_volume)} ({self.order_count} orders)"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MarketDataPoint:
    """
    Represents a single market data point for ML strategy processing
    
    Educational Notes:
    - Contains price, volume, and order book information at a specific time
    - Used by ML strategies for feature engineering and signal generation
    - Includes bid/ask data for spread and imbalance calculations
    """
    
    timestamp: pd.Timestamp
    price: float
    volume: int
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_volume: Optional[int] = None
    ask_volume: Optional[int] = None
    
    # Additional market data
    trade_direction: Optional[str] = None  # 'buy', 'sell', or None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate market data point"""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask"""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def order_imbalance(self) -> Optional[float]:
        """Calculate order book imbalance"""
        if self.bid_volume is not None and self.ask_volume is not None:
            total_volume = self.bid_volume + self.ask_volume
            if total_volume > 0:
                return (self.bid_volume - self.ask_volume) / total_volume
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'bid_volume': self.bid_volume,
            'ask_volume': self.ask_volume,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'order_imbalance': self.order_imbalance,
            'trade_direction': self.trade_direction,
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation"""
        mid_str = f" (mid: {self.mid_price:.4f})" if self.mid_price else ""
        return f"MarketData({self.timestamp}): {self.price:.4f} x {self.volume}{mid_str}"


# Type aliases for better code readability
OrderID = str
TradeID = str
UpdateID = str
Symbol = str
Price = float
Volume = int