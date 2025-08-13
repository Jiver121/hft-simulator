"""
Market Data Structures for HFT Simulator

This module defines data structures for market data, order book snapshots,
and related market information used throughout the HFT simulator.

Educational Notes:
- Market data represents the current state of the market
- Book snapshots capture the order book at specific points in time
- These structures are essential for strategy development and analysis
- Real-time market data feeds would populate these structures in live trading
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from src.utils.constants import OrderSide, PRICE_PRECISION
from src.utils.helpers import format_price, format_volume, safe_divide
from .order_types import PriceLevel, Order


@dataclass
class BookSnapshot:
    """
    Represents a snapshot of the order book at a specific point in time
    
    Educational Notes:
    - Book snapshots show the complete state of buy and sell orders
    - They include multiple price levels on both bid and ask sides
    - Snapshots are used for strategy decision making and analysis
    - The depth of the book affects market impact and execution quality
    """
    
    symbol: str
    timestamp: pd.Timestamp
    
    # Bid side (buy orders) - sorted by price descending
    bids: List[PriceLevel] = field(default_factory=list)
    
    # Ask side (sell orders) - sorted by price ascending  
    asks: List[PriceLevel] = field(default_factory=list)
    
    # Market statistics
    last_trade_price: Optional[float] = None
    last_trade_volume: Optional[int] = None
    last_trade_timestamp: Optional[pd.Timestamp] = None
    
    # Sequence number for ordering snapshots
    sequence_number: int = 0
    
    @classmethod
    def create_from_best_quotes(
        cls, 
        symbol: str, 
        timestamp: pd.Timestamp, 
        best_bid: float,
        best_bid_volume: int,
        best_ask: float,
        best_ask_volume: int,
        **kwargs
    ) -> 'BookSnapshot':
        """Create BookSnapshot from best bid/ask quotes.
        
        Args:
            symbol: Trading symbol
            timestamp: Snapshot timestamp
            best_bid: Best bid price
            best_bid_volume: Volume at best bid
            best_ask: Best ask price  
            best_ask_volume: Volume at best ask
            **kwargs: Additional optional parameters
        
        Returns:
            BookSnapshot instance
        """
        bid_level = PriceLevel(price=best_bid, total_volume=best_bid_volume)
        ask_level = PriceLevel(price=best_ask, total_volume=best_ask_volume)
        
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            bids=[bid_level],
            asks=[ask_level],
            **kwargs
        )
    
    def __post_init__(self):
        """Validate, normalize, and sort price levels after creation.

        Accepts either PriceLevel instances or (price, volume) tuples/lists
        and normalizes them to PriceLevel for internal consistency.
        """
        # Normalize bids to PriceLevel
        normalized_bids: List[PriceLevel] = []
        for level in self.bids:
            if isinstance(level, PriceLevel):
                normalized_bids.append(level)
            else:
                # Expect tuple/list like (price, volume)
                price, volume = level  # type: ignore[misc]
                normalized_bids.append(PriceLevel(price=float(price), total_volume=int(volume)))
        self.bids = normalized_bids

        # Normalize asks to PriceLevel
        normalized_asks: List[PriceLevel] = []
        for level in self.asks:
            if isinstance(level, PriceLevel):
                normalized_asks.append(level)
            else:
                price, volume = level  # type: ignore[misc]
                normalized_asks.append(PriceLevel(price=float(price), total_volume=int(volume)))
        self.asks = normalized_asks

        # Sort bids by price descending (highest first)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        # Sort asks by price ascending (lowest first)
        self.asks.sort(key=lambda x: x.price, reverse=False)
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price"""
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price"""
        return self.asks[0].price if self.asks else None
    
    @property
    def best_bid_volume(self) -> Optional[int]:
        """Get the volume at the best bid"""
        return self.bids[0].total_volume if self.bids else None
    
    @property
    def best_ask_volume(self) -> Optional[int]:
        """Get the volume at the best ask"""
        return self.asks[0].total_volume if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the mid-price (average of best bid and ask)"""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate the bid-ask spread"""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate the spread in basis points"""
        if self.spread is not None and self.mid_price is not None:
            return (self.spread / self.mid_price) * 10000
        return None
    
    @property
    def total_bid_volume(self) -> int:
        """Get total volume on the bid side"""
        return sum(level.total_volume for level in self.bids)
    
    @property
    def total_ask_volume(self) -> int:
        """Get total volume on the ask side"""
        return sum(level.total_volume for level in self.asks)
    
    @property
    def order_book_imbalance(self) -> Optional[float]:
        """
        Calculate order book imbalance
        
        Returns:
            Imbalance ratio: (bid_volume - ask_volume) / (bid_volume + ask_volume)
            Positive values indicate more buying pressure
            Negative values indicate more selling pressure
        """
        bid_vol = self.total_bid_volume
        ask_vol = self.total_ask_volume
        
        if bid_vol + ask_vol == 0:
            return None
        
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    def get_depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]:
        """
        Get market depth for specified side
        
        Args:
            side: Which side of the book (bid or ask)
            levels: Number of price levels to return
            
        Returns:
            List of (price, volume) tuples
        """
        if side in [OrderSide.BID, OrderSide.BUY]:
            price_levels = self.bids[:levels]
        else:
            price_levels = self.asks[:levels]
        
        return [(level.price, level.total_volume) for level in price_levels]
    
    def get_volume_at_price(self, price: float, side: OrderSide) -> int:
        """
        Get total volume available at a specific price
        
        Args:
            price: Price level to check
            side: Which side of the book
            
        Returns:
            Total volume at that price level
        """
        price_levels = self.bids if side in [OrderSide.BID, OrderSide.BUY] else self.asks
        
        for level in price_levels:
            if abs(level.price - price) < 1e-8:  # Handle floating point precision
                return level.total_volume
        
        return 0
    
    def get_market_impact(self, side: OrderSide, volume: int) -> Tuple[float, float]:
        """
        Calculate market impact for a given order size
        
        Args:
            side: Side of the order (buy orders impact asks, sell orders impact bids)
            volume: Order volume
            
        Returns:
            Tuple of (average_price, total_cost)
        """
        # Buy orders consume ask liquidity, sell orders consume bid liquidity
        price_levels = self.asks if side in [OrderSide.BID, OrderSide.BUY] else self.bids
        
        if not price_levels:
            return 0.0, 0.0
        
        remaining_volume = volume
        total_cost = 0.0
        
        for level in price_levels:
            if remaining_volume <= 0:
                break
            
            volume_at_level = min(remaining_volume, level.total_volume)
            total_cost += volume_at_level * level.price
            remaining_volume -= volume_at_level
        
        if remaining_volume > 0:
            # Not enough liquidity - use last available price for remaining volume
            last_price = price_levels[-1].price
            total_cost += remaining_volume * last_price
        
        average_price = total_cost / volume if volume > 0 else 0.0
        
        return average_price, total_cost
    
    def is_crossed(self) -> bool:
        """
        Check if the market is crossed (bid >= ask)
        
        Returns:
            True if best bid >= best ask (abnormal market condition)
        """
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_bid >= self.best_ask
        return False
    
    def is_locked(self) -> bool:
        """
        Check if the market is locked (bid == ask)
        
        Returns:
            True if best bid == best ask
        """
        if self.best_bid is not None and self.best_ask is not None:
            return abs(self.best_bid - self.best_ask) < 1e-8
        return False
    
    def to_dict(self, depth: int = 5) -> Dict[str, Any]:
        """
        Convert snapshot to dictionary representation
        
        Args:
            depth: Number of price levels to include
            
        Returns:
            Dictionary representation of the book snapshot
        """
        result = {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'sequence_number': self.sequence_number,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'best_bid_volume': self.best_bid_volume,
            'best_ask_volume': self.best_ask_volume,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'total_bid_volume': self.total_bid_volume,
            'total_ask_volume': self.total_ask_volume,
            'order_book_imbalance': self.order_book_imbalance,
            'last_trade_price': self.last_trade_price,
            'last_trade_volume': self.last_trade_volume,
            'last_trade_timestamp': self.last_trade_timestamp,
            'is_crossed': self.is_crossed(),
            'is_locked': self.is_locked(),
        }
        
        # Add bid levels
        for i, level in enumerate(self.bids[:depth]):
            result[f'bid_price_{i+1}'] = level.price
            result[f'bid_volume_{i+1}'] = level.total_volume
            result[f'bid_orders_{i+1}'] = level.order_count
        
        # Add ask levels
        for i, level in enumerate(self.asks[:depth]):
            result[f'ask_price_{i+1}'] = level.price
            result[f'ask_volume_{i+1}'] = level.total_volume
            result[f'ask_orders_{i+1}'] = level.order_count
        
        return result
    
    def __str__(self) -> str:
        """String representation of the book snapshot"""
        bid_str = f"{format_price(self.best_bid)}x{format_volume(self.best_bid_volume)}" if self.best_bid else "---"
        ask_str = f"{format_price(self.best_ask)}x{format_volume(self.best_ask_volume)}" if self.best_ask else "---"
        spread_str = f"{self.spread:.4f}" if self.spread else "---"
        
        return f"BookSnapshot({self.symbol}): {bid_str} | {ask_str} (spread: {spread_str})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class MarketData:
    """
    Container for comprehensive market data including current state and history
    
    Educational Notes:
    - MarketData aggregates all market information for a symbol
    - It includes current book state, recent trades, and statistics
    - This structure is used by strategies to make trading decisions
    - Historical data enables backtesting and performance analysis
    """
    
    symbol: str
    current_snapshot: Optional[BookSnapshot] = None
    
    # Historical data
    snapshots_history: List[BookSnapshot] = field(default_factory=list)
    trades_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Market statistics
    daily_high: Optional[float] = None
    daily_low: Optional[float] = None
    daily_open: Optional[float] = None
    daily_close: Optional[float] = None
    daily_volume: int = 0
    daily_trade_count: int = 0
    
    # Real-time statistics
    last_update_timestamp: Optional[pd.Timestamp] = None
    update_count: int = 0
    
    def update_snapshot(self, snapshot: BookSnapshot) -> None:
        """Update the current market snapshot"""
        self.current_snapshot = snapshot
        self.last_update_timestamp = snapshot.timestamp
        self.update_count += 1
        
        # Add to history (keep last N snapshots)
        self.snapshots_history.append(snapshot)
        if len(self.snapshots_history) > 1000:  # Keep last 1000 snapshots
            self.snapshots_history.pop(0)
        
        # Update daily statistics
        if snapshot.last_trade_price is not None:
            self._update_daily_stats(snapshot.last_trade_price, snapshot.last_trade_volume or 0)
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add a trade to the history"""
        self.trades_history.append(trade_data)
        self.daily_trade_count += 1
        
        # Keep last 1000 trades
        if len(self.trades_history) > 1000:
            self.trades_history.pop(0)
        
        # Update daily statistics
        if 'price' in trade_data and 'volume' in trade_data:
            self._update_daily_stats(trade_data['price'], trade_data['volume'])
    
    def _update_daily_stats(self, price: float, volume: int) -> None:
        """Update daily trading statistics"""
        if self.daily_high is None or price > self.daily_high:
            self.daily_high = price
        
        if self.daily_low is None or price < self.daily_low:
            self.daily_low = price
        
        if self.daily_open is None:
            self.daily_open = price
        
        self.daily_close = price
        self.daily_volume += volume
    
    @property
    def current_price(self) -> Optional[float]:
        """Get the current market price (mid-price or last trade)"""
        if self.current_snapshot:
            if self.current_snapshot.mid_price is not None:
                return self.current_snapshot.mid_price
            elif self.current_snapshot.last_trade_price is not None:
                return self.current_snapshot.last_trade_price
        return None
    
    @property
    def daily_return(self) -> Optional[float]:
        """Calculate daily return as percentage"""
        if self.daily_open is not None and self.daily_close is not None and self.daily_open != 0:
            return ((self.daily_close - self.daily_open) / self.daily_open) * 100
        return None
    
    @property
    def daily_range(self) -> Optional[float]:
        """Calculate daily trading range"""
        if self.daily_high is not None and self.daily_low is not None:
            return self.daily_high - self.daily_low
        return None
    
    def get_recent_snapshots(self, count: int = 10) -> List[BookSnapshot]:
        """Get the most recent book snapshots"""
        return self.snapshots_history[-count:] if self.snapshots_history else []
    
    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent trades"""
        return self.trades_history[-count:] if self.trades_history else []
    
    def get_price_history(self, count: int = 100) -> List[float]:
        """Get recent price history from snapshots"""
        prices = []
        for snapshot in self.snapshots_history[-count:]:
            if snapshot.mid_price is not None:
                prices.append(snapshot.mid_price)
            elif snapshot.last_trade_price is not None:
                prices.append(snapshot.last_trade_price)
        return prices
    
    def get_spread_history(self, count: int = 100) -> List[float]:
        """Get recent spread history"""
        spreads = []
        for snapshot in self.snapshots_history[-count:]:
            if snapshot.spread is not None:
                spreads.append(snapshot.spread)
        return spreads
    
    def get_volume_profile(self, side: OrderSide, levels: int = 10) -> Dict[float, int]:
        """
        Get volume profile for a specific side
        
        Args:
            side: Which side of the book
            levels: Number of price levels
            
        Returns:
            Dictionary mapping prices to volumes
        """
        if not self.current_snapshot:
            return {}
        
        depth = self.current_snapshot.get_depth(side, levels)
        return {price: volume for price, volume in depth}
    
    def calculate_vwap(self, lookback_trades: int = 100) -> Optional[float]:
        """
        Calculate Volume Weighted Average Price
        
        Args:
            lookback_trades: Number of recent trades to include
            
        Returns:
            VWAP or None if insufficient data
        """
        recent_trades = self.get_recent_trades(lookback_trades)
        
        if not recent_trades:
            return None
        
        total_value = sum(trade['price'] * trade['volume'] for trade in recent_trades)
        total_volume = sum(trade['volume'] for trade in recent_trades)
        
        return safe_divide(total_value, total_volume)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert market data to dictionary representation"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'daily_high': self.daily_high,
            'daily_low': self.daily_low,
            'daily_open': self.daily_open,
            'daily_close': self.daily_close,
            'daily_volume': self.daily_volume,
            'daily_trade_count': self.daily_trade_count,
            'daily_return': self.daily_return,
            'daily_range': self.daily_range,
            'last_update_timestamp': self.last_update_timestamp,
            'update_count': self.update_count,
            'current_snapshot': self.current_snapshot.to_dict() if self.current_snapshot else None,
            'vwap': self.calculate_vwap(),
        }
    
    def __str__(self) -> str:
        """String representation of market data"""
        price_str = format_price(self.current_price) if self.current_price else "---"
        volume_str = format_volume(self.daily_volume)
        
        return (f"MarketData({self.symbol}): {price_str} "
                f"(Vol: {volume_str}, Updates: {self.update_count})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for market data analysis
def calculate_microprice(snapshot: BookSnapshot, alpha: float = 0.5) -> Optional[float]:
    """
    Calculate microprice - a more accurate estimate of fair value
    
    Args:
        snapshot: Order book snapshot
        alpha: Weight parameter (0.5 = equal weight to bid/ask volumes)
        
    Returns:
        Microprice or None if insufficient data
        
    Educational Notes:
    - Microprice considers order book imbalance in price calculation
    - It's often more accurate than simple mid-price for fair value
    - Formula: microprice = (alpha * ask * bid_vol + (1-alpha) * bid * ask_vol) / (bid_vol + ask_vol)
    """
    if not snapshot.best_bid or not snapshot.best_ask:
        return None
    
    if not snapshot.best_bid_volume or not snapshot.best_ask_volume:
        return snapshot.mid_price
    
    bid_vol = snapshot.best_bid_volume
    ask_vol = snapshot.best_ask_volume
    total_vol = bid_vol + ask_vol
    
    if total_vol == 0:
        return snapshot.mid_price
    
    microprice = (alpha * snapshot.best_ask * bid_vol + 
                  (1 - alpha) * snapshot.best_bid * ask_vol) / total_vol
    
    return microprice


def calculate_effective_spread(trade_price: float, snapshot: BookSnapshot) -> Optional[float]:
    """
    Calculate effective spread - the cost of immediate execution
    
    Args:
        trade_price: Actual trade price
        snapshot: Order book snapshot at time of trade
        
    Returns:
        Effective spread or None if insufficient data
    """
    if not snapshot.mid_price:
        return None
    
    return 2 * abs(trade_price - snapshot.mid_price)