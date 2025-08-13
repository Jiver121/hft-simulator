"""
Base Strategy Framework for HFT Simulator

This module provides the abstract base class and common functionality for all
trading strategies in the HFT simulator.

Educational Notes:
- All trading strategies inherit from BaseStrategy
- Strategies receive market data updates and generate trading decisions
- Risk management and position tracking are handled at the strategy level
- The framework supports both reactive and predictive trading approaches
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot
from config.settings import get_config


@dataclass
class StrategyResult:
    """
    Result of strategy decision-making process
    
    Contains orders to be submitted and any additional metadata
    about the strategy's decision process.
    """
    
    orders: List[Order] = field(default_factory=list)
    cancellations: List[str] = field(default_factory=list)  # Order IDs to cancel
    modifications: List[Tuple[str, Optional[float], Optional[int]]] = field(default_factory=list)  # (order_id, new_price, new_volume)
    
    # Strategy metadata
    decision_reason: str = ""
    confidence: float = 0.0
    expected_pnl: float = 0.0
    risk_score: float = 0.0
    
    # Performance tracking
    timestamp: Optional[pd.Timestamp] = None
    processing_time_us: int = 0
    
    def add_order(self, order: Order, reason: str = "") -> None:
        """Add an order to the result"""
        self.orders.append(order)
        if reason and not self.decision_reason:
            self.decision_reason = reason
    
    def add_cancellation(self, order_id: str) -> None:
        """Add an order cancellation"""
        self.cancellations.append(order_id)
    
    def add_modification(self, order_id: str, new_price: Optional[float] = None, 
                        new_volume: Optional[int] = None) -> None:
        """Add an order modification"""
        self.modifications.append((order_id, new_price, new_volume))
    
    def has_actions(self) -> bool:
        """Check if the result contains any actions"""
        return bool(self.orders or self.cancellations or self.modifications)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    This class provides the common framework that all trading strategies must implement.
    It handles position tracking, risk management, and provides utility methods for
    strategy development.
    
    Educational Notes:
    - Strategies are event-driven: they react to market data updates
    - Each strategy maintains its own state and position tracking
    - Risk management is integrated at the strategy level
    - Strategies can be backtested and run in live simulation
    
    Key Concepts:
    - Market Making: Provide liquidity by quoting both bid and ask
    - Liquidity Taking: Consume liquidity by hitting existing quotes
    - Mean Reversion: Bet that prices will return to historical average
    - Momentum: Follow price trends and breakouts
    """
    
    def __init__(self, 
                 strategy_name: str,
                 symbol: str,
                 max_position_size: int = 1000,
                 max_order_size: int = 100,
                 risk_limit: float = 10000.0):
        """
        Initialize base strategy
        
        Args:
            strategy_name: Name of the strategy
            symbol: Trading symbol
            max_position_size: Maximum absolute position size
            max_order_size: Maximum single order size
            risk_limit: Maximum risk exposure in dollars
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.max_position_size = max_position_size
        self.max_order_size = max_order_size
        self.risk_limit = risk_limit
        
        self.logger = get_logger(f"{__name__}.{strategy_name}")
        self.config = get_config()
        
        # Position and P&L tracking
        self.current_position = 0
        self.average_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_trades: List[Trade] = []
        
        # Market data history
        self.price_history: List[float] = []
        self.snapshot_history: List[BookSnapshot] = []
        self.max_history_length = 1000
        
        # Strategy state
        self.is_active = True
        self.last_update_time: Optional[pd.Timestamp] = None
        self.update_count = 0
        self.data_mode = "live"  # "live", "historical", "backtest"
        
        # Performance metrics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_volume_traded': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        }
        
        self.logger.info(f"Strategy {strategy_name} initialized for {symbol}")
    
    @abstractmethod
    def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> StrategyResult:
        """
        Process market data update and generate trading decisions
        
        This is the main method that strategies must implement. It receives
        market data updates and returns trading decisions.
        
        Args:
            snapshot: Current market snapshot
            timestamp: Update timestamp
            
        Returns:
            StrategyResult with orders and other actions
        """
        pass
    
    def on_trade_update(self, trade: Trade) -> StrategyResult:
        """
        Process trade update (when strategy's order is filled)
        
        Args:
            trade: Trade that occurred
            
        Returns:
            StrategyResult with any follow-up actions
        """
        # Update position and P&L
        self._update_position_from_trade(trade)
        
        # Remove filled order from active orders
        if trade.buy_order_id in self.active_orders:
            del self.active_orders[trade.buy_order_id]
        if trade.sell_order_id in self.active_orders:
            del self.active_orders[trade.sell_order_id]
        
        # Track trade
        self.completed_trades.append(trade)
        self.stats['total_volume_traded'] += trade.volume
        
        self.logger.debug(f"Trade processed: {trade.volume} @ {trade.price:.4f}")
        
        return StrategyResult()
    
    def on_order_update(self, order: Order, update_type: str) -> StrategyResult:
        """
        Process order update (filled, cancelled, rejected, etc.)
        
        Args:
            order: Updated order
            update_type: Type of update ('filled', 'cancelled', 'rejected', etc.)
            
        Returns:
            StrategyResult with any follow-up actions
        """
        if update_type == 'filled':
            self.stats['orders_filled'] += 1
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        
        elif update_type == 'cancelled':
            self.stats['orders_cancelled'] += 1
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        
        elif update_type == 'rejected':
            self.logger.warning(f"Order rejected: {order.order_id}")
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        
        return StrategyResult()
    
    def _update_position_from_trade(self, trade: Trade) -> None:
        """Update position and P&L from trade"""
        
        # Determine if this trade increases or decreases our position
        if trade.buy_order_id in self.active_orders:
            # We bought - increase position
            old_position = self.current_position
            trade_volume = trade.volume
            trade_price = trade.price
            
            # Update average price
            if self.current_position == 0:
                self.average_price = trade_price
            else:
                total_value = (old_position * self.average_price) + (trade_volume * trade_price)
                self.current_position = old_position + trade_volume
                self.average_price = total_value / self.current_position if self.current_position != 0 else 0
            
        elif trade.sell_order_id in self.active_orders:
            # We sold - decrease position
            trade_volume = trade.volume
            trade_price = trade.price
            
            # Calculate realized P&L
            if self.current_position > 0:
                realized_pnl = trade_volume * (trade_price - self.average_price)
                self.realized_pnl += realized_pnl
            
            self.current_position -= trade_volume
    
    def _update_market_history(self, snapshot: BookSnapshot) -> None:
        """Update market data history with error handling"""
        
        try:
            # Validate snapshot before processing
            if not self._validate_snapshot(snapshot):
                self.logger.warning(f"Invalid BookSnapshot received: {snapshot}")
                return
            
            # Add to snapshot history
            self.snapshot_history.append(snapshot)
            if len(self.snapshot_history) > self.max_history_length:
                self.snapshot_history.pop(0)
            
            # Add to price history
            if snapshot.mid_price is not None:
                self.price_history.append(snapshot.mid_price)
                if len(self.price_history) > self.max_history_length:
                    self.price_history.pop(0)
            else:
                # Try to calculate mid_price from best bid/ask if available
                if snapshot.best_bid is not None and snapshot.best_ask is not None:
                    mid_price = (snapshot.best_bid + snapshot.best_ask) / 2.0
                    self.price_history.append(mid_price)
                    if len(self.price_history) > self.max_history_length:
                        self.price_history.pop(0)
                    
        except Exception as e:
            self.logger.error(f"Error updating market history: {e}")
            # Continue processing without crashing
    
    def _validate_snapshot(self, snapshot: BookSnapshot) -> bool:
        """Validate BookSnapshot for basic data integrity"""
        try:
            # Check required attributes
            if not hasattr(snapshot, 'symbol'):
                self.logger.warning("BookSnapshot missing symbol")
                return False
            
            if not hasattr(snapshot, 'timestamp'):
                self.logger.warning("BookSnapshot missing timestamp")
                return False
            
            # Check that we have some market data
            has_bid_ask = (hasattr(snapshot, 'best_bid') and hasattr(snapshot, 'best_ask') and 
                          snapshot.best_bid is not None and snapshot.best_ask is not None)
            
            has_mid_price = hasattr(snapshot, 'mid_price') and snapshot.mid_price is not None
            
            has_price_levels = (hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks') and 
                              len(snapshot.bids) > 0 and len(snapshot.asks) > 0)
            
            if not (has_bid_ask or has_mid_price or has_price_levels):
                self.logger.warning("BookSnapshot missing market data (no bid/ask, mid_price, or levels)")
                return False
            
            # Check for reasonable price values
            if has_bid_ask:
                if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
                    self.logger.warning(f"Invalid prices: bid={snapshot.best_bid}, ask={snapshot.best_ask}")
                    return False
                
                if snapshot.best_bid >= snapshot.best_ask:
                    self.logger.warning(f"Crossed market: bid={snapshot.best_bid} >= ask={snapshot.best_ask}")
                    # Don't fail validation for crossed markets in tests/backtests
                    if self.data_mode == "live":
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating BookSnapshot: {e}")
            return False
    
    def set_data_mode(self, mode: str) -> None:
        """Set the data mode for the strategy
        
        Args:
            mode: One of 'live', 'historical', 'backtest'
        """
        valid_modes = ['live', 'historical', 'backtest']
        if mode in valid_modes:
            self.data_mode = mode
            self.logger.info(f"Strategy data mode set to: {mode}")
            
            # Adjust strategy behavior based on mode
            if mode == 'backtest':
                # More lenient validation for backtesting
                self.max_history_length = 10000  # Keep more history for analysis
            elif mode == 'live':
                # Strict validation for live trading
                self.max_history_length = 1000
        else:
            self.logger.warning(f"Invalid data mode: {mode}. Valid modes: {valid_modes}")
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L based on current price"""
        if self.current_position == 0:
            return 0.0
        
        return self.current_position * (current_price - self.average_price)
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics"""
        
        if not self.completed_trades:
            return
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.completed_trades 
                           if self._is_winning_trade(trade))
        self.stats['win_rate'] = winning_trades / len(self.completed_trades)
        
        # Calculate Sharpe ratio (simplified)
        if len(self.price_history) > 1:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            if np.std(returns) > 0:
                self.stats['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252 * 390)  # Annualized
    
    def _is_winning_trade(self, trade: Trade) -> bool:
        """Determine if a trade was profitable (simplified)"""
        # This is a simplified calculation - in reality, we'd need to track
        # the complete trade lifecycle and P&L attribution
        return True  # Placeholder
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current strategy state"""
        current_price = self.price_history[-1] if self.price_history else 0.0
        
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'is_active': self.is_active,
            'current_position': self.current_position,
            'average_price': self.average_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self._calculate_unrealized_pnl(current_price),
            'total_pnl': self.realized_pnl + self._calculate_unrealized_pnl(current_price),
            'active_orders': len(self.active_orders),
            'completed_trades': len(self.completed_trades),
            'last_update_time': self.last_update_time,
            'update_count': self.update_count,
            'stats': self.stats.copy(),
        }
    
    def check_risk_limits(self, proposed_order: Order, current_price: float) -> Tuple[bool, str]:
        """
        Check if proposed order violates risk limits
        
        Args:
            proposed_order: Order to check
            current_price: Current market price
            
        Returns:
            Tuple of (is_valid, reason)
        """
        
        # Check position limits
        new_position = self.current_position
        if proposed_order.is_buy():
            new_position += proposed_order.volume
        else:
            new_position -= proposed_order.volume
        
        if abs(new_position) > self.max_position_size:
            return False, f"Position limit exceeded: {new_position} > {self.max_position_size}"
        
        # Check order size limits
        if proposed_order.volume > self.max_order_size:
            return False, f"Order size too large: {proposed_order.volume} > {self.max_order_size}"
        
        # Check risk limit
        potential_loss = abs(new_position) * current_price * 0.1  # Assume 10% adverse move
        if potential_loss > self.risk_limit:
            return False, f"Risk limit exceeded: potential loss ${potential_loss:.2f} > ${self.risk_limit:.2f}"
        
        return True, ""
    
    def create_order(self, side: OrderSide, volume: int, price: Optional[float] = None,
                    order_type: OrderType = OrderType.LIMIT, reason: str = "") -> Order:
        """
        Create an order with strategy metadata
        
        Args:
            side: Order side (buy/sell)
            volume: Order volume
            price: Order price (None for market orders)
            order_type: Order type
            reason: Reason for the order
            
        Returns:
            Created order
        """
        # If no price provided, default to MARKET order for test compatibility
        if price is None:
            order_type = OrderType.MARKET

        order = Order(
            order_id=f"{self.strategy_name}_{self.update_count}_{len(self.active_orders)}",
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            price=price,
            volume=volume,
            timestamp=pd.Timestamp.now(),
            source=self.strategy_name,
            metadata={
                'strategy': self.strategy_name,
                'reason': reason,
                'position_before': self.current_position,
            }
        )
        
        return order
    
    def submit_order(self, order: Order) -> None:
        """Submit an order and track it"""
        self.active_orders[order.order_id] = order
        self.stats['orders_submitted'] += 1
        
        self.logger.debug(f"Order submitted: {order}")
    
    def cancel_all_orders(self) -> List[str]:
        """Cancel all active orders"""
        order_ids = list(self.active_orders.keys())
        self.active_orders.clear()
        
        self.logger.info(f"Cancelled {len(order_ids)} active orders")
        return order_ids
    
    def set_active(self, active: bool) -> None:
        """Set strategy active/inactive state"""
        self.is_active = active
        
        if not active:
            # Cancel all orders when deactivating
            self.cancel_all_orders()
        
        self.logger.info(f"Strategy {'activated' if active else 'deactivated'}")
    
    def reset(self) -> None:
        """Reset strategy state"""
        self.current_position = 0
        self.average_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        
        self.active_orders.clear()
        self.completed_trades.clear()
        self.price_history.clear()
        self.snapshot_history.clear()
        
        self.update_count = 0
        self.last_update_time = None
        
        # Reset stats
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'total_volume_traded': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
        }
        
        self.logger.info("Strategy state reset")
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return (f"{self.strategy_name}({self.symbol}): "
                f"Pos={self.current_position}, "
                f"PnL=${self.total_pnl:.2f}, "
                f"Orders={len(self.active_orders)}")
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for strategy development
def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """Calculate rolling volatility from price series"""
    if len(prices) < window:
        return 0.0
    
    recent_prices = prices[-window:]
    returns = [recent_prices[i] / recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
    
    return np.std(returns) if returns else 0.0


def calculate_momentum(prices: List[float], window: int = 10) -> float:
    """Calculate price momentum over specified window"""
    if len(prices) < window + 1:
        return 0.0
    
    return (prices[-1] / prices[-window-1]) - 1


def calculate_mean_reversion_signal(prices: List[float], window: int = 20) -> float:
    """Calculate mean reversion signal"""
    if len(prices) < window:
        return 0.0
    
    recent_prices = prices[-window:]
    mean_price = np.mean(recent_prices)
    current_price = prices[-1]
    
    # Negative signal means price is above mean (sell signal)
    # Positive signal means price is below mean (buy signal)
    return (mean_price - current_price) / mean_price


def calculate_order_book_imbalance(snapshot: BookSnapshot) -> float:
    """Calculate order book imbalance signal"""
    if not snapshot.best_bid_volume or not snapshot.best_ask_volume:
        return 0.0
    
    total_volume = snapshot.best_bid_volume + snapshot.best_ask_volume
    imbalance = (snapshot.best_bid_volume - snapshot.best_ask_volume) / total_volume
    
    return imbalance