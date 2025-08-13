"""
Matching Engine for HFT Simulator

This module implements the core matching engine that processes orders and
generates trades according to market rules and priority systems.

Educational Notes:
- The matching engine is the heart of any electronic trading system
- It enforces price-time priority and other matching rules
- Real exchanges use highly optimized matching engines for microsecond latency
- Understanding matching logic is crucial for HFT strategy development
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Iterator
from collections import deque
from datetime import datetime
import uuid

from src.utils.logger import get_logger, log_performance
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.engine.order_types import Order, Trade, OrderUpdate
from src.engine.order_book import OrderBook
from src.engine.market_data import BookSnapshot
from .fill_models import FillModel, RealisticFillModel, FillResult


class MatchingEngine:
    """
    High-performance matching engine for order processing
    
    The matching engine is responsible for:
    - Processing incoming orders
    - Matching buy and sell orders
    - Generating trades
    - Maintaining order book integrity
    - Enforcing trading rules and priorities
    
    Key Features:
    - Price-time priority matching
    - Support for multiple order types
    - Realistic fill modeling
    - Trade reporting and audit trail
    - Performance monitoring
    
    Educational Notes:
    - Price priority: better prices get matched first
    - Time priority: earlier orders at same price get matched first
    - Pro-rata matching: some venues split large orders proportionally
    - Hidden orders: some orders don't show full size to market
    """
    
    def __init__(self, 
                 symbol: str,
                 fill_model: Optional[FillModel] = None,
                 tick_size: float = 0.01,
                 min_order_size: int = 1,
                 max_order_size: int = 1000000):
        """
        Initialize the matching engine
        
        Args:
            symbol: Trading symbol
            fill_model: Model for simulating realistic fills
            tick_size: Minimum price increment
            min_order_size: Minimum order size
            max_order_size: Maximum order size
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        
        self.logger = get_logger(f"{__name__}.{symbol}")
        
        # Order book for this symbol
        self.order_book = OrderBook(symbol, tick_size)
        
        # Fill model for realistic execution simulation
        self.fill_model = fill_model or RealisticFillModel()
        
        # Trade and order tracking
        self.trades: List[Trade] = []
        self.order_updates: List[OrderUpdate] = []
        
        # Performance statistics
        self.stats = {
            'orders_processed': 0,
            'trades_generated': 0,
            'total_volume_matched': 0,
            'orders_rejected': 0,
            'processing_time_us': 0,
        }
        
        # Market state
        self.is_open = True
        self.last_trade_price: Optional[float] = None
        self.session_high: Optional[float] = None
        self.session_low: Optional[float] = None
        
        self.logger.info(f"MatchingEngine initialized for {symbol}")
    
    def submit_order(self, order: Order) -> Tuple[List[Trade], OrderUpdate]:
        """
        Submit an order to the matching engine (delegates to OrderBook)
        
        Args:
            order: Order to submit
            
        Returns:
            Tuple of (trades_generated, order_update)
        """
        return self.process_order(order)
    
    @log_performance
    def process_order(self, order: Order) -> Tuple[List[Trade], OrderUpdate]:
        """
        Process an incoming order through the matching engine
        
        Args:
            order: Order to process
            
        Returns:
            Tuple of (trades_generated, order_update)
            
        Educational Notes:
        - Orders are validated before processing
        - Market orders are matched immediately
        - Limit orders are added to book if not immediately matched
        - All changes generate order updates for audit trail
        """
        start_time = pd.Timestamp.now()
        
        # Log order entry into matching engine
        self.logger.debug(f"[MATCHING_ENGINE_ENTRY] Order {order.order_id} entering MatchingEngine: "
                         f"Symbol={order.symbol}, Side={order.side.value}, Type={order.order_type.value}, "
                         f"Price={order.price}, Volume={order.volume}")
        
        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result['valid']:
            self.logger.debug(f"[ORDER_VALIDATION] Order {order.order_id} REJECTED: {validation_result['reason']}")
            order.status = OrderStatus.REJECTED
            update = OrderUpdate(
                update_id=str(uuid.uuid4()),
                symbol=self.symbol,
                timestamp=start_time,
                update_type='reject',
                order_id=order.order_id,
                metadata={'rejection_reason': validation_result['reason']}
            )
            
            self.stats['orders_rejected'] += 1
            return [], update
        
        self.logger.debug(f"[ORDER_VALIDATION] Order {order.order_id} passed validation - forwarding to OrderBook")
        
        # Get current market snapshot for pre-execution analysis
        snapshot = self.order_book.get_snapshot()
        pre_execution_best_bid = snapshot.best_bid if snapshot else None
        pre_execution_best_ask = snapshot.best_ask if snapshot else None
        
        self.logger.debug(f"[PRE_EXECUTION_STATE] Market state before execution: "
                         f"BestBid={pre_execution_best_bid}, BestAsk={pre_execution_best_ask}")
        
        # Use fill model for post-execution analysis only (for realistic simulation metrics)
        # Don't let fill model pre-filter orders - let OrderBook handle matching
        
        # Delegate to OrderBook for actual matching and execution
        try:
            self.logger.debug(f"[ORDERBOOK_DELEGATION] Delegating order {order.order_id} to OrderBook for matching")
            trades = self.order_book.add_order(order)
            self.logger.debug(f"[ORDERBOOK_RESPONSE] OrderBook returned {len(trades)} trades for order {order.order_id}")
        except Exception as e:
            self.logger.error(f"[ORDERBOOK_ERROR] Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            update = OrderUpdate(
                update_id=str(uuid.uuid4()),
                symbol=self.symbol,
                timestamp=start_time,
                update_type='reject',
                order_id=order.order_id,
                metadata={'rejection_reason': f'OrderBook error: {str(e)}'}
            )
            
            self.stats['orders_rejected'] += 1
            return [], update
        
        # Log trade propagation and post-execution analysis
        if trades:
            self.logger.debug(f"[TRADE_PROPAGATION] Order {order.order_id} generated {len(trades)} trades:")
            for i, trade in enumerate(trades):
                self.logger.debug(f"[TRADE_DETAILS] Trade {i+1}: ID={trade.trade_id}, "
                                f"Price={trade.price}, Volume={trade.volume}, "
                                f"BuyOrder={trade.buy_order_id}, SellOrder={trade.sell_order_id}, "
                                f"Aggressor={trade.aggressor_side.value}")
            
            # Update session statistics from actual trades
            last_trade = trades[-1]
            old_high = self.session_high
            old_low = self.session_low
            self._update_session_stats(last_trade.price)
            
            # Log market impact
            self.logger.debug(f"[MARKET_IMPACT] Last trade price: {last_trade.price}, "
                            f"SessionHigh: {old_high} -> {self.session_high}, "
                            f"SessionLow: {old_low} -> {self.session_low}")
        else:
            self.logger.debug(f"[NO_TRADES] Order {order.order_id} did not generate any trades")
        
        # Get post-execution market state
        post_snapshot = self.order_book.get_snapshot()
        post_execution_best_bid = post_snapshot.best_bid if post_snapshot else None
        post_execution_best_ask = post_snapshot.best_ask if post_snapshot else None
        
        self.logger.debug(f"[POST_EXECUTION_STATE] Market state after execution: "
                         f"BestBid={pre_execution_best_bid} -> {post_execution_best_bid}, "
                         f"BestAsk={pre_execution_best_ask} -> {post_execution_best_ask}")
        
        # Create order update based on actual execution results
        update = self._create_order_update(order, trades)
        self.logger.debug(f"[ORDER_UPDATE] Created update: Type={update.update_type}, "
                         f"OrderID={update.order_id}, Status={order.status}")
        
        # Update statistics
        self.stats['orders_processed'] += 1
        self.stats['trades_generated'] += len(trades)
        total_volume = sum(trade.volume for trade in trades)
        self.stats['total_volume_matched'] += total_volume
        
        processing_time = (pd.Timestamp.now() - start_time).total_seconds() * 1_000_000
        self.stats['processing_time_us'] += processing_time
        
        self.logger.debug(f"[PROCESSING_STATS] Order {order.order_id} processed in {processing_time:.2f}μs, "
                         f"TotalVolume={total_volume}, CumulativeStats: "
                         f"Orders={self.stats['orders_processed']}, Trades={self.stats['trades_generated']}")
        
        # Store trades and updates for propagation to downstream systems
        self.trades.extend(trades)
        self.order_updates.append(update)
        
        self.logger.debug(f"[TRADE_STORAGE] Stored {len(trades)} trades and 1 update. "
                         f"Total stored: Trades={len(self.trades)}, Updates={len(self.order_updates)}")
        
        return trades, update
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was successfully cancelled
        """
        success = self.order_book.cancel_order(order_id)
        
        if success:
            # Create cancellation update
            order = self.order_book.get_order(order_id)
            if order:
                update = OrderUpdate.cancel_order(order)
                self.order_updates.append(update)
        
        return success
    
    def modify_order(self, order_id: str, 
                    new_price: Optional[float] = None,
                    new_volume: Optional[int] = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: ID of order to modify
            new_price: New price (None to keep current)
            new_volume: New volume (None to keep current)
            
        Returns:
            True if order was successfully modified
        """
        return self.order_book.modify_order(order_id, new_price, new_volume)
    
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """
        Validate an order before processing
        
        Args:
            order: Order to validate
            
        Returns:
            Dictionary with validation result
        """
        # Check if market is open
        if not self.is_open:
            return {'valid': False, 'reason': 'Market is closed'}
        
        # Check symbol
        if order.symbol != self.symbol:
            return {'valid': False, 'reason': f'Invalid symbol: {order.symbol}'}
        
        # Check order size
        if order.volume < self.min_order_size:
            return {'valid': False, 'reason': f'Order size below minimum: {order.volume}'}
        
        if order.volume > self.max_order_size:
            return {'valid': False, 'reason': f'Order size above maximum: {order.volume}'}
        
        # Check price for limit orders
        if order.is_limit_order():
            if order.price is None or order.price <= 0:
                return {'valid': False, 'reason': 'Invalid price for limit order'}
            
            # Check tick size compliance
            price_ticks = order.price / self.tick_size
            if abs(price_ticks - round(price_ticks)) > 1e-8:
                return {'valid': False, 'reason': f'Price not aligned to tick size: {self.tick_size}'}
        
        # Check for duplicate order ID
        if order.order_id in self.order_book.orders:
            return {'valid': False, 'reason': f'Duplicate order ID: {order.order_id}'}
        
        return {'valid': True, 'reason': None}
    
    def _create_order_update(self, order: Order, trades: List[Trade]) -> OrderUpdate:
        """Create an order update based on processing results"""
        
        if order.status == OrderStatus.REJECTED:
            # Order was rejected
            return OrderUpdate(
                update_id=str(uuid.uuid4()),
                symbol=self.symbol,
                timestamp=pd.Timestamp.now(),
                update_type='reject',
                order_id=order.order_id,
                side=order.side,
                price=order.price,
                volume=order.volume,
                metadata={
                    'rejection_reason': 'Order processing failed'
                }
            )
        
        elif trades:
            # Order resulted in trades
            total_fill_volume = sum(trade.volume for trade in trades)
            avg_fill_price = sum(trade.price * trade.volume for trade in trades) / total_fill_volume if total_fill_volume > 0 else 0
            
            return OrderUpdate(
                update_id=str(uuid.uuid4()),
                symbol=self.symbol,
                timestamp=pd.Timestamp.now(),
                update_type='trade',
                order_id=order.order_id,
                trade_id=trades[0].trade_id,
                trade_price=avg_fill_price,
                trade_volume=total_fill_volume,
                metadata={
                    'num_trades': len(trades),
                    'order_status': order.status.value if hasattr(order.status, 'value') else str(order.status)
                }
            )
        
        else:
            # Order was added to book
            return OrderUpdate.new_order(order)
    
    def _update_session_stats(self, trade_price: float) -> None:
        """Update session trading statistics"""
        self.last_trade_price = trade_price
        
        if self.session_high is None or trade_price > self.session_high:
            self.session_high = trade_price
        
        if self.session_low is None or trade_price < self.session_low:
            self.session_low = trade_price
    
    def get_market_snapshot(self) -> BookSnapshot:
        """Get current market snapshot"""
        return self.order_book.get_snapshot()
    
    def get_order_book(self) -> OrderBook:
        """Get the order book instance"""
        return self.order_book
    
    def get_recent_trades(self, count: int = 10) -> List[Trade]:
        """Get recent trades"""
        return self.trades[-count:] if self.trades else []
    
    def get_trade_history(self) -> List[Trade]:
        """Get complete trade history"""
        return self.trades.copy()
    
    def get_order_updates(self) -> List[OrderUpdate]:
        """Get all order updates"""
        return self.order_updates.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matching engine statistics"""
        snapshot = self.get_market_snapshot()
        
        avg_processing_time = (self.stats['processing_time_us'] / 
                             max(1, self.stats['orders_processed']))
        
        return {
            'symbol': self.symbol,
            'is_open': self.is_open,
            'orders_processed': self.stats['orders_processed'],
            'trades_generated': self.stats['trades_generated'],
            'total_volume_matched': self.stats['total_volume_matched'],
            'orders_rejected': self.stats['orders_rejected'],
            'rejection_rate': self.stats['orders_rejected'] / max(1, self.stats['orders_processed']),
            'avg_processing_time_us': avg_processing_time,
            'last_trade_price': self.last_trade_price,
            'session_high': self.session_high,
            'session_low': self.session_low,
            'current_snapshot': snapshot.to_dict() if snapshot else None,
            'order_book_stats': self.order_book.get_statistics(),
        }
    
    def reset_session(self) -> None:
        """Reset session statistics and state"""
        self.trades.clear()
        self.order_updates.clear()
        self.order_book.clear()
        
        self.stats = {
            'orders_processed': 0,
            'trades_generated': 0,
            'total_volume_matched': 0,
            'orders_rejected': 0,
            'processing_time_us': 0,
        }
        
        self.last_trade_price = None
        self.session_high = None
        self.session_low = None
        
        self.logger.info(f"Session reset for {self.symbol}")
    
    def set_market_state(self, is_open: bool) -> None:
        """Set market open/closed state"""
        self.is_open = is_open
        self.logger.info(f"Market state changed to {'OPEN' if is_open else 'CLOSED'}")
    
    def process_market_data_update(self, update: OrderUpdate) -> None:
        """
        Process market data update (for replay scenarios)
        
        Args:
            update: Market data update to process
            
        Educational Notes:
        - This method allows replaying historical market data
        - Useful for backtesting strategies against real market conditions
        - Updates are processed in chronological order
        """
        if update.update_type == 'new' and update.order_id:
            # Create order from update
            order = Order(
                order_id=update.order_id,
                symbol=update.symbol,
                side=update.side,
                order_type=OrderType.LIMIT,  # Assume limit orders from market data
                price=update.price,
                volume=update.volume,
                timestamp=update.timestamp,
                source='market_data'
            )
            
            # Process through matching engine
            self.process_order(order)
        
        elif update.update_type == 'cancel' and update.order_id:
            self.cancel_order(update.order_id)
        
        elif update.update_type == 'trade':
            # External trade - update market statistics
            if update.trade_price:
                self._update_session_stats(update.trade_price)
    
    def __str__(self) -> str:
        """String representation of the matching engine"""
        return (f"MatchingEngine({self.symbol}): "
                f"{self.stats['orders_processed']} orders, "
                f"{self.stats['trades_generated']} trades, "
                f"{'OPEN' if self.is_open else 'CLOSED'}")
    
    def __repr__(self) -> str:
        return self.__str__()


class MultiSymbolMatchingEngine:
    """
    Matching engine that handles multiple symbols
    
    This class manages separate matching engines for different trading symbols,
    providing a unified interface for multi-asset trading.
    """
    
    def __init__(self, symbols: List[str], **engine_kwargs):
        """
        Initialize multi-symbol matching engine
        
        Args:
            symbols: List of trading symbols
            **engine_kwargs: Arguments passed to individual matching engines
        """
        self.symbols = symbols
        self.engines: Dict[str, MatchingEngine] = {}
        
        # Create matching engine for each symbol
        for symbol in symbols:
            self.engines[symbol] = MatchingEngine(symbol, **engine_kwargs)
        
        self.logger = get_logger(__name__)
        self.logger.info(f"MultiSymbolMatchingEngine initialized for {len(symbols)} symbols")
    
    def process_order(self, order: Order) -> Tuple[List[Trade], OrderUpdate]:
        """Process order for the appropriate symbol"""
        if order.symbol not in self.engines:
            raise ValueError(f"No matching engine for symbol: {order.symbol}")
        
        return self.engines[order.symbol].process_order(order)
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order for specific symbol"""
        if symbol not in self.engines:
            return False
        
        return self.engines[symbol].cancel_order(order_id)
    
    def get_snapshot(self, symbol: str) -> Optional[BookSnapshot]:
        """Get market snapshot for specific symbol"""
        if symbol not in self.engines:
            return None
        
        return self.engines[symbol].get_market_snapshot()
    
    def get_all_snapshots(self) -> Dict[str, BookSnapshot]:
        """Get market snapshots for all symbols"""
        return {symbol: engine.get_market_snapshot() 
                for symbol, engine in self.engines.items()}
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all symbols"""
        return {symbol: engine.get_statistics() 
                for symbol, engine in self.engines.items()}
    
    def set_market_state(self, is_open: bool) -> None:
        """Set market state for all symbols"""
        for engine in self.engines.values():
            engine.set_market_state(is_open)
    
    def reset_session(self) -> None:
        """Reset session for all symbols"""
        for engine in self.engines.values():
            engine.reset_session()


# Utility functions for matching engine analysis
def analyze_matching_performance(engine: MatchingEngine) -> Dict[str, Any]:
    """
    Analyze matching engine performance
    
    Args:
        engine: MatchingEngine to analyze
        
    Returns:
        Dictionary with performance analysis
    """
    stats = engine.get_statistics()
    trades = engine.get_trade_history()
    
    if not trades:
        return {'error': 'No trades to analyze'}
    
    # Calculate trade statistics
    trade_prices = [trade.price for trade in trades]
    trade_volumes = [trade.volume for trade in trades]
    
    analysis = {
        'basic_stats': stats,
        'trade_analysis': {
            'total_trades': len(trades),
            'avg_trade_price': np.mean(trade_prices),
            'price_volatility': np.std(trade_prices),
            'avg_trade_size': np.mean(trade_volumes),
            'total_volume': sum(trade_volumes),
            'price_range': {
                'min': min(trade_prices),
                'max': max(trade_prices),
                'range': max(trade_prices) - min(trade_prices)
            }
        },
        'performance_metrics': {
            'throughput_orders_per_second': stats['orders_processed'] / max(1, stats['processing_time_us'] / 1_000_000),
            'fill_rate': 1 - stats['rejection_rate'],
            'avg_processing_latency_us': stats['avg_processing_time_us']
        }
    }
    
    return analysis