"""
Order Book Engine for HFT Simulator

This module implements the core order book functionality that maintains real-time
bid-ask ladders and processes order updates efficiently.

Educational Notes:
- The order book is the central data structure in electronic trading
- It maintains all buy (bid) and sell (ask) orders sorted by price and time
- Price-time priority: better prices get priority, then earlier orders
- The order book enables price discovery and efficient order matching
- Real exchanges use similar (but more complex) order book implementations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Iterator
from collections import defaultdict, OrderedDict
from datetime import datetime
import bisect
import warnings

from src.utils.logger import get_logger, log_performance, log_memory_usage
from src.utils.helpers import Timer, format_price, format_volume
from src.utils.constants import OrderSide, OrderType, OrderStatus, EPSILON
from src.utils.validation import validate_price, validate_quantity, validate_symbol, OrderValidationError
from src.utils.error_handler import ErrorHandler
from src.performance.circuit_breaker import CircuitBreaker
from .order_types import Order, Trade, OrderUpdate, PriceLevel
from .market_data import BookSnapshot, MarketData
from .recovery import OrderBookRecovery


class OrderBook:
    """
    High-performance order book implementation for HFT simulation
    
    This class maintains the complete state of buy and sell orders for a financial
    instrument, providing efficient operations for:
    - Adding new orders
    - Canceling existing orders
    - Matching orders to create trades
    - Querying market depth and statistics
    - Creating market data snapshots
    
    Key Features:
    - Price-time priority matching
    - Efficient price level management
    - Real-time market data generation
    - Comprehensive order tracking
    - Performance optimized for high-frequency updates
    
    Example Usage:
        >>> book = OrderBook("AAPL")
        >>> order = Order.create_limit_order("AAPL", OrderSide.BID, 100, 150.00)
        >>> book.add_order(order)
        >>> snapshot = book.get_snapshot()
        >>> print(f"Best bid: {snapshot.best_bid}")
    """
    
    def __init__(self, symbol: str, tick_size: float = 0.01):
        """
        Initialize the order book for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., "AAPL", "MSFT")
            tick_size: Minimum price increment
        """
        try:
            validate_symbol(symbol)
        except OrderValidationError as e:
            raise ValueError(f"Invalid symbol: {e}")
            
        self.symbol = symbol
        self.tick_size = tick_size
        self.logger = get_logger(f"{__name__}.{symbol}")
        
        # Error handling and circuit breaker
        self.circuit_breaker = CircuitBreaker()
        self.error_handler = ErrorHandler()
        self.correlation_id = None
        
        # Order storage: price -> PriceLevel
        self.bids: OrderedDict[float, PriceLevel] = OrderedDict()  # Sorted descending
        self.asks: OrderedDict[float, PriceLevel] = OrderedDict()  # Sorted ascending
        
        # Order tracking: order_id -> Order
        self.orders: Dict[str, Order] = {}
        
        # Price level tracking for efficient operations
        self.bid_prices: List[float] = []  # Sorted descending
        self.ask_prices: List[float] = []  # Sorted ascending
        
        # Market data
        self.market_data = MarketData(symbol)
        
        # Statistics and state
        self.sequence_number = 0
        self.last_trade_price: Optional[float] = None
        self.last_trade_volume: Optional[int] = None
        self.last_trade_timestamp: Optional[pd.Timestamp] = None
        
        # Performance tracking
        self.stats = {
            'orders_added': 0,
            'orders_cancelled': 0,
            'trades_executed': 0,
            'total_volume_traded': 0,
            'updates_processed': 0,
        }
        
        self.logger.info(f"OrderBook initialized for {symbol} with tick_size={tick_size}")
    
    @log_performance
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add a new order to the book
        
        Args:
            order: Order to add
            
        Returns:
            List of trades generated (empty for limit orders that don't match)
            
        Educational Notes:
        - Market orders are matched immediately against available liquidity
        - Limit orders are added to the book if they don't match immediately
        - Matching follows price-time priority rules
        - Partial fills are supported for large orders
        """
        # Generate correlation ID for error tracking
        self.correlation_id = self.error_handler.log_info(f"Processing order {order.order_id}")
        
        try:
            # Validate order parameters
            validate_price(order.price)
            validate_quantity(order.volume)
            validate_symbol(order.symbol)
        except OrderValidationError as e:
            self.error_handler.log_error(e, self.correlation_id, {"order_id": order.order_id})
            raise ValueError(f"Order validation failed: {e}")
        
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol} doesn't match book symbol {self.symbol}")
        
        if order.order_id in self.orders:
            raise ValueError(f"Order {order.order_id} already exists in book")
        
        # Check circuit breaker conditions before processing
        current_price = order.price if order.is_limit_order() else (self.get_mid_price() or order.price)
        reference_price = self.last_trade_price or current_price
        total_liquidity = self.get_total_volume(OrderSide.BID) + self.get_total_volume(OrderSide.ASK)
        
        if self.circuit_breaker.check(current_price, reference_price, total_liquidity, self.correlation_id):
            self.error_handler.log_warning(f"Order {order.order_id} rejected due to circuit breaker", self.correlation_id)
            order.status = OrderStatus.REJECTED
            return []
        
        # Log order entry into matching logic
        self.logger.debug(f"[ORDER_ENTRY] Order {order.order_id} entering matching pipeline: "
                         f"Side={order.side.value}, Type={order.order_type.value}, "
                         f"Price={order.price}, Volume={order.volume}, "
                         f"Symbol={order.symbol}")
        
        # Log current market state before matching
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        mid_price = self.get_mid_price()
        spread = self.get_spread()
        
        self.logger.debug(f"[MARKET_STATE_PRE_MATCH] Before processing order {order.order_id}: "
                         f"BestBid={best_bid}, BestAsk={best_ask}, Mid={mid_price}, "
                         f"Spread={spread}, BidLevels={len(self.bids)}, AskLevels={len(self.asks)}")
        
        trades = []
        
        # Market orders: try to match immediately
        if order.is_market_order():
            self.logger.debug(f"[MATCHING_DECISION] Order {order.order_id} is MARKET order - routing to market matching")
            trades = self._match_market_order(order)
        
        # Limit orders: try to match, then add remainder to book
        elif order.is_limit_order():
            self.logger.debug(f"[MATCHING_DECISION] Order {order.order_id} is LIMIT order at {order.price} - routing to limit matching")
            trades = self._match_limit_order(order)
            
            # Add remaining volume to book if not fully filled
            if order.remaining_volume > 0 and order.is_active():
                self.logger.debug(f"[ORDER_STATE_CHANGE] Order {order.order_id} partially filled: "
                                f"FilledVolume={order.volume - order.remaining_volume}, "
                                f"RemainingVolume={order.remaining_volume}, Status={order.status} - adding to book")
                self._add_order_to_book(order)
            elif order.remaining_volume == 0:
                self.logger.debug(f"[ORDER_STATE_CHANGE] Order {order.order_id} fully filled: "
                                f"Status={order.status}")
        
        # Update statistics
        self.stats['orders_added'] += 1
        self.stats['trades_executed'] += len(trades)
        self.stats['total_volume_traded'] += sum(trade.volume for trade in trades)
        self.stats['updates_processed'] += 1
        
        # Update market data
        if trades:
            last_trade = trades[-1]
            self.last_trade_price = last_trade.price
            self.last_trade_volume = last_trade.volume
            self.last_trade_timestamp = last_trade.timestamp
            
            # Add trades to market data
            for trade in trades:
                self.market_data.add_trade(trade.to_dict())
        
        self._increment_sequence()
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if order was successfully cancelled
        """
        if order_id not in self.orders:
            self.logger.warning(f"Cannot cancel order {order_id}: not found")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active():
            self.logger.warning(f"Cannot cancel order {order_id}: not active (status: {order.status})")
            return False
        
        self.logger.debug(f"Cancelling order: {order_id}")
        
        # Remove from price level
        self._remove_order_from_book(order)
        
        # Update order status
        order.cancel()
        
        # Update statistics
        self.stats['orders_cancelled'] += 1
        self.stats['updates_processed'] += 1
        
        self._increment_sequence()
        
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_volume: Optional[int] = None) -> bool:
        """
        Modify an existing order (cancel and replace)
        
        Args:
            order_id: ID of order to modify
            new_price: New price (None to keep current)
            new_volume: New volume (None to keep current)
            
        Returns:
            True if order was successfully modified
            
        Educational Notes:
        - Order modification is typically implemented as cancel-replace
        - The modified order loses its time priority
        - Some exchanges support price improvement without losing priority
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active():
            return False
        
        # Support alternate test signature: modify_order(order_id, original_order, modified_order)
        # If callers pass Order objects instead of price/volume, adapt accordingly
        if isinstance(new_price, Order) and isinstance(new_volume, Order):
            original_order: Order = new_price
            modified_order: Order = new_volume
            # Sanity check
            if original_order.order_id != order_id or modified_order.order_id != order_id:
                return False
            target_price = modified_order.price
            target_volume = modified_order.volume
        else:
            target_price = new_price if new_price is not None else order.price
            # Use original requested total volume when modifying if new volume unspecified
            target_volume = new_volume if new_volume is not None else order.volume
        
        # Cancel the existing order
        if not self.cancel_order(order_id):
            return False
        
        # Create new order with modified parameters
        new_order = Order(
            order_id=order_id,  # Keep same ID
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            price=target_price,
            volume=target_volume,
            timestamp=pd.Timestamp.now(),  # New timestamp (loses time priority)
            source=order.source,
            metadata=order.metadata.copy()
        )
        
        # Remove any residual reference of old order id to avoid duplicate check during add
        if order_id in self.orders:
            del self.orders[order_id]
        # Add the modified order (keeping same ID)
        self.add_order(new_order)
        
        return True
    
    def _match_market_order(self, order: Order) -> List[Trade]:
        """Match a market order against available liquidity
        
        Market orders should:
        1. Check against the opposite side of the book (buy orders check asks, sell orders check bids)
        2. Match at the best available price regardless of their price attribute
        3. Be processed immediately and not added to the book if unfilled
        4. Be rejected if no liquidity is available
        """
        self.logger.debug(f"[MARKET_ORDER_MATCHING] Starting market order matching for order {order.order_id}")
        trades = []
        
        # Buy market orders match against asks (sell side)
        if order.is_buy():
            price_levels = self.asks
            prices = self.ask_prices.copy()  # Use copy to avoid modification during iteration
            opposite_side = "ASK"
        else:
            # Sell market orders match against bids (buy side)
            price_levels = self.bids
            prices = self.bid_prices.copy()  # Use copy to avoid modification during iteration
            opposite_side = "BID"
        
        self.logger.debug(f"[PRICE_COMPARISON] Market {order.side.value} order {order.order_id} "
                         f"matching against {opposite_side} side with {len(prices)} price levels")
        
        remaining_volume = order.volume
        
        # If no liquidity available on the opposite side, reject the order immediately
        if not prices or not price_levels:
            self.logger.debug(f"[MATCHING_DECISION] No liquidity on {opposite_side} side - rejecting market order {order.order_id}")
            order.status = OrderStatus.REJECTED
            # Market orders should not be added to the book - just track for history
            self.orders[order.order_id] = order
            return []
        
        self.logger.debug(f"[LIQUIDITY_ANALYSIS] Available {opposite_side} levels: {[(p, price_levels[p].total_volume) for p in prices[:5]]}")
        
        # Match against price levels in order (best prices first)
        for i, price in enumerate(prices):
            if remaining_volume <= 0:
                self.logger.debug(f"[MATCHING_COMPLETE] Order {order.order_id} fully matched after {i} levels")
                break
            
            level = price_levels[price]
            if level.is_empty():
                self.logger.debug(f"[PRICE_LEVEL_SKIP] Skipping empty price level at {price}")
                continue
                
            volume_to_match = min(remaining_volume, level.total_volume)
            
            self.logger.debug(f"[PRICE_COMPARISON] Level {i+1}: Price={price}, Available={level.total_volume}, "
                             f"ToMatch={volume_to_match}, Remaining={remaining_volume}")
            
            if volume_to_match > 0:
                self.logger.debug(f"[MATCHING_DECISION] Executing {volume_to_match} volume at price {price} "
                                f"for order {order.order_id}")
                # Execute trades with orders at this price level
                level_trades = self._execute_trades_at_level(order, level, volume_to_match)
                trades.extend(level_trades)
                remaining_volume -= volume_to_match
                
                self.logger.debug(f"[TRADE_GENERATION] Generated {len(level_trades)} trades, "
                                f"remaining volume: {remaining_volume}")
        
        # Update order status based on fill results
        order.remaining_volume = remaining_volume
        if remaining_volume == 0:
            order.status = OrderStatus.FILLED
        elif remaining_volume < order.volume:
            # Market orders with partial fills: the filled portion is complete
            # The remaining volume is effectively cancelled/rejected since market orders
            # should not remain in the book
            order.status = OrderStatus.PARTIAL
            # Set remaining volume to 0 to indicate the unfilled portion is cancelled
            order.remaining_volume = 0
        else:
            # No liquidity available - reject the order
            order.status = OrderStatus.REJECTED
        
        # Store the order for tracking (but not in the active book)
        # Market orders should not remain in the book regardless of fill status
        self.orders[order.order_id] = order
        
        return trades
    
    def _match_limit_order(self, order: Order) -> List[Trade]:
        """Match a limit order against available liquidity
        
        Limit orders should:
        1. Buy orders match against asks at or below the limit price
        2. Sell orders match against bids at or above the limit price
        3. Match at the best available prices in price-time priority order
        4. Remaining unfilled volume stays in the book
        """
        self.logger.debug(f"[LIMIT_ORDER_MATCHING] Starting limit order matching for order {order.order_id} "
                         f"at limit price {order.price}")
        trades = []
        
        # Buy limit orders can match against asks at or below the limit price
        if order.is_buy():
            price_levels = self.asks
            all_ask_prices = self.ask_prices.copy()
            prices = [p for p in all_ask_prices if p <= order.price]
            opposite_side = "ASK"
            comparison_op = "<="
        else:
            # Sell limit orders can match against bids at or above the limit price
            price_levels = self.bids
            all_bid_prices = self.bid_prices.copy()
            prices = [p for p in all_bid_prices if p >= order.price]
            opposite_side = "BID"
            comparison_op = ">="
        
        self.logger.debug(f"[PRICE_COMPARISON] Limit {order.side.value} order {order.order_id} at {order.price} "
                         f"can match against {opposite_side} prices {comparison_op} {order.price}")
        self.logger.debug(f"[ELIGIBLE_PRICES] Found {len(prices)} eligible price levels: {prices[:10]}")
        
        remaining_volume = order.volume
        
        if not prices:
            self.logger.debug(f"[MATCHING_DECISION] No eligible prices for limit order {order.order_id} - "
                             f"order will be added to book")
        
        # Match against eligible price levels in order of best prices first
        for i, price in enumerate(prices):
            if remaining_volume <= 0:
                self.logger.debug(f"[MATCHING_COMPLETE] Limit order {order.order_id} fully matched after {i} levels")
                break
            
            level = price_levels[price]
            if level.is_empty():
                self.logger.debug(f"[PRICE_LEVEL_SKIP] Skipping empty price level at {price}")
                continue
                
            volume_to_match = min(remaining_volume, level.total_volume)
            
            self.logger.debug(f"[PRICE_COMPARISON] Level {i+1}: Price={price} (favorable to limit {order.price}), "
                             f"Available={level.total_volume}, ToMatch={volume_to_match}, Remaining={remaining_volume}")
            
            if volume_to_match > 0:
                self.logger.debug(f"[MATCHING_DECISION] Executing {volume_to_match} volume at price {price} "
                                f"for limit order {order.order_id}")
                level_trades = self._execute_trades_at_level(order, level, volume_to_match)
                trades.extend(level_trades)
                remaining_volume -= volume_to_match
                
                self.logger.debug(f"[TRADE_GENERATION] Generated {len(level_trades)} trades, "
                                f"remaining volume: {remaining_volume}")
        
        # Update order status and remaining volume
        order.remaining_volume = remaining_volume
        if remaining_volume == 0:
            order.status = OrderStatus.FILLED
            self.logger.debug(f"[ORDER_STATE_CHANGE] Limit order {order.order_id} FULLY FILLED")
        elif remaining_volume < order.volume:
            order.status = OrderStatus.PARTIAL
            self.logger.debug(f"[ORDER_STATE_CHANGE] Limit order {order.order_id} PARTIALLY FILLED: "
                             f"FilledVolume={order.volume - remaining_volume}, RemainingVolume={remaining_volume}")
        else:
            self.logger.debug(f"[ORDER_STATE_CHANGE] Limit order {order.order_id} NO FILL - remains PENDING")
        
        # Store the order for tracking
        self.orders[order.order_id] = order
        
        return trades
    
    def _execute_trades_at_level(self, incoming_order: Order, level: PriceLevel, 
                                volume: int) -> List[Trade]:
        """Execute trades between incoming order and orders at a price level"""
        self.logger.debug(f"[TRADE_EXECUTION] Executing trades at price level {level.price}: "
                         f"IncomingOrder={incoming_order.order_id}, Volume={volume}, "
                         f"RestingOrders={len(level.orders)}")
        
        trades = []
        remaining_volume = volume
        
        # Match against orders in time priority (FIFO)
        orders_to_remove = []
        
        for i, resting_order in enumerate(level.orders):
            if remaining_volume <= 0:
                self.logger.debug(f"[TRADE_EXECUTION] All volume matched after {i} resting orders")
                break
            
            if not resting_order.is_active():
                self.logger.debug(f"[TRADE_EXECUTION] Skipping inactive resting order {resting_order.order_id}")
                orders_to_remove.append(resting_order)
                continue
            
            # Determine trade volume
            trade_volume = min(remaining_volume, resting_order.remaining_volume)
            
            self.logger.debug(f"[TIME_PRIORITY] Order {i+1} in queue: RestingOrder={resting_order.order_id}, "
                             f"Available={resting_order.remaining_volume}, ToTrade={trade_volume}")
            
            if trade_volume > 0:
                # Create trade
                trade_id = f"T{self.sequence_number}_{len(trades)}"
                trade = Trade(
                    trade_id=trade_id,
                    symbol=self.symbol,
                    price=level.price,  # Trade at resting order's price
                    volume=trade_volume,
                    timestamp=pd.Timestamp.now(),
                    buy_order_id=incoming_order.order_id if incoming_order.is_buy() else resting_order.order_id,
                    sell_order_id=resting_order.order_id if resting_order.is_sell() else incoming_order.order_id,
                    aggressor_side=incoming_order.side
                )
                
                self.logger.debug(f"[TRADE_CREATED] Trade {trade_id} created: "
                                f"BuyOrder={trade.buy_order_id}, SellOrder={trade.sell_order_id}, "
                                f"Price={trade.price}, Volume={trade.volume}, Aggressor={trade.aggressor_side.value}")
                
                trades.append(trade)
                
                # Update orders with fill information
                incoming_before = incoming_order.remaining_volume
                resting_before = resting_order.remaining_volume
                
                incoming_order.partial_fill(trade_volume, level.price)
                resting_order.partial_fill(trade_volume, level.price)
                
                self.logger.debug(f"[ORDER_FILLS] IncomingOrder {incoming_order.order_id}: "
                                f"{incoming_before} -> {incoming_order.remaining_volume}, "
                                f"RestingOrder {resting_order.order_id}: "
                                f"{resting_before} -> {resting_order.remaining_volume}")
                
                remaining_volume -= trade_volume
                
                # Mark filled orders for removal
                if resting_order.is_filled():
                    self.logger.debug(f"[ORDER_STATE_CHANGE] Resting order {resting_order.order_id} "
                                    f"fully filled - marking for removal")
                    orders_to_remove.append(resting_order)
                elif resting_order.status == OrderStatus.PARTIAL:
                    self.logger.debug(f"[ORDER_STATE_CHANGE] Resting order {resting_order.order_id} "
                                    f"partially filled - remaining: {resting_order.remaining_volume}")
        
        # Remove filled orders from the level
        for order in orders_to_remove:
            level.remove_order(order.order_id)
        
        # Update level volume
        level.total_volume = sum(order.remaining_volume for order in level.orders)
        
        # Remove empty price level and ensure price list ordering
        if level.is_empty():
            # Remove from the correct side by inspecting the level price location
            is_buy_side = incoming_order.is_sell()  # If we matched sells, we consumed bids
            # However, level belongs to opposite side of incoming_order
            # Determine side based on presence in asks/bids
            is_buy_side = level.price in self.bid_prices
            self._remove_price_level(level.price, is_buy_side)
        
        return trades
    
    def _add_order_to_book(self, order: Order) -> None:
        """Add an order to the appropriate price level in the book"""
        price = order.price
        
        # Determine which side of the book
        if order.is_buy():
            price_levels = self.bids
            prices = self.bid_prices
        else:
            price_levels = self.asks
            prices = self.ask_prices
        
        # Create price level if it doesn't exist
        if price not in price_levels:
            price_levels[price] = PriceLevel(price)
            
            # Insert price in sorted order efficiently
            if order.is_buy():
                # Maintain descending order (highest first) without full sort
                # Custom binary search for descending list
                lo, hi = 0, len(prices)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if prices[mid] > price:
                        lo = mid + 1
                    else:
                        hi = mid
                prices.insert(lo, price)
                # Rebuild OrderedDict to reflect sorted order
                self.bids = OrderedDict((p, self.bids[p]) for p in prices)
            else:
                # Ascending order
                idx = bisect.bisect_left(prices, price)
                prices.insert(idx, price)
                # Rebuild OrderedDict to reflect sorted order
                self.asks = OrderedDict((p, self.asks[p]) for p in prices)
        
        # Add order to price level
        price_levels[price].add_order(order)
        
        # Store order reference
        self.orders[order.order_id] = order
    
    def _remove_order_from_book(self, order: Order) -> None:
        """Remove an order from the book"""
        price = order.price
        
        # Determine which side of the book
        if order.is_buy():
            price_levels = self.bids
            prices = self.bid_prices
        else:
            price_levels = self.asks
            prices = self.ask_prices
        
        # Remove from price level
        if price in price_levels:
            level = price_levels[price]
            level.remove_order(order.order_id)
            
            # Remove empty price level
            if level.is_empty():
                self._remove_price_level(price, order.is_buy())
    
    def _remove_price_level(self, price: float, is_buy_side: bool) -> None:
        """Remove an empty price level from the book"""
        if is_buy_side:
            if price in self.bids:
                del self.bids[price]
            if price in self.bid_prices:
                self.bid_prices.remove(price)
            # Ensure order is preserved (descending)
            self.bids = OrderedDict((p, self.bids[p]) for p in sorted(self.bids.keys(), reverse=True))
        else:
            if price in self.asks:
                del self.asks[price]
            if price in self.ask_prices:
                self.ask_prices.remove(price)
            # Ensure order is preserved (ascending)
            self.asks = OrderedDict((p, self.asks[p]) for p in sorted(self.asks.keys()))
    
    def _increment_sequence(self) -> None:
        """Increment sequence number and update market data"""
        self.sequence_number += 1
        
        # Update market data snapshot
        snapshot = self.get_snapshot()
        self.market_data.update_snapshot(snapshot)
    
    @log_performance
    def get_snapshot(self) -> BookSnapshot:
        """
        Create a snapshot of the current order book state
        
        Returns:
            BookSnapshot with current bid/ask levels
        """
        # Convert price levels to snapshot format
        bid_levels = [level for level in self.bids.values() if not level.is_empty()]
        ask_levels = [level for level in self.asks.values() if not level.is_empty()]
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=pd.Timestamp.now(),
            bids=bid_levels,
            asks=ask_levels,
            last_trade_price=self.last_trade_price,
            last_trade_volume=self.last_trade_volume,
            last_trade_timestamp=self.last_trade_timestamp,
            sequence_number=self.sequence_number
        )
        
        return snapshot
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price"""
        return self.bid_prices[0] if self.bid_prices else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price"""
        return self.ask_prices[0] if self.ask_prices else None

    # Compatibility helpers for performance tests
    def get_bids(self, levels: int = 10) -> List[Tuple[float, int]]:
        return [(p, self.bids[p].total_volume) for p in self.bid_prices[:levels]]

    def get_asks(self, levels: int = 10) -> List[Tuple[float, int]]:
        return [(p, self.asks[p].total_volume) for p in self.ask_prices[:levels]]

    def process_order(self, order: Order) -> List[Trade]:
        return self.add_order(order)
    
    def get_best_bid_volume(self) -> Optional[int]:
        """Get the volume at the best bid"""
        best_bid = self.get_best_bid()
        return self.bids[best_bid].total_volume if best_bid else None
    
    def get_best_ask_volume(self) -> Optional[int]:
        """Get the volume at the best ask"""
        best_ask = self.get_best_ask()
        return self.asks[best_ask].total_volume if best_ask else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid-price (average of best bid and ask)"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_depth(self, side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]:
        """
        Get market depth for specified side
        
        Args:
            side: Which side of the book
            levels: Number of price levels to return
            
        Returns:
            List of (price, volume) tuples
        """
        if side in [OrderSide.BID, OrderSide.BUY]:
            prices = self.bid_prices[:levels]
            price_levels = self.bids
        else:
            prices = self.ask_prices[:levels]
            price_levels = self.asks
        
        return [(price, price_levels[price].total_volume) for price in prices]
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get an order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_at_price(self, price: float, side: OrderSide) -> List[Order]:
        """Get all orders at a specific price level"""
        if side in [OrderSide.BID, OrderSide.BUY]:
            level = self.bids.get(price)
        else:
            level = self.asks.get(price)
        
        return level.orders.copy() if level else []
    
    def get_total_volume(self, side: OrderSide) -> int:
        """Get total volume on one side of the book"""
        if side in [OrderSide.BID, OrderSide.BUY]:
            return sum(level.total_volume for level in self.bids.values())
        else:
            return sum(level.total_volume for level in self.asks.values())
    
    def get_order_count(self) -> int:
        """Get total number of active orders in the book"""
        return len([order for order in self.orders.values() if order.is_active()])
    
    def is_crossed(self) -> bool:
        """Check if the book is crossed (bid >= ask)"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_bid >= best_ask
        return False
    
    def is_empty(self) -> bool:
        """Check if the book is empty"""
        return len(self.bid_prices) == 0 and len(self.ask_prices) == 0
    
    def clear(self) -> None:
        """Clear all orders from the book"""
        self.bids.clear()
        self.asks.clear()
        self.bid_prices.clear()
        self.ask_prices.clear()
        self.orders.clear()
        self.sequence_number = 0
        
        self.logger.info(f"Order book cleared for {self.symbol}")
    
    def create_snapshot(self) -> bool:
        """Create a snapshot of the current order book state"""
        try:
            return OrderBookRecovery.snapshot(self, self.correlation_id)
        except Exception as e:
            self.error_handler.log_error(e, self.correlation_id, {"symbol": self.symbol})
            return False
    
    def restore_from_snapshot(self) -> bool:
        """Restore order book from the most recent snapshot"""
        try:
            restored_book = OrderBookRecovery.restore(self.correlation_id)
            if restored_book is not None:
                # Replace current state with restored state
                self.bids = restored_book.bids
                self.asks = restored_book.asks
                self.orders = restored_book.orders
                self.bid_prices = restored_book.bid_prices
                self.ask_prices = restored_book.ask_prices
                self.sequence_number = restored_book.sequence_number
                self.last_trade_price = restored_book.last_trade_price
                self.last_trade_volume = restored_book.last_trade_volume
                self.last_trade_timestamp = restored_book.last_trade_timestamp
                self.stats = restored_book.stats
                self.error_handler.log_info(f"Order book restored for {self.symbol}", self.correlation_id)
                return True
            return False
        except Exception as e:
            self.error_handler.log_error(e, self.correlation_id, {"symbol": self.symbol})
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive order book statistics"""
        snapshot = self.get_snapshot()
        
        return {
            'symbol': self.symbol,
            'sequence_number': self.sequence_number,
            'total_orders': len(self.orders),
            'active_orders': self.get_order_count(),
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'total_bid_volume': self.get_total_volume(OrderSide.BID),
            'total_ask_volume': self.get_total_volume(OrderSide.ASK),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'is_crossed': self.is_crossed(),
            'is_empty': self.is_empty(),
            'last_trade_price': self.last_trade_price,
            'processing_stats': self.stats.copy(),
        }
    
    def to_dataframe(self, side: Optional[OrderSide] = None, levels: int = 10) -> pd.DataFrame:
        """
        Convert order book to DataFrame representation
        
        Args:
            side: Which side to include (None for both)
            levels: Number of price levels to include
            
        Returns:
            DataFrame with order book data
        """
        data = []
        
        if side is None or side in [OrderSide.BID, OrderSide.BUY]:
            # Add bid levels
            for price in self.bid_prices[:levels]:
                level = self.bids[price]
                data.append({
                    'side': 'bid',
                    'price': price,
                    'volume': level.total_volume,
                    'orders': level.order_count,
                    'distance_from_mid': price - (self.get_mid_price() or price)
                })
        
        if side is None or side in [OrderSide.ASK, OrderSide.SELL]:
            # Add ask levels
            for price in self.ask_prices[:levels]:
                level = self.asks[price]
                data.append({
                    'side': 'ask',
                    'price': price,
                    'volume': level.total_volume,
                    'orders': level.order_count,
                    'distance_from_mid': price - (self.get_mid_price() or price)
                })
        
        return pd.DataFrame(data)
    
    def __str__(self) -> str:
        """String representation of the order book"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        
        bid_str = f"{format_price(best_bid)}x{format_volume(self.get_best_bid_volume())}" if best_bid else "---"
        ask_str = f"{format_price(best_ask)}x{format_volume(self.get_best_ask_volume())}" if best_ask else "---"
        spread_str = f"{spread:.4f}" if spread else "---"
        
        return (f"OrderBook({self.symbol}): {bid_str} | {ask_str} "
                f"(spread: {spread_str}, orders: {self.get_order_count()})")
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for order book analysis
def reconstruct_book_from_updates(symbol: str, updates: List[OrderUpdate]) -> OrderBook:
    """
    Reconstruct an order book from a sequence of updates
    
    Args:
        symbol: Trading symbol
        updates: List of order updates in chronological order
        
    Returns:
        Reconstructed OrderBook
        
    Educational Notes:
    - This function simulates how order books are built from market data feeds
    - It's essential for backtesting with historical tick data
    - Updates must be processed in chronological order
    """
    book = OrderBook(symbol)
    
    for update in updates:
        if update.update_type == 'new' and update.order_id:
            # Create order from update
            order = Order(
                order_id=update.order_id,
                symbol=update.symbol,
                side=update.side,
                order_type=OrderType.LIMIT,  # Assume limit orders
                price=update.price,
                volume=update.volume,
                timestamp=update.timestamp,
                source=update.source
            )
            book.add_order(order)
        
        elif update.update_type == 'cancel' and update.order_id:
            book.cancel_order(update.order_id)
    
    return book


def compare_books(book1: OrderBook, book2: OrderBook, tolerance: float = 1e-8) -> Dict[str, Any]:
    """
    Compare two order books for differences
    
    Args:
        book1: First order book
        book2: Second order book
        tolerance: Price comparison tolerance
        
    Returns:
        Dictionary with comparison results
    """
    if book1.symbol != book2.symbol:
        return {'error': 'Different symbols'}
    
    snapshot1 = book1.get_snapshot()
    snapshot2 = book2.get_snapshot()
    
    differences = {
        'symbols_match': book1.symbol == book2.symbol,
        'best_bid_match': abs((snapshot1.best_bid or 0) - (snapshot2.best_bid or 0)) < tolerance,
        'best_ask_match': abs((snapshot1.best_ask or 0) - (snapshot2.best_ask or 0)) < tolerance,
        'bid_volume_match': snapshot1.best_bid_volume == snapshot2.best_bid_volume,
        'ask_volume_match': snapshot1.best_ask_volume == snapshot2.best_ask_volume,
        'order_count_match': book1.get_order_count() == book2.get_order_count(),
        'level_count_match': {
            'bids': len(book1.bids) == len(book2.bids),
            'asks': len(book1.asks) == len(book2.asks)
        }
    }
    
    differences['books_identical'] = all([
        differences['symbols_match'],
        differences['best_bid_match'],
        differences['best_ask_match'],
        differences['bid_volume_match'],
        differences['ask_volume_match'],
        differences['order_count_match'],
        differences['level_count_match']['bids'],
        differences['level_count_match']['asks']
    ])
    
    return differences