"""
Unified Order Book for Multi-Asset Trading

This module provides a unified order book implementation that can handle
different asset classes with their specific characteristics and requirements.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import pandas as pd
import numpy as np
import heapq
from dataclasses import dataclass

from src.engine.order_types import Order, Trade, OrderUpdate
from src.engine.order_book import OrderBook as BaseOrderBook
from src.utils.constants import OrderSide, OrderType, OrderStatus
from .base_asset import BaseAsset, AssetType


@dataclass
class OrderBookLevel:
    """Order book price level information"""
    price: float
    volume: int
    order_count: int
    total_size: int = 0
    
    def __post_init__(self):
        if self.total_size == 0:
            self.total_size = self.volume


class UnifiedOrderBook:
    """
    Multi-asset order book that adapts to different asset class requirements
    
    Features:
    - Asset-specific tick sizes and lot sizes
    - Different matching algorithms per asset type
    - Cross-asset spread and synthetic instrument support
    - Real-time market data aggregation
    """
    
    def __init__(self, asset: BaseAsset):
        self.asset = asset
        self.symbol = asset.symbol
        self.asset_type = asset.asset_type
        
        # Order book data structures
        self._bids: Dict[float, List[Order]] = defaultdict(list)  # Price -> [Orders]
        self._asks: Dict[float, List[Order]] = defaultdict(list)  # Price -> [Orders]
        self._orders: Dict[str, Order] = {}  # order_id -> Order
        self._trades: List[Trade] = []
        
        # Market data
        self._bid_prices = []  # Max heap (negate prices)
        self._ask_prices = []  # Min heap
        self._last_trade_price: Optional[float] = None
        self._last_trade_time: Optional[pd.Timestamp] = None
        
        # Statistics
        self._daily_volume = 0
        self._daily_trade_count = 0
        self._total_value_traded = 0.0
        
        # Asset-specific configurations
        self._setup_asset_specific_config()
    
    def _setup_asset_specific_config(self):
        """Configure order book based on asset type"""
        self.tick_size = self.asset.asset_info.tick_size
        self.lot_size = self.asset.asset_info.lot_size
        self.min_quantity = self.asset.asset_info.min_trade_size
        
        # Asset-specific matching rules
        if self.asset_type == AssetType.CRYPTO:
            self._use_pro_rata_matching = True
            self._allow_fractional_shares = True
        elif self.asset_type == AssetType.FX:
            self._use_pro_rata_matching = True
            self._allow_fractional_shares = True
        elif self.asset_type == AssetType.OPTIONS:
            self._use_price_time_priority = True
            self._allow_fractional_shares = False
        else:  # EQUITY, FUTURES, FIXED_INCOME
            self._use_price_time_priority = True
            self._allow_fractional_shares = False
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best (highest) bid price"""
        if not self._bid_prices:
            return None
        return -self._bid_prices[0]  # Negate back to positive
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best (lowest) ask price"""
        if not self._ask_prices:
            return None
        return self._ask_prices[0]
    
    @property
    def bid_size(self) -> int:
        """Get total volume at best bid"""
        best_bid = self.best_bid
        if best_bid is None:
            return 0
        return sum(order.remaining_volume for order in self._bids[best_bid])
    
    @property
    def ask_size(self) -> int:
        """Get total volume at best ask"""
        best_ask = self.best_ask
        if best_ask is None:
            return 0
        return sum(order.remaining_volume for order in self._asks[best_ask])
    
    @property
    def spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid
    
    @property
    def mid_price(self) -> Optional[float]:
        """Get mid price"""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2
    
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add order to book and return any resulting trades
        
        Args:
            order: Order to add
            
        Returns:
            List of trades generated from matching
        """
        # Validate order with asset
        is_valid, error_msg = self.asset.validate_order(order)
        if not is_valid:
            order.status = OrderStatus.REJECTED
            raise ValueError(f"Order validation failed: {error_msg}")
        
        # Round price to tick size
        if order.price is not None:
            order.price = self._round_to_tick_size(order.price)
        
        # Store order
        self._orders[order.order_id] = order
        
        # Try to match order
        trades = self._match_order(order)
        
        # Add remaining quantity to book if not fully filled
        if order.remaining_volume > 0 and order.status != OrderStatus.CANCELLED:
            self._add_order_to_book(order)
        
        return trades
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if not order.is_active():
            return False
        
        # Remove from book
        self._remove_order_from_book(order)
        
        # Update order status
        order.cancel()
        
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_volume: Optional[int] = None) -> List[Trade]:
        """Modify an existing order"""
        if order_id not in self._orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self._orders[order_id]
        if not order.is_active():
            raise ValueError(f"Cannot modify inactive order {order_id}")
        
        # Remove from book
        self._remove_order_from_book(order)
        
        # Update order parameters
        if new_price is not None:
            order.price = self._round_to_tick_size(new_price)
        if new_volume is not None:
            # Can only reduce volume
            if new_volume < order.remaining_volume:
                order.volume = order.filled_volume + new_volume
                order.remaining_volume = new_volume
        
        # Try to match modified order
        trades = self._match_order(order)
        
        # Add back to book if not fully filled
        if order.remaining_volume > 0:
            self._add_order_to_book(order)
        
        return trades
    
    def get_market_depth(self, levels: int = 10) -> Dict[str, List[OrderBookLevel]]:
        """Get market depth information"""
        bids = self._get_depth_side(self._bids, levels, reverse=True)
        asks = self._get_depth_side(self._asks, levels, reverse=False)
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': pd.Timestamp.now(),
            'symbol': self.symbol
        }
    
    def get_order_book_snapshot(self) -> Dict[str, Any]:
        """Get complete order book snapshot"""
        depth = self.get_market_depth()
        
        return {
            'symbol': self.symbol,
            'asset_type': self.asset_type.value,
            'timestamp': pd.Timestamp.now(),
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'last_trade_price': self._last_trade_price,
            'last_trade_time': self._last_trade_time,
            'daily_volume': self._daily_volume,
            'daily_trades': self._daily_trade_count,
            'depth': depth
        }
    
    def get_recent_trades(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades"""
        recent_trades = self._trades[-count:] if count < len(self._trades) else self._trades
        return [trade.to_dict() for trade in recent_trades]
    
    def _match_order(self, order: Order) -> List[Trade]:
        """Match order against the book"""
        # Debug logging
        print(f"DEBUG: _match_order called for order {order.order_id}")
        print(f"  Order type: {order.order_type}")
        print(f"  Order side: {order.side}")
        print(f"  Order price: {order.price}")
        print(f"  Order volume: {order.volume}")
        print(f"  Order remaining_volume: {order.remaining_volume}")
        print(f"  Order is_buy(): {order.is_buy()}")
        
        # Print current book state
        print(f"  Current bids: {list(self._bids.keys())}")
        print(f"  Current asks: {list(self._asks.keys())}")
        
        trades = []
        
        if order.order_type == OrderType.MARKET:
            print("  Routing to _match_market_order")
            trades = self._match_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            print("  Routing to _match_limit_order")
            trades = self._match_limit_order(order)
        else:
            print(f"  WARNING: Unsupported order type: {order.order_type}")
        
        print(f"  Generated {len(trades)} trades")
        for i, trade in enumerate(trades):
            print(f"    Trade {i+1}: {trade.volume} @ {trade.price}")
        
        # Update statistics for each trade
        for trade in trades:
            self._update_trade_statistics(trade)
            self.asset.process_trade(trade)
        
        print(f"DEBUG: _match_order completed for order {order.order_id}")
        return trades
    
    def _match_market_order(self, order: Order) -> List[Trade]:
        """Match a market order"""
        trades = []
        remaining_volume = order.remaining_volume
        
        # Determine which side to match against
        if order.is_buy():
            price_levels = sorted(self._asks.keys())
        else:
            price_levels = sorted(self._bids.keys(), reverse=True)
        
        for price in price_levels:
            if remaining_volume <= 0:
                break
            
            level_orders = self._asks[price] if order.is_buy() else self._bids[price]
            level_trades, remaining_volume = self._match_against_level(
                order, level_orders, price, remaining_volume
            )
            trades.extend(level_trades)
        
        return trades
    
    def _match_limit_order(self, order: Order) -> List[Trade]:
        """Match a limit order"""
        print(f"DEBUG: _match_limit_order called for order {order.order_id}")
        print(f"  Order is_buy(): {order.is_buy()}")
        print(f"  Order price: {order.price}")
        print(f"  Order remaining_volume: {order.remaining_volume}")
        
        trades = []
        remaining_volume = order.remaining_volume
        
        if order.is_buy():
            # Buy orders match against asks at or below limit price
            print(f"  Available asks: {list(self._asks.keys())}")
            matching_prices = [p for p in self._asks.keys() if p <= order.price]
            print(f"  Matching ask prices (at or below {order.price}): {matching_prices}")
            matching_prices.sort()  # Start with best (lowest) ask prices
            print(f"  Sorted matching prices: {matching_prices}")
        else:
            # Sell orders match against bids at or above limit price
            print(f"  Available bids: {list(self._bids.keys())}")
            matching_prices = [p for p in self._bids.keys() if p >= order.price]
            print(f"  Matching bid prices (at or above {order.price}): {matching_prices}")
            matching_prices.sort(reverse=True)  # Start with best (highest) bid prices
            print(f"  Sorted matching prices: {matching_prices}")
        
        for price in matching_prices:
            if remaining_volume <= 0:
                break
            
            level_orders = self._asks[price] if order.is_buy() else self._bids[price]
            print(f"  Matching at price {price} with {len(level_orders)} orders")
            level_trades, remaining_volume = self._match_against_level(
                order, level_orders, price, remaining_volume
            )
            print(f"  Generated {len(level_trades)} trades at price {price}")
            trades.extend(level_trades)
        
        print(f"DEBUG: _match_limit_order completed: {len(trades)} total trades")
        return trades
    
    def _match_against_level(self, incoming_order: Order, level_orders: List[Order], 
                           price: float, remaining_volume: int) -> Tuple[List[Trade], int]:
        """Match incoming order against orders at a price level"""
        print(f"DEBUG: _match_against_level called at price {price}")
        print(f"  Incoming order: {incoming_order.order_id} ({incoming_order.side})")
        print(f"  Level has {len(level_orders)} orders")
        print(f"  Remaining volume to match: {remaining_volume}")
        print(f"  Using price_time_priority: {getattr(self, '_use_price_time_priority', False)}")
        
        trades = []
        orders_to_remove = []
        
        if self._use_price_time_priority:
            # FIFO matching
            for i, existing_order in enumerate(level_orders):
                if remaining_volume <= 0:
                    break
                
                print(f"  Checking order {i}: {existing_order.order_id} ({existing_order.side}, vol: {existing_order.remaining_volume}, active: {existing_order.is_active()})")
                
                if not existing_order.is_active():
                    orders_to_remove.append(existing_order)
                    continue
                
                # Execute trade
                fill_volume = min(remaining_volume, existing_order.remaining_volume)
                print(f"  Creating trade: {fill_volume} @ {price}")
                trade = self._create_trade(incoming_order, existing_order, price, fill_volume)
                trades.append(trade)
                
                # Update orders
                print(f"  Updating orders with fill_volume: {fill_volume}")
                existing_order.partial_fill(fill_volume, price)
                incoming_order.partial_fill(fill_volume, price)
                remaining_volume -= fill_volume
                
                if existing_order.is_filled():
                    print(f"  Order {existing_order.order_id} is now filled, marking for removal")
                    orders_to_remove.append(existing_order)
                    
                print(f"  Remaining volume: {remaining_volume}")
        
        # Remove filled orders from level
        print(f"  Removing {len(orders_to_remove)} filled orders")
        for order in orders_to_remove:
            level_orders.remove(order)
        
        # Clean up empty price level
        if not level_orders:
            print(f"  Price level {price} is now empty, cleaning up")
            if incoming_order.is_buy():
                del self._asks[price]
                if price in self._ask_prices:
                    self._ask_prices.remove(price)
                    heapq.heapify(self._ask_prices)
            else:
                del self._bids[price]
                if -price in self._bid_prices:
                    self._bid_prices.remove(-price)
                    heapq.heapify(self._bid_prices)
        
        print(f"DEBUG: _match_against_level completed: {len(trades)} trades generated")
        return trades, remaining_volume
    
    def _create_trade(self, incoming_order: Order, existing_order: Order, price: float, volume: int) -> Trade:
        """Create a trade between two orders"""
        print(f"DEBUG: _create_trade called")
        print(f"  Incoming order: {incoming_order.order_id} ({incoming_order.side})")
        print(f"  Existing order: {existing_order.order_id} ({existing_order.side})")
        print(f"  Price: {price}, Volume: {volume}")
        
        # Determine buy and sell orders based on their sides
        if incoming_order.is_buy() and existing_order.is_sell():
            buy_order = incoming_order
            sell_order = existing_order
            aggressor_side = incoming_order.side  # Incoming order is aggressor
        elif incoming_order.is_sell() and existing_order.is_buy():
            buy_order = existing_order
            sell_order = incoming_order
            aggressor_side = incoming_order.side  # Incoming order is aggressor
        else:
            # This should not happen in a well-functioning order book
            raise ValueError(f"Invalid trade: both orders have the same side. Incoming: {incoming_order.side}, Existing: {existing_order.side}")
        
        print(f"  Resolved: Buy={buy_order.order_id}, Sell={sell_order.order_id}, Aggressor={aggressor_side}")
        
        trade = Trade(
            trade_id=f"{self.symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')}",
            symbol=self.symbol,
            price=price,
            volume=volume,
            timestamp=pd.Timestamp.now(),
            buy_order_id=buy_order.order_id,
            sell_order_id=sell_order.order_id,
            aggressor_side=aggressor_side
        )
        
        print(f"  Created trade: {trade.trade_id}")
        return trade
    
    def _add_order_to_book(self, order: Order) -> None:
        """Add order to the appropriate side of the book"""
        if order.price is None:
            return  # Market orders shouldn't be added to book
        
        if order.is_buy():
            self._bids[order.price].append(order)
            heapq.heappush(self._bid_prices, -order.price)  # Negative for max heap
        else:
            self._asks[order.price].append(order)
            heapq.heappush(self._ask_prices, order.price)
    
    def _remove_order_from_book(self, order: Order) -> None:
        """Remove order from the book"""
        if order.price is None:
            return
        
        try:
            if order.is_buy():
                self._bids[order.price].remove(order)
                if not self._bids[order.price]:
                    del self._bids[order.price]
                    if -order.price in self._bid_prices:
                        self._bid_prices.remove(-order.price)
                        heapq.heapify(self._bid_prices)
            else:
                self._asks[order.price].remove(order)
                if not self._asks[order.price]:
                    del self._asks[order.price]
                    if order.price in self._ask_prices:
                        self._ask_prices.remove(order.price)
                        heapq.heapify(self._ask_prices)
        except ValueError:
            # Order not found in book
            pass
    
    def _get_depth_side(self, side_dict: Dict[float, List[Order]], 
                       levels: int, reverse: bool) -> List[OrderBookLevel]:
        """Get market depth for one side of the book"""
        prices = sorted(side_dict.keys(), reverse=reverse)[:levels]
        depth = []
        
        for price in prices:
            orders = side_dict[price]
            active_orders = [o for o in orders if o.is_active()]
            
            if active_orders:
                total_volume = sum(o.remaining_volume for o in active_orders)
                depth.append(OrderBookLevel(
                    price=price,
                    volume=total_volume,
                    order_count=len(active_orders),
                    total_size=total_volume
                ))
        
        return depth
    
    def _round_to_tick_size(self, price: float) -> float:
        """Round price to valid tick size"""
        return round(price / self.tick_size) * self.tick_size
    
    def _update_trade_statistics(self, trade: Trade) -> None:
        """Update daily trading statistics"""
        self._trades.append(trade)
        self._daily_volume += trade.volume
        self._daily_trade_count += 1
        self._total_value_traded += trade.trade_value
        self._last_trade_price = trade.price
        self._last_trade_time = trade.timestamp
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics"""
        self._daily_volume = 0
        self._daily_trade_count = 0
        self._total_value_traded = 0.0
        self._trades.clear()
    
    def __str__(self) -> str:
        return f"UnifiedOrderBook({self.symbol}, {self.asset_type.value})"
