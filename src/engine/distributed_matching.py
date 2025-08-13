"""
Distributed order matching engine with Redis backend for high-frequency trading.
Provides scalable, fault-tolerant order matching across multiple nodes.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP

try:
    import redis.asyncio as redis
    from redis.exceptions import WatchError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    # Mock Redis for development
    class redis:
        @staticmethod
        def from_url(url): return None
    class WatchError(Exception): pass

from ..events.event_store import EventType, Event
from ..events.cqrs import ExecuteTradeCommand

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal]
    order_type: OrderType
    status: OrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    timestamp: datetime
    strategy_id: Optional[str] = None
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "filled_quantity": str(self.filled_quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "client_id": self.client_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary"""
        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            quantity=Decimal(data["quantity"]),
            price=Decimal(data["price"]) if data["price"] else None,
            order_type=OrderType(data["order_type"]),
            status=OrderStatus(data["status"]),
            filled_quantity=Decimal(data["filled_quantity"]),
            remaining_quantity=Decimal(data["remaining_quantity"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            strategy_id=data.get("strategy_id"),
            client_id=data.get("client_id")
        )


@dataclass
class Trade:
    """Trade execution representation"""
    trade_id: str
    symbol: str
    buy_order_id: str
    sell_order_id: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    buyer_client_id: Optional[str] = None
    seller_client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "buy_order_id": self.buy_order_id,
            "sell_order_id": self.sell_order_id,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "buyer_client_id": self.buyer_client_id,
            "seller_client_id": self.seller_client_id
        }


class PriceLevel:
    """Price level in order book"""
    
    def __init__(self, price: Decimal):
        self.price = price
        self.orders: List[str] = []  # Order IDs at this price level
        self.total_quantity = Decimal('0')
    
    def add_order(self, order_id: str, quantity: Decimal):
        """Add order to price level"""
        self.orders.append(order_id)
        self.total_quantity += quantity
    
    def remove_order(self, order_id: str, quantity: Decimal):
        """Remove order from price level"""
        if order_id in self.orders:
            self.orders.remove(order_id)
            self.total_quantity -= quantity
            if self.total_quantity < 0:
                self.total_quantity = Decimal('0')
    
    def is_empty(self) -> bool:
        """Check if price level is empty"""
        return len(self.orders) == 0 or self.total_quantity == 0


class OrderBook:
    """In-memory order book representation"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[Decimal, PriceLevel] = {}  # Buy orders
        self.asks: Dict[Decimal, PriceLevel] = {}  # Sell orders
        self.orders: Dict[str, Order] = {}  # All orders by ID
        
    def add_order(self, order: Order):
        """Add order to order book"""
        if order.order_type != OrderType.LIMIT:
            return  # Only handle limit orders for now
            
        self.orders[order.order_id] = order
        
        price_levels = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price not in price_levels:
            price_levels[order.price] = PriceLevel(order.price)
        
        price_levels[order.price].add_order(order.order_id, order.remaining_quantity)
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from order book"""
        order = self.orders.get(order_id)
        if not order:
            return None
            
        price_levels = self.bids if order.side == OrderSide.BUY else self.asks
        
        if order.price in price_levels:
            price_level = price_levels[order.price]
            price_level.remove_order(order_id, order.remaining_quantity)
            
            if price_level.is_empty():
                del price_levels[order.price]
        
        del self.orders[order_id]
        return order
    
    def get_best_bid(self) -> Optional[Decimal]:
        """Get best bid price"""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def get_best_ask(self) -> Optional[Decimal]:
        """Get best ask price"""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_spread(self) -> Optional[Decimal]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_top_of_book(self) -> Dict[str, Any]:
        """Get top of book data"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        bid_size = self.bids[best_bid].total_quantity if best_bid else Decimal('0')
        ask_size = self.asks[best_ask].total_quantity if best_ask else Decimal('0')
        
        return {
            "symbol": self.symbol,
            "bid_price": str(best_bid) if best_bid else None,
            "bid_size": str(bid_size),
            "ask_price": str(best_ask) if best_ask else None,
            "ask_size": str(ask_size),
            "spread": str(self.get_spread()) if self.get_spread() else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class DistributedMatchingEngine:
    """Distributed order matching engine using Redis"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", node_id: str = None):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is required for distributed matching engine")
            
        self.redis_url = redis_url
        self.node_id = node_id or f"matching_node_{uuid.uuid4().hex[:8]}"
        self.redis: Optional[redis.Redis] = None
        self.order_books: Dict[str, OrderBook] = {}
        self.is_running = False
        
        # Redis key prefixes
        self.ORDER_PREFIX = "order:"
        self.ORDERBOOK_PREFIX = "orderbook:"
        self.TRADE_PREFIX = "trade:"
        self.SEQUENCE_PREFIX = "seq:"
        self.LOCK_PREFIX = "lock:"
        
        logger.info(f"Initialized matching engine node: {self.node_id}")
    
    async def start(self):
        """Start the matching engine"""
        try:
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            self.is_running = True
            
            logger.info(f"Matching engine {self.node_id} started")
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            
        except Exception as e:
            logger.error(f"Failed to start matching engine: {e}")
            raise
    
    async def stop(self):
        """Stop the matching engine"""
        self.is_running = False
        if self.redis:
            await self.redis.close()
        logger.info(f"Matching engine {self.node_id} stopped")
    
    async def _heartbeat_loop(self):
        """Send heartbeat to indicate node is alive"""
        while self.is_running:
            try:
                await self.redis.setex(
                    f"heartbeat:{self.node_id}",
                    30,  # 30 second TTL
                    int(time.time())
                )
                await asyncio.sleep(10)  # Send heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _acquire_lock(self, resource: str, timeout: int = 10) -> Optional[str]:
        """Acquire distributed lock"""
        lock_key = f"{self.LOCK_PREFIX}{resource}"
        lock_value = f"{self.node_id}:{int(time.time())}"
        
        # Try to acquire lock
        success = await self.redis.set(
            lock_key, 
            lock_value, 
            nx=True,  # Only set if not exists
            ex=timeout  # Expire after timeout
        )
        
        return lock_value if success else None
    
    async def _release_lock(self, resource: str, lock_value: str) -> bool:
        """Release distributed lock"""
        lock_key = f"{self.LOCK_PREFIX}{resource}"
        
        # Use Lua script for atomic check and delete
        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        
        result = await self.redis.eval(lua_script, 1, lock_key, lock_value)
        return bool(result)
    
    async def _get_next_sequence(self, sequence_name: str) -> int:
        """Get next sequence number"""
        seq_key = f"{self.SEQUENCE_PREFIX}{sequence_name}"
        return await self.redis.incr(seq_key)
    
    async def submit_order(self, order: Order) -> bool:
        """Submit order to matching engine"""
        try:
            # Acquire lock for the symbol
            lock_value = await self._acquire_lock(f"symbol:{order.symbol}")
            if not lock_value:
                logger.warning(f"Failed to acquire lock for symbol {order.symbol}")
                return False
            
            try:
                # Store order in Redis
                order_key = f"{self.ORDER_PREFIX}{order.order_id}"
                await self.redis.hset(order_key, mapping=order.to_dict())
                
                # Load order book
                order_book = await self._load_order_book(order.symbol)
                
                # Process order (match if possible)
                trades = await self._process_order(order_book, order)
                
                # Save updated order book
                await self._save_order_book(order_book)
                
                # Store trades
                for trade in trades:
                    trade_key = f"{self.TRADE_PREFIX}{trade.trade_id}"
                    await self.redis.hset(trade_key, mapping=trade.to_dict())
                
                logger.info(f"Order {order.order_id} processed, {len(trades)} trades generated")
                return True
                
            finally:
                await self._release_lock(f"symbol:{order.symbol}", lock_value)
                
        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        try:
            # Get order details
            order_key = f"{self.ORDER_PREFIX}{order_id}"
            order_data = await self.redis.hgetall(order_key)
            
            if not order_data:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = Order.from_dict({k.decode(): v.decode() for k, v in order_data.items()})
            
            # Acquire lock for the symbol
            lock_value = await self._acquire_lock(f"symbol:{order.symbol}")
            if not lock_value:
                logger.warning(f"Failed to acquire lock for symbol {order.symbol}")
                return False
            
            try:
                # Load order book and remove order
                order_book = await self._load_order_book(order.symbol)
                removed_order = order_book.remove_order(order_id)
                
                if removed_order:
                    # Update order status
                    removed_order.status = OrderStatus.CANCELLED
                    await self.redis.hset(order_key, mapping=removed_order.to_dict())
                    
                    # Save updated order book
                    await self._save_order_book(order_book)
                    
                    logger.info(f"Order {order_id} cancelled")
                    return True
                else:
                    logger.warning(f"Order {order_id} not found in order book")
                    return False
                    
            finally:
                await self._release_lock(f"symbol:{order.symbol}", lock_value)
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _load_order_book(self, symbol: str) -> OrderBook:
        """Load order book from Redis or create new one"""
        if symbol in self.order_books:
            return self.order_books[symbol]
        
        order_book = OrderBook(symbol)
        
        # Load orders from Redis
        orderbook_key = f"{self.ORDERBOOK_PREFIX}{symbol}"
        order_ids = await self.redis.smembers(orderbook_key)
        
        for order_id_bytes in order_ids:
            order_id = order_id_bytes.decode()
            order_key = f"{self.ORDER_PREFIX}{order_id}"
            order_data = await self.redis.hgetall(order_key)
            
            if order_data:
                order_dict = {k.decode(): v.decode() for k, v in order_data.items()}
                order = Order.from_dict(order_dict)
                
                if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    order_book.add_order(order)
        
        self.order_books[symbol] = order_book
        return order_book
    
    async def _save_order_book(self, order_book: OrderBook):
        """Save order book to Redis"""
        orderbook_key = f"{self.ORDERBOOK_PREFIX}{order_book.symbol}"
        
        # Clear existing order book
        await self.redis.delete(orderbook_key)
        
        # Add all order IDs
        if order_book.orders:
            await self.redis.sadd(orderbook_key, *order_book.orders.keys())
        
        # Cache the order book
        self.order_books[order_book.symbol] = order_book
    
    async def _process_order(self, order_book: OrderBook, new_order: Order) -> List[Trade]:
        """Process incoming order and generate trades"""
        trades = []
        
        if new_order.order_type == OrderType.MARKET:
            trades.extend(await self._process_market_order(order_book, new_order))
        elif new_order.order_type == OrderType.LIMIT:
            trades.extend(await self._process_limit_order(order_book, new_order))
        
        return trades
    
    async def _process_market_order(self, order_book: OrderBook, market_order: Order) -> List[Trade]:
        """Process market order"""
        trades = []
        remaining_qty = market_order.remaining_quantity
        
        # Determine opposite side
        opposite_side_levels = order_book.asks if market_order.side == OrderSide.BUY else order_book.bids
        
        # Sort price levels (best prices first)
        sorted_prices = sorted(opposite_side_levels.keys()) if market_order.side == OrderSide.BUY else sorted(opposite_side_levels.keys(), reverse=True)
        
        for price in sorted_prices:
            if remaining_qty <= 0:
                break
                
            price_level = opposite_side_levels[price]
            
            # Match with orders at this price level
            for order_id in price_level.orders.copy():
                if remaining_qty <= 0:
                    break
                
                resting_order = order_book.orders.get(order_id)
                if not resting_order:
                    continue
                
                # Calculate trade quantity
                trade_qty = min(remaining_qty, resting_order.remaining_quantity)
                
                # Create trade
                trade = await self._create_trade(
                    market_order, resting_order, trade_qty, price, order_book.symbol
                )
                trades.append(trade)
                
                # Update orders
                await self._update_order_after_trade(market_order, trade_qty)
                await self._update_order_after_trade(resting_order, trade_qty)
                
                # Update order book
                if resting_order.remaining_quantity <= 0:
                    order_book.remove_order(order_id)
                
                remaining_qty -= trade_qty
        
        # Update market order status
        if market_order.remaining_quantity <= 0:
            market_order.status = OrderStatus.FILLED
        elif market_order.filled_quantity > 0:
            market_order.status = OrderStatus.PARTIALLY_FILLED
        
        return trades
    
    async def _process_limit_order(self, order_book: OrderBook, limit_order: Order) -> List[Trade]:
        """Process limit order"""
        trades = []
        
        # First try to match with existing orders
        opposite_side_levels = order_book.asks if limit_order.side == OrderSide.BUY else order_book.bids
        
        # Get matchable prices
        if limit_order.side == OrderSide.BUY:
            matchable_prices = [p for p in opposite_side_levels.keys() if p <= limit_order.price]
            matchable_prices.sort()  # Best prices first for buyer
        else:
            matchable_prices = [p for p in opposite_side_levels.keys() if p >= limit_order.price]
            matchable_prices.sort(reverse=True)  # Best prices first for seller
        
        remaining_qty = limit_order.remaining_quantity
        
        for price in matchable_prices:
            if remaining_qty <= 0:
                break
                
            price_level = opposite_side_levels[price]
            
            for order_id in price_level.orders.copy():
                if remaining_qty <= 0:
                    break
                
                resting_order = order_book.orders.get(order_id)
                if not resting_order:
                    continue
                
                # Calculate trade quantity
                trade_qty = min(remaining_qty, resting_order.remaining_quantity)
                
                # Create trade at the resting order's price (price improvement)
                trade = await self._create_trade(
                    limit_order, resting_order, trade_qty, resting_order.price, order_book.symbol
                )
                trades.append(trade)
                
                # Update orders
                await self._update_order_after_trade(limit_order, trade_qty)
                await self._update_order_after_trade(resting_order, trade_qty)
                
                # Remove filled order from book
                if resting_order.remaining_quantity <= 0:
                    order_book.remove_order(order_id)
                
                remaining_qty -= trade_qty
        
        # Add remaining quantity to order book if not fully filled
        if limit_order.remaining_quantity > 0:
            order_book.add_order(limit_order)
            if limit_order.filled_quantity > 0:
                limit_order.status = OrderStatus.PARTIALLY_FILLED
        else:
            limit_order.status = OrderStatus.FILLED
        
        return trades
    
    async def _create_trade(self, aggressive_order: Order, resting_order: Order, 
                           quantity: Decimal, price: Decimal, symbol: str) -> Trade:
        """Create trade object"""
        trade_id = f"trade_{await self._get_next_sequence('trade')}"
        
        # Determine buy/sell orders
        if aggressive_order.side == OrderSide.BUY:
            buy_order_id = aggressive_order.order_id
            sell_order_id = resting_order.order_id
            buyer_client_id = aggressive_order.client_id
            seller_client_id = resting_order.client_id
        else:
            buy_order_id = resting_order.order_id
            sell_order_id = aggressive_order.order_id
            buyer_client_id = resting_order.client_id
            seller_client_id = aggressive_order.client_id
        
        return Trade(
            trade_id=trade_id,
            symbol=symbol,
            buy_order_id=buy_order_id,
            sell_order_id=sell_order_id,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(timezone.utc),
            buyer_client_id=buyer_client_id,
            seller_client_id=seller_client_id
        )
    
    async def _update_order_after_trade(self, order: Order, trade_quantity: Decimal):
        """Update order after trade execution"""
        order.filled_quantity += trade_quantity
        order.remaining_quantity -= trade_quantity
        
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
        elif order.filled_quantity > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update order in Redis
        order_key = f"{self.ORDER_PREFIX}{order.order_id}"
        await self.redis.hset(order_key, mapping=order.to_dict())
    
    async def get_order_book_snapshot(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """Get order book snapshot"""
        try:
            order_book = await self._load_order_book(symbol)
            
            # Get top levels
            bid_levels = []
            ask_levels = []
            
            # Sort bids (highest first)
            sorted_bids = sorted(order_book.bids.items(), key=lambda x: x[0], reverse=True)
            for price, price_level in sorted_bids[:levels]:
                bid_levels.append({
                    "price": str(price),
                    "size": str(price_level.total_quantity),
                    "orders": len(price_level.orders)
                })
            
            # Sort asks (lowest first)
            sorted_asks = sorted(order_book.asks.items(), key=lambda x: x[0])
            for price, price_level in sorted_asks[:levels]:
                ask_levels.append({
                    "price": str(price),
                    "size": str(price_level.total_quantity),
                    "orders": len(price_level.orders)
                })
            
            return {
                "symbol": symbol,
                "bids": bid_levels,
                "asks": ask_levels,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting order book snapshot for {symbol}: {e}")
            return {"symbol": symbol, "bids": [], "asks": [], "error": str(e)}
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        try:
            order_key = f"{self.ORDER_PREFIX}{order_id}"
            order_data = await self.redis.hgetall(order_key)
            
            if not order_data:
                return None
            
            return {k.decode(): v.decode() for k, v in order_data.items()}
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for symbol"""
        try:
            # This is a simplified implementation
            # In production, you'd use Redis streams or sorted sets for efficient trade storage
            trades = []
            
            # Get all trade keys (in production, use better indexing)
            pattern = f"{self.TRADE_PREFIX}*"
            trade_keys = await self.redis.keys(pattern)
            
            for trade_key in trade_keys[-limit:]:  # Get most recent
                trade_data = await self.redis.hgetall(trade_key)
                if trade_data:
                    trade_dict = {k.decode(): v.decode() for k, v in trade_data.items()}
                    if trade_dict.get("symbol") == symbol:
                        trades.append(trade_dict)
            
            # Sort by timestamp (most recent first)
            trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return trades[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {e}")
            return []


# Example usage and testing
async def main():
    """Example usage of distributed matching engine"""
    try:
        # Create matching engine
        engine = DistributedMatchingEngine()
        await engine.start()
        
        # Create test orders
        buy_order = Order(
            order_id="buy_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('45000.00'),
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            filled_quantity=Decimal('0'),
            remaining_quantity=Decimal('1.0'),
            timestamp=datetime.now(timezone.utc),
            client_id="client_001"
        )
        
        sell_order = Order(
            order_id="sell_001",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=Decimal('0.5'),
            price=Decimal('45000.00'),
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            filled_quantity=Decimal('0'),
            remaining_quantity=Decimal('0.5'),
            timestamp=datetime.now(timezone.utc),
            client_id="client_002"
        )
        
        # Submit orders
        print("Submitting buy order...")
        await engine.submit_order(buy_order)
        
        print("Submitting sell order (should match)...")
        await engine.submit_order(sell_order)
        
        # Get order book snapshot
        print("\nOrder book snapshot:")
        snapshot = await engine.get_order_book_snapshot("BTCUSDT")
        print(json.dumps(snapshot, indent=2))
        
        # Get recent trades
        print("\nRecent trades:")
        trades = await engine.get_recent_trades("BTCUSDT")
        for trade in trades:
            print(f"Trade: {trade['quantity']} at {trade['price']}")
        
        await engine.stop()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
