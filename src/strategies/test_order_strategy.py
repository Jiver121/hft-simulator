"""
Test Order Generation Strategy for HFT Simulator

This strategy is designed specifically for testing the end-to-end order flow.
It generates at least one order regardless of market conditions to verify
that orders flow through the entire system properly.

Key Features:
- Always generates at least one order per market update
- Uses hardcoded valid price and volume for testing
- Bypasses most risk management for testing purposes
- Tracks orders generated for verification
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.engine.order_types import Order
from src.engine.market_data import BookSnapshot
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.utils.logger import get_logger


class TestOrderStrategy(BaseStrategy):
    """
    Test strategy that always generates orders for end-to-end testing
    
    This strategy is specifically designed to test the order flow through
    the entire system. It will generate at least one order on every market
    update, regardless of market conditions.
    
    Features:
    - Hardcoded valid price ($100.00) and volume (10) for consistent testing
    - Alternates between buy and sell orders
    - Tracks all generated orders for verification
    - Bypasses risk management for testing purposes
    """
    
    def __init__(self, 
                 symbol: str = "TEST",
                 test_price: float = 100.00,
                 test_volume: int = 10,
                 **kwargs):
        """
        Initialize test order strategy
        
        Args:
            symbol: Trading symbol
            test_price: Hardcoded price for test orders
            test_volume: Hardcoded volume for test orders
        """
        super().__init__(
            strategy_name="TestOrder",
            symbol=symbol,
            **kwargs
        )
        
        # Test parameters
        self.test_price = test_price
        self.test_volume = test_volume
        
        # Order tracking
        self.orders_generated = 0
        self.buy_orders_generated = 0
        self.sell_orders_generated = 0
        self.order_history = []
        
        # Alternating order side for variety
        self.next_side_is_buy = True
        
        # Set data mode to backtest for lenient validation
        self.set_data_mode("backtest")
        
        self.logger.info(f"TestOrderStrategy initialized for {symbol} "
                        f"with test_price=${test_price:.2f} and test_volume={test_volume}")
    
    def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> StrategyResult:
        """
        Generate forced test orders on every market update
        
        This method will ALWAYS generate at least one order, regardless of
        market conditions. This ensures we can test the end-to-end order flow.
        
        Args:
            snapshot: Current market snapshot (can be None)
            timestamp: Update timestamp
            
        Returns:
            StrategyResult with at least one test order
        """
        self.last_update_time = timestamp
        self.update_count += 1
        
        # Create result container
        result = StrategyResult(
            timestamp=timestamp,
            processing_time_us=0
        )
        
        # Log the market update
        self.logger.debug(f"TestOrderStrategy processing market update #{self.update_count} at {timestamp}")
        
        # Update market history if snapshot is valid
        if snapshot is not None:
            self._update_market_history(snapshot)
            self.logger.debug(f"Market data: mid_price={getattr(snapshot, 'mid_price', 'N/A')}, "
                            f"best_bid={getattr(snapshot, 'best_bid', 'N/A')}, "
                            f"best_ask={getattr(snapshot, 'best_ask', 'N/A')}")
        
        # FORCE ORDER GENERATION - This is the key part for testing
        # Generate at least one order regardless of conditions
        test_order = self._generate_forced_test_order(timestamp, snapshot)
        
        if test_order:
            result.add_order(test_order, "Forced test order generation")
            self.logger.info(f"Generated forced test order: {test_order.order_id} "
                           f"({test_order.side.value}, {test_order.volume}@${test_order.price:.2f})")
        
        # Optionally generate a second order for more comprehensive testing
        if self.update_count % 3 == 0:  # Every 3rd update
            second_order = self._generate_forced_test_order(timestamp, snapshot, opposite_side=True)
            if second_order:
                result.add_order(second_order, "Additional test order")
                self.logger.debug(f"Generated additional test order: {second_order.order_id}")
        
        # Update statistics
        if result.orders:
            self.orders_generated += len(result.orders)
            for order in result.orders:
                if order.side in [OrderSide.BUY, OrderSide.BID]:
                    self.buy_orders_generated += 1
                else:
                    self.sell_orders_generated += 1
                
                # Track order in history
                self.order_history.append({
                    'timestamp': timestamp,
                    'order_id': order.order_id,
                    'side': order.side.value,
                    'price': order.price,
                    'volume': order.volume,
                    'order_type': order.order_type.value
                })
        
        result.decision_reason = f"Forced test order generation (update #{self.update_count})"
        result.confidence = 1.0  # Always confident in test orders
        
        self.logger.debug(f"TestOrderStrategy generated {len(result.orders)} orders")
        return result
    
    def _generate_forced_test_order(self, 
                                  timestamp: pd.Timestamp, 
                                  snapshot: Optional[BookSnapshot] = None,
                                  opposite_side: bool = False) -> Optional[Order]:
        """
        Generate a forced test order with hardcoded parameters
        
        Args:
            timestamp: Order timestamp
            snapshot: Market snapshot (optional)
            opposite_side: If True, use opposite of the normal side
            
        Returns:
            Test order with hardcoded parameters
        """
        try:
            # Determine order side (alternate for variety)
            if opposite_side:
                side = OrderSide.SELL if self.next_side_is_buy else OrderSide.BUY
            else:
                side = OrderSide.BUY if self.next_side_is_buy else OrderSide.SELL
                self.next_side_is_buy = not self.next_side_is_buy  # Alternate for next time
            
            # Use hardcoded test price and volume
            price = self.test_price
            volume = self.test_volume
            
            # Optionally adjust price slightly based on market data
            if snapshot and snapshot.mid_price:
                # Stay close to market price but use our test parameters
                market_mid = snapshot.mid_price
                if abs(market_mid - self.test_price) > 50:  # If market is very far from test price
                    # Adjust test price to be closer to market
                    price = market_mid if side == OrderSide.BUY else market_mid
                    self.logger.debug(f"Adjusted test price from ${self.test_price} to ${price:.2f} "
                                    f"based on market mid ${market_mid:.2f}")
            
            # Create the order
            order = self.create_order(
                side=side,
                volume=volume,
                price=price,
                order_type=OrderType.LIMIT,
                reason=f"Forced test order #{self.orders_generated + 1}"
            )
            
            # Add test-specific metadata
            order.metadata.update({
                'test_order': True,
                'strategy_update_count': self.update_count,
                'forced_generation': True,
                'original_test_price': self.test_price,
                'original_test_volume': self.test_volume
            })
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to generate forced test order: {e}")
            # Even if order creation fails, try to create a minimal order
            try:
                fallback_order = Order(
                    order_id=f"test_{self.strategy_name}_{self.update_count}_{uuid.uuid4().hex[:8]}",
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=self.test_price,
                    volume=self.test_volume,
                    timestamp=timestamp,
                    source=self.strategy_name
                )
                self.logger.warning(f"Created fallback test order: {fallback_order.order_id}")
                return fallback_order
            except Exception as fallback_error:
                self.logger.error(f"Even fallback order creation failed: {fallback_error}")
                return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = super().get_current_state()
        
        test_info = {
            'strategy_type': 'test_order',
            'test_parameters': {
                'test_price': self.test_price,
                'test_volume': self.test_volume
            },
            'order_statistics': {
                'total_orders_generated': self.orders_generated,
                'buy_orders_generated': self.buy_orders_generated,
                'sell_orders_generated': self.sell_orders_generated,
                'orders_per_update': self.orders_generated / max(self.update_count, 1)
            },
            'recent_orders': self.order_history[-10:] if self.order_history else [],
            'next_side_is_buy': self.next_side_is_buy
        }
        
        base_info.update(test_info)
        return base_info
    
    def reset(self) -> None:
        """Reset strategy state including test-specific counters"""
        super().reset()
        
        # Reset test-specific state
        self.orders_generated = 0
        self.buy_orders_generated = 0
        self.sell_orders_generated = 0
        self.order_history.clear()
        self.next_side_is_buy = True
        
        self.logger.info("TestOrderStrategy state reset")
    
    def set_test_parameters(self, price: float, volume: int) -> None:
        """
        Update test parameters for order generation
        
        Args:
            price: New test price
            volume: New test volume
        """
        self.test_price = price
        self.test_volume = volume
        self.logger.info(f"Updated test parameters: price=${price:.2f}, volume={volume}")
    
    def __str__(self) -> str:
        """String representation of the strategy"""
        return (f"TestOrderStrategy({self.symbol}): "
                f"Generated {self.orders_generated} orders "
                f"(Price=${self.test_price:.2f}, Volume={self.test_volume})")


# Factory function for easy strategy creation
def create_test_order_strategy(symbol: str = "TEST", 
                             test_price: float = 100.00, 
                             test_volume: int = 10,
                             **kwargs) -> TestOrderStrategy:
    """
    Factory function to create a TestOrderStrategy
    
    Args:
        symbol: Trading symbol
        test_price: Hardcoded test price
        test_volume: Hardcoded test volume
        **kwargs: Additional strategy parameters
        
    Returns:
        Configured TestOrderStrategy instance
    """
    return TestOrderStrategy(
        symbol=symbol,
        test_price=test_price,
        test_volume=test_volume,
        **kwargs
    )


if __name__ == "__main__":
    # Simple test of the strategy
    from src.engine.market_data import BookSnapshot, PriceLevel
    
    # Create test strategy
    strategy = TestOrderStrategy("AAPL", test_price=150.00, test_volume=25)
    
    # Create test market data
    timestamp = pd.Timestamp.now()
    snapshot = BookSnapshot(
        symbol="AAPL",
        timestamp=timestamp,
        bids=[PriceLevel(149.95, 100)],
        asks=[PriceLevel(150.05, 100)]
    )
    
    # Test order generation
    print(f"Testing {strategy.strategy_name} strategy...")
    for i in range(5):
        result = strategy.on_market_update(snapshot, timestamp + pd.Timedelta(seconds=i))
        print(f"Update {i+1}: Generated {len(result.orders)} orders")
        for order in result.orders:
            print(f"  - {order.side.value} {order.volume}@${order.price:.2f}")
    
    # Print strategy statistics
    print(f"\nStrategy Statistics:")
    info = strategy.get_strategy_info()
    print(f"Total orders generated: {info['order_statistics']['total_orders_generated']}")
    print(f"Buy orders: {info['order_statistics']['buy_orders_generated']}")
    print(f"Sell orders: {info['order_statistics']['sell_orders_generated']}")
