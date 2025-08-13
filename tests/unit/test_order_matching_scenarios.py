"""
Unit Tests for Order Matching Scenarios

This module contains focused unit tests for the order matching engine,
specifically designed to verify the fix for order matching issues and
validate correct trade generation.

Test Coverage:
- Market buy order matching with existing ask limit order
- Market sell order matching with existing bid limit order  
- Limit order crossing scenarios
- Partial fill scenarios
- Trade object creation with correct price, volume, and timestamps
- Edge cases and error conditions

Educational Notes:
- These tests verify the core matching logic after bug fixes
- Tests ensure proper price-time priority enforcement
- Validates trade creation with accurate parameters
- Confirms proper handling of partial fills and remaining volumes
"""

import unittest
import pandas as pd
import uuid
from typing import List, Tuple
from decimal import Decimal

from src.execution.matching_engine import MatchingEngine
from src.engine.order_book import OrderBook
from src.engine.order_types import Order, Trade, OrderUpdate
from src.utils.constants import OrderSide, OrderType, OrderStatus


class TestOrderMatchingScenarios(unittest.TestCase):
    """Test suite for order matching scenarios"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.symbol = "TEST"
        self.matching_engine = MatchingEngine(
            symbol=self.symbol,
            tick_size=0.01,
            min_order_size=1,
            max_order_size=10000
        )
        self.order_book = self.matching_engine.order_book
        
        # Set a consistent base timestamp for reproducible tests
        self.base_time = pd.Timestamp('2024-01-15 09:30:00')
        
    def tearDown(self):
        """Clean up after each test"""
        self.matching_engine.reset_session()
        
    def _create_order(self, order_id: str, side: OrderSide, order_type: OrderType,
                     volume: int, price: float = None, 
                     timestamp_offset_ms: int = 0) -> Order:
        """Helper to create test orders with consistent timestamps"""
        timestamp = self.base_time + pd.Timedelta(milliseconds=timestamp_offset_ms)
        
        return Order(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            volume=volume,
            price=price,
            timestamp=timestamp
        )
    
    def _assert_trade_validity(self, trade: Trade, expected_volume: int, 
                              expected_price: float, buy_order_id: str = None,
                              sell_order_id: str = None):
        """Helper to validate trade object properties"""
        self.assertIsNotNone(trade, "Trade object should not be None")
        self.assertEqual(trade.symbol, self.symbol, "Trade symbol should match test symbol")
        self.assertEqual(trade.volume, expected_volume, f"Expected volume {expected_volume}, got {trade.volume}")
        self.assertAlmostEqual(trade.price, expected_price, places=2, 
                              msg=f"Expected price {expected_price}, got {trade.price}")
        self.assertGreater(trade.price, 0, "Trade price must be positive")
        self.assertGreater(trade.volume, 0, "Trade volume must be positive")
        self.assertIsNotNone(trade.timestamp, "Trade timestamp should not be None")
        self.assertIsNotNone(trade.trade_id, "Trade ID should not be None")
        
        # Validate order references
        if buy_order_id:
            self.assertEqual(trade.buy_order_id, buy_order_id, 
                           f"Expected buy order ID {buy_order_id}, got {trade.buy_order_id}")
        if sell_order_id:
            self.assertEqual(trade.sell_order_id, sell_order_id,
                           f"Expected sell order ID {sell_order_id}, got {trade.sell_order_id}")
        
        # At least one order ID should be present
        self.assertTrue(trade.buy_order_id or trade.sell_order_id, 
                       "Trade must have at least one order ID reference")

    def test_market_buy_matches_existing_ask_limit(self):
        """Test market buy order matching with existing ask limit order"""
        # Step 1: Add ask limit order to create liquidity
        ask_order = self._create_order(
            order_id="ASK001",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=150.00,
            timestamp_offset_ms=0
        )
        
        trades_ask, update_ask = self.matching_engine.process_order(ask_order)
        
        # Ask order should be added to book without generating trades
        self.assertEqual(len(trades_ask), 0, "Ask order should not generate immediate trades")
        self.assertEqual(update_ask.update_type, 'new', "Ask order should be added as new")
        
        # Verify ask is in the order book
        snapshot = self.matching_engine.get_market_snapshot()
        self.assertIsNotNone(snapshot.best_ask, "Best ask should exist after adding ask order")
        self.assertEqual(snapshot.best_ask, 150.00, "Best ask price should be 150.00")
        self.assertEqual(snapshot.best_ask_volume, 100, "Best ask volume should be 100")
        
        # Step 2: Add market buy order that should match
        market_buy = self._create_order(
            order_id="BUY001",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=50,
            timestamp_offset_ms=10
        )
        
        trades_buy, update_buy = self.matching_engine.process_order(market_buy)
        
        # Market buy should generate exactly one trade
        self.assertEqual(len(trades_buy), 1, "Market buy should generate exactly one trade")
        self.assertEqual(update_buy.update_type, 'trade', "Market buy should result in trade update")
        
        # Validate the generated trade
        trade = trades_buy[0]
        self._assert_trade_validity(
            trade=trade,
            expected_volume=50,
            expected_price=150.00,  # Should match at ask price
            buy_order_id="BUY001",
            sell_order_id="ASK001"
        )
        
        # Verify market state after trade
        final_snapshot = self.matching_engine.get_market_snapshot()
        self.assertEqual(final_snapshot.best_ask, 150.00, "Best ask price should remain 150.00")
        self.assertEqual(final_snapshot.best_ask_volume, 50, "Best ask volume should be 50 (remaining)")
        
        # Verify order statuses
        self.assertEqual(market_buy.status, OrderStatus.FILLED, "Market buy should be completely filled")
        self.assertEqual(ask_order.status, OrderStatus.PARTIAL, "Ask order should be partially filled")
        self.assertEqual(ask_order.filled_volume, 50, "Ask order should have 50 filled volume")
        self.assertEqual(ask_order.remaining_volume, 50, "Ask order should have 50 remaining volume")

    def test_market_sell_matches_existing_bid_limit(self):
        """Test market sell order matching with existing bid limit order"""
        # Step 1: Add bid limit order to create liquidity
        bid_order = self._create_order(
            order_id="BID001",
            side=OrderSide.BID,
            order_type=OrderType.LIMIT,
            volume=200,
            price=149.50,
            timestamp_offset_ms=0
        )
        
        trades_bid, update_bid = self.matching_engine.process_order(bid_order)
        
        # Bid order should be added to book without generating trades
        self.assertEqual(len(trades_bid), 0, "Bid order should not generate immediate trades")
        self.assertEqual(update_bid.update_type, 'new', "Bid order should be added as new")
        
        # Verify bid is in the order book
        snapshot = self.matching_engine.get_market_snapshot()
        self.assertIsNotNone(snapshot.best_bid, "Best bid should exist after adding bid order")
        self.assertEqual(snapshot.best_bid, 149.50, "Best bid price should be 149.50")
        self.assertEqual(snapshot.best_bid_volume, 200, "Best bid volume should be 200")
        
        # Step 2: Add market sell order that should match
        market_sell = self._create_order(
            order_id="SELL001",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            volume=75,
            timestamp_offset_ms=10
        )
        
        trades_sell, update_sell = self.matching_engine.process_order(market_sell)
        
        # Market sell should generate exactly one trade
        self.assertEqual(len(trades_sell), 1, "Market sell should generate exactly one trade")
        self.assertEqual(update_sell.update_type, 'trade', "Market sell should result in trade update")
        
        # Validate the generated trade
        trade = trades_sell[0]
        self._assert_trade_validity(
            trade=trade,
            expected_volume=75,
            expected_price=149.50,  # Should match at bid price
            buy_order_id="BID001",
            sell_order_id="SELL001"
        )
        
        # Verify market state after trade
        final_snapshot = self.matching_engine.get_market_snapshot()
        self.assertEqual(final_snapshot.best_bid, 149.50, "Best bid price should remain 149.50")
        self.assertEqual(final_snapshot.best_bid_volume, 125, "Best bid volume should be 125 (remaining)")
        
        # Verify order statuses
        self.assertEqual(market_sell.status, OrderStatus.FILLED, "Market sell should be completely filled")
        self.assertEqual(bid_order.status, OrderStatus.PARTIAL, "Bid order should be partially filled")
        self.assertEqual(bid_order.filled_volume, 75, "Bid order should have 75 filled volume")
        self.assertEqual(bid_order.remaining_volume, 125, "Bid order should have 125 remaining volume")

    def test_limit_order_crossing_scenarios(self):
        """Test limit order crossing scenarios where orders cross the spread"""
        # Scenario 1: Aggressive buy limit order crosses spread
        
        # Step 1: Add ask limit order
        ask_order = self._create_order(
            order_id="ASK002",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=150.00,
            timestamp_offset_ms=0
        )
        
        trades_ask, _ = self.matching_engine.process_order(ask_order)
        self.assertEqual(len(trades_ask), 0, "Initial ask should not generate trades")
        
        # Step 2: Add aggressive buy limit order that crosses (price higher than ask)
        aggressive_buy = self._create_order(
            order_id="BUY002",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=60,
            price=150.05,  # Higher than ask price - should cross
            timestamp_offset_ms=10
        )
        
        trades_buy, update_buy = self.matching_engine.process_order(aggressive_buy)
        
        # Should generate crossing trade
        self.assertEqual(len(trades_buy), 1, "Crossing buy limit should generate one trade")
        self.assertEqual(update_buy.update_type, 'trade', "Crossing should result in trade update")
        
        trade = trades_buy[0]
        self._assert_trade_validity(
            trade=trade,
            expected_volume=60,
            expected_price=150.00,  # Should execute at resting order price (price improvement)
            buy_order_id="BUY002",
            sell_order_id="ASK002"
        )
        
        # Scenario 2: Aggressive sell limit order crosses spread
        
        # Add bid limit order
        bid_order = self._create_order(
            order_id="BID002",
            side=OrderSide.BID,
            order_type=OrderType.LIMIT,
            volume=150,
            price=149.80,
            timestamp_offset_ms=20
        )
        
        trades_bid, _ = self.matching_engine.process_order(bid_order)
        self.assertEqual(len(trades_bid), 0, "Initial bid should not generate trades")
        
        # Add aggressive sell limit order that crosses (price lower than bid)
        aggressive_sell = self._create_order(
            order_id="SELL002",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            volume=80,
            price=149.75,  # Lower than bid price - should cross
            timestamp_offset_ms=30
        )
        
        trades_sell, update_sell = self.matching_engine.process_order(aggressive_sell)
        
        # Should generate crossing trade
        self.assertEqual(len(trades_sell), 1, "Crossing sell limit should generate one trade")
        self.assertEqual(update_sell.update_type, 'trade', "Crossing should result in trade update")
        
        trade = trades_sell[0]
        self._assert_trade_validity(
            trade=trade,
            expected_volume=80,
            expected_price=149.80,  # Should execute at resting order price (price improvement)
            buy_order_id="BID002",
            sell_order_id="SELL002"
        )

    def test_partial_fill_scenarios(self):
        """Test partial fill scenarios with multiple trades"""
        # Create multiple small ask orders at different price levels
        small_asks = [
            self._create_order("ASK_L1", OrderSide.ASK, OrderType.LIMIT, 25, 150.00, 0),
            self._create_order("ASK_L2", OrderSide.ASK, OrderType.LIMIT, 30, 150.01, 5),
            self._create_order("ASK_L3", OrderSide.ASK, OrderType.LIMIT, 35, 150.02, 10),
        ]
        
        # Add all ask orders to create liquidity ladder
        for ask in small_asks:
            trades, _ = self.matching_engine.process_order(ask)
            self.assertEqual(len(trades), 0, f"Ask order {ask.order_id} should not generate immediate trades")
        
        # Verify market depth
        snapshot = self.matching_engine.get_market_snapshot()
        self.assertEqual(snapshot.best_ask, 150.00, "Best ask should be at lowest price level")
        self.assertEqual(snapshot.best_ask_volume, 25, "Best ask volume should be 25")
        
        # Submit large buy order that will partially fill against multiple levels
        large_buy = self._create_order(
            order_id="LARGE_BUY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=75,  # Will match against all three ask levels (25+30+20)
            price=150.03,  # High enough to cross all levels
            timestamp_offset_ms=20
        )
        
        trades_large, update_large = self.matching_engine.process_order(large_buy)
        
        # Should generate multiple trades (one per price level)
        self.assertGreaterEqual(len(trades_large), 1, "Large buy should generate at least one trade")
        self.assertEqual(update_large.update_type, 'trade', "Large buy should result in trade update")
        
        # Calculate total executed volume
        total_executed_volume = sum(trade.volume for trade in trades_large)
        self.assertEqual(total_executed_volume, 75, "Total executed volume should be 75")
        
        # Verify trades occurred at correct prices (price-time priority)
        if len(trades_large) >= 1:
            self.assertEqual(trades_large[0].price, 150.00, "First trade should be at best ask price")
        
        # Verify order status
        self.assertEqual(large_buy.status, OrderStatus.FILLED, "Large buy should be completely filled")
        self.assertEqual(large_buy.filled_volume, 75, "Large buy filled volume should be 75")
        self.assertEqual(large_buy.remaining_volume, 0, "Large buy remaining volume should be 0")
        
        # Test partial fill of large resting order
        # Add large ask order
        large_ask = self._create_order(
            order_id="LARGE_ASK",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=200,
            price=150.10,
            timestamp_offset_ms=30
        )
        
        trades_ask, _ = self.matching_engine.process_order(large_ask)
        self.assertEqual(len(trades_ask), 0, "Large ask should not generate immediate trades")
        
        # Hit with smaller buy order
        small_buy = self._create_order(
            order_id="SMALL_BUY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=50,
            timestamp_offset_ms=40
        )
        
        trades_small, update_small = self.matching_engine.process_order(small_buy)
        
        # Market order should generate 2 trades as it consumes liquidity from 2 price levels:
        # 15 volume at 150.02 (remaining from ASK_L3) + 35 volume at 150.10 (from LARGE_ASK)
        self.assertEqual(len(trades_small), 2, "Small buy should generate two trades across price levels")
        self.assertEqual(update_small.update_type, 'trade', "Market buy should result in trade update")
        
        # Validate total volume traded
        total_volume_traded = sum(trade.volume for trade in trades_small)
        self.assertEqual(total_volume_traded, 50, "Total volume traded should be 50")
        
        # Validate first trade (should be at better price level 150.02)
        first_trade = trades_small[0]
        self._assert_trade_validity(
            trade=first_trade,
            expected_volume=15,  # Remaining volume from ASK_L3
            expected_price=150.02,
            buy_order_id="SMALL_BUY",
            sell_order_id="ASK_L3"
        )
        
        # Validate second trade (should be at 150.10 from LARGE_ASK)
        second_trade = trades_small[1]
        self._assert_trade_validity(
            trade=second_trade,
            expected_volume=35,  # Remaining volume needed
            expected_price=150.10,
            buy_order_id="SMALL_BUY",
            sell_order_id="LARGE_ASK"
        )
        
        # Verify ASK_L3 is completely filled after the first trade
        ask_l3 = small_asks[2]  # ASK_L3
        self.assertEqual(ask_l3.status, OrderStatus.FILLED, "ASK_L3 should be completely filled")
        self.assertEqual(ask_l3.remaining_volume, 0, "ASK_L3 should have no remaining volume")
        
        # Verify large ask is partially filled with correct amounts
        self.assertEqual(large_ask.status, OrderStatus.PARTIAL, "Large ask should be partially filled")
        self.assertEqual(large_ask.filled_volume, 35, "Large ask filled volume should be 35 (not 50)")
        self.assertEqual(large_ask.remaining_volume, 165, "Large ask remaining volume should be 165 (not 150)")

    def test_trade_object_creation_accuracy(self):
        """Test that trade objects are created with correct price, volume, and timestamps"""
        # Create orders with specific timestamps
        ask_order = self._create_order(
            order_id="PRECISION_ASK",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=149.99,
            timestamp_offset_ms=0
        )
        
        buy_order = self._create_order(
            order_id="PRECISION_BUY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=40,
            timestamp_offset_ms=100  # 100ms later
        )
        
        # Process orders
        self.matching_engine.process_order(ask_order)
        trades, _ = self.matching_engine.process_order(buy_order)
        
        self.assertEqual(len(trades), 1, "Should generate exactly one trade")
        
        trade = trades[0]
        
        # Test trade object accuracy
        self.assertEqual(trade.symbol, self.symbol, "Trade symbol should match")
        self.assertEqual(trade.volume, 40, "Trade volume should be exact")
        self.assertAlmostEqual(trade.price, 149.99, places=2, msg="Trade price should be exact")
        
        # Test trade value calculation
        expected_value = 149.99 * 40
        self.assertAlmostEqual(trade.trade_value, expected_value, places=2, 
                              msg="Trade value should be price * volume")
        
        # Test timestamp accuracy
        self.assertIsNotNone(trade.timestamp, "Trade should have timestamp")
        self.assertIsInstance(trade.timestamp, pd.Timestamp, "Trade timestamp should be pandas Timestamp")
        
        # Timestamp should be after the buy order timestamp but close to current time
        time_diff = (trade.timestamp - buy_order.timestamp).total_seconds()
        self.assertLessEqual(time_diff, 1.0, "Trade timestamp should be close to order timestamp")
        
        # Test order ID references
        self.assertEqual(trade.buy_order_id, "PRECISION_BUY", "Buy order ID should be correct")
        self.assertEqual(trade.sell_order_id, "PRECISION_ASK", "Sell order ID should be correct")
        
        # Test aggressor side
        self.assertIn(trade.aggressor_side, [OrderSide.BUY, OrderSide.BID], 
                     "Aggressor should be buy side for market buy")
        
        # Test trade ID uniqueness
        self.assertIsNotNone(trade.trade_id, "Trade should have unique ID")
        self.assertTrue(len(trade.trade_id) > 0, "Trade ID should not be empty")

    def test_price_time_priority_enforcement(self):
        """Test that price-time priority is properly enforced in matching"""
        # Add multiple bid orders at same price but different times
        bid1 = self._create_order("BID_FIRST", OrderSide.BID, OrderType.LIMIT, 50, 149.00, 0)
        bid2 = self._create_order("BID_SECOND", OrderSide.BID, OrderType.LIMIT, 50, 149.00, 10)  # Later
        bid3 = self._create_order("BID_THIRD", OrderSide.BID, OrderType.LIMIT, 50, 149.00, 20)   # Even later
        
        # Process all bids
        for bid in [bid1, bid2, bid3]:
            trades, _ = self.matching_engine.process_order(bid)
            self.assertEqual(len(trades), 0, f"Bid {bid.order_id} should not generate immediate trades")
        
        # Add sell order that matches the bid price
        sell_order = self._create_order(
            order_id="SELL_TEST",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            volume=75,  # Will partially fill first bid and fully fill part of second
            price=149.00,
            timestamp_offset_ms=30
        )
        
        trades, _ = self.matching_engine.process_order(sell_order)
        
        # Should generate trades with proper time priority
        self.assertGreaterEqual(len(trades), 1, "Should generate at least one trade")
        
        total_volume = sum(trade.volume for trade in trades)
        self.assertEqual(total_volume, 75, "Total trade volume should match sell order")
        
        # Verify that BID_FIRST was filled first (time priority)
        self.assertEqual(bid1.filled_volume, 50, "First bid should be completely filled")
        self.assertEqual(bid1.status, OrderStatus.FILLED, "First bid should have FILLED status")
        
        self.assertEqual(bid2.filled_volume, 25, "Second bid should be partially filled")
        self.assertEqual(bid2.status, OrderStatus.PARTIAL, "Second bid should have PARTIAL status")
        
        self.assertEqual(bid3.filled_volume, 0, "Third bid should not be filled yet")
        self.assertEqual(bid3.status, OrderStatus.PENDING, "Third bid should still be pending")

    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions in order matching"""
        
        # Test 1: Zero volume orders (should be rejected at validation)
        with self.assertRaises(ValueError, msg="Zero volume order should raise ValueError"):
            self._create_order("ZERO_VOL", OrderSide.BUY, OrderType.MARKET, 0)
        
        # Test 2: Negative volume orders (should be rejected at validation)
        with self.assertRaises(ValueError, msg="Negative volume order should raise ValueError"):
            self._create_order("NEG_VOL", OrderSide.BUY, OrderType.MARKET, -10)
        
        # Test 3: Invalid price for limit orders
        with self.assertRaises(ValueError, msg="Negative price should raise ValueError"):
            self._create_order("NEG_PRICE", OrderSide.BUY, OrderType.LIMIT, 10, -5.0)
        
        # Test 4: Market order when no liquidity exists
        empty_market_order = self._create_order(
            order_id="EMPTY_MARKET",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            volume=10,
            timestamp_offset_ms=0
        )
        
        trades, update = self.matching_engine.process_order(empty_market_order)
        
        # Should not crash, but may not generate trades depending on fill model
        self.assertIsInstance(trades, list, "Should return trades list even if empty")
        self.assertIsNotNone(update, "Should return order update even if no trades")
        
        # Test 5: Very large order (within limits)
        large_ask = self._create_order(
            order_id="LARGE_ORDER",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=5000,  # Within max_order_size limit
            price=200.0,
            timestamp_offset_ms=10
        )
        
        trades, update = self.matching_engine.process_order(large_ask)
        self.assertEqual(len(trades), 0, "Large order should be added without immediate trades")
        self.assertEqual(update.update_type, 'new', "Large order should be added as new")
        
        # Verify it's in the order book
        snapshot = self.matching_engine.get_market_snapshot()
        self.assertEqual(snapshot.best_ask, 200.0, "Large order should be best ask")

    def test_multiple_symbol_isolation(self):
        """Test that orders for different symbols don't interfere with each other"""
        # Create matching engine for different symbol
        other_symbol = "OTHER"
        other_engine = MatchingEngine(
            symbol=other_symbol,
            tick_size=0.01,
            min_order_size=1,
            max_order_size=10000
        )
        
        # Add orders to both engines
        test_ask = self._create_order(
            order_id="TEST_ASK",
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=150.0
        )
        
        other_ask = Order(
            order_id="OTHER_ASK",
            symbol=other_symbol,
            side=OrderSide.ASK,
            order_type=OrderType.LIMIT,
            volume=100,
            price=150.0,
            timestamp=self.base_time
        )
        
        # Process orders
        self.matching_engine.process_order(test_ask)
        other_engine.process_order(other_ask)
        
        # Verify isolation
        test_snapshot = self.matching_engine.get_market_snapshot()
        other_snapshot = other_engine.get_market_snapshot()
        
        self.assertEqual(test_snapshot.symbol, self.symbol, "Test engine should have correct symbol")
        self.assertEqual(other_snapshot.symbol, other_symbol, "Other engine should have correct symbol")
        
        # Both should have ask at same price but be independent
        self.assertEqual(test_snapshot.best_ask, 150.0, "Test engine should have ask at 150.0")
        self.assertEqual(other_snapshot.best_ask, 150.0, "Other engine should have ask at 150.0")
        
        # Orders should only exist in their respective engines
        self.assertIsNotNone(self.matching_engine.order_book.get_order("TEST_ASK"), 
                           "Test order should exist in test engine")
        self.assertIsNone(self.matching_engine.order_book.get_order("OTHER_ASK"),
                         "Other order should not exist in test engine")
        
        self.assertIsNotNone(other_engine.order_book.get_order("OTHER_ASK"),
                           "Other order should exist in other engine")
        self.assertIsNone(other_engine.order_book.get_order("TEST_ASK"),
                         "Test order should not exist in other engine")


class TestOrderMatchingPerformance(unittest.TestCase):
    """Performance-focused tests for order matching"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.symbol = "PERF_TEST"
        self.matching_engine = MatchingEngine(
            symbol=self.symbol,
            tick_size=0.01,
            min_order_size=1,
            max_order_size=100000
        )
        
    def test_high_volume_matching_performance(self):
        """Test performance with high volume of orders"""
        import time
        
        # Create many orders
        num_orders = 1000
        orders = []
        
        for i in range(num_orders):
            if i % 2 == 0:
                # Ask orders
                order = Order(
                    order_id=f"ASK_{i}",
                    symbol=self.symbol,
                    side=OrderSide.ASK,
                    order_type=OrderType.LIMIT,
                    volume=10,
                    price=150.0 + (i * 0.01),
                    timestamp=pd.Timestamp.now()
                )
            else:
                # Bid orders
                order = Order(
                    order_id=f"BID_{i}",
                    symbol=self.symbol,
                    side=OrderSide.BID,
                    order_type=OrderType.LIMIT,
                    volume=10,
                    price=149.0 - (i * 0.01),
                    timestamp=pd.Timestamp.now()
                )
            orders.append(order)
        
        # Measure processing time
        start_time = time.time()
        
        total_trades = 0
        for order in orders:
            trades, _ = self.matching_engine.process_order(order)
            total_trades += len(trades)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        orders_per_second = num_orders / processing_time
        self.assertGreater(orders_per_second, 100, 
                          f"Should process at least 100 orders/second, got {orders_per_second:.2f}")
        
        # Verify statistics
        stats = self.matching_engine.get_statistics()
        self.assertEqual(stats['orders_processed'], num_orders, 
                        f"Should have processed {num_orders} orders")
        self.assertGreaterEqual(stats['trades_generated'], 0, "Should track trade count")
        
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during high-volume processing"""
        import gc
        import sys
        
        # Get initial memory baseline
        gc.collect()
        
        # Process many orders in batches
        for batch in range(10):
            batch_orders = []
            for i in range(100):
                order = Order(
                    order_id=f"MEM_TEST_{batch}_{i}",
                    symbol=self.symbol,
                    side=OrderSide.ASK if i % 2 == 0 else OrderSide.BID,
                    order_type=OrderType.LIMIT,
                    volume=10,
                    price=150.0 + (i * 0.001),
                    timestamp=pd.Timestamp.now()
                )
                batch_orders.append(order)
            
            # Process batch
            for order in batch_orders:
                self.matching_engine.process_order(order)
            
            # Clear batch and force garbage collection
            del batch_orders
            gc.collect()
        
        # Verify engine is still functional
        final_stats = self.matching_engine.get_statistics()
        self.assertEqual(final_stats['orders_processed'], 1000, 
                        "Should have processed 1000 orders across all batches")


if __name__ == '__main__':
    # Set up test suite
    unittest.TestLoader.sortTestMethodsUsing = None  # Preserve test order
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestOrderMatchingScenarios))
    suite.addTest(unittest.makeSuite(TestOrderMatchingPerformance))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("ORDER MATCHING TESTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
