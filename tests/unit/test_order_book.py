"""
Unit Tests for Order Book Engine

This module contains comprehensive unit tests for the order book engine,
testing all aspects of order management, matching, and book reconstruction.

Educational Notes:
- Order book tests verify correct price-time priority
- Tests ensure proper order matching and execution
- Edge cases like empty books and invalid orders are tested
- Performance characteristics are validated
"""

import unittest
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import List, Tuple

from src.engine.optimized_order_book import OptimizedOrderBook as OrderBook
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot, MarketData
from src.utils.constants import OrderSide, OrderType


class TestOrderBook(unittest.TestCase):
    """Test cases for OrderBook class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.symbol = "AAPL"
        self.order_book = OrderBook(self.symbol)
        self.timestamp = pd.Timestamp.now()
    
    def test_initialization(self):
        """Test order book initialization"""
        self.assertEqual(self.order_book.symbol, self.symbol)
        self.assertEqual(len(self.order_book.bids), 0)
        self.assertEqual(len(self.order_book.asks), 0)
        self.assertIsNone(self.order_book.get_best_bid())
        self.assertIsNone(self.order_book.get_best_ask())
    
    def test_add_bid_order(self):
        """Test adding bid orders"""
        order = Order(
            order_id="BID001",
            symbol=self.symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=100,
            timestamp=self.timestamp,
            price=150.00
        )
        
        self.order_book.add_order(order)
        
        # Check order was added
        self.assertEqual(len(self.order_book.bids), 1)
        self.assertEqual(self.order_book.get_best_bid(), 150.00)
        self.assertEqual(self.order_book.get_best_bid_volume(), 100)
        
        # Check order is in the book
        self.assertIn(150.00, self.order_book.bids)
        self.assertEqual(self.order_book.bids[150.00].total_volume, 100)
    
    def test_add_ask_order(self):
        """Test adding ask orders"""
        order = Order(
            order_id="ASK001",
            symbol=self.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            volume=200,
            timestamp=self.timestamp,
            price=151.00
        )
        
        self.order_book.add_order(order)
        
        # Check order was added
        self.assertEqual(len(self.order_book.asks), 1)
        self.assertEqual(self.order_book.get_best_ask(), 151.00)
        self.assertEqual(self.order_book.get_best_ask_volume(), 200)
        
        # Check order is in the book
        self.assertIn(151.00, self.order_book.asks)
        self.assertEqual(self.order_book.asks[151.00].total_volume, 200)
    
    def test_price_time_priority_bids(self):
        """Test price-time priority for bid orders"""
        # Add orders with different prices
        order1 = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        order2 = Order("BID002", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=151.00)
        order3 = Order("BID003", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=149.00)
        
        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        self.order_book.add_order(order3)
        
        # Best bid should be highest price
        self.assertEqual(self.order_book.get_best_bid(), 151.00)
        
        # Check bid ordering
        bid_prices = list(self.order_book.bids.keys())
        self.assertEqual(bid_prices, [151.00, 150.00, 149.00])  # Descending order
    
    def test_price_time_priority_asks(self):
        """Test price-time priority for ask orders"""
        # Add orders with different prices
        order1 = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=152.00)
        order2 = Order("ASK002", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=151.00)
        order3 = Order("ASK003", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=153.00)
        
        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        self.order_book.add_order(order3)
        
        # Best ask should be lowest price
        self.assertEqual(self.order_book.get_best_ask(), 151.00)
        
        # Check ask ordering
        ask_prices = list(self.order_book.asks.keys())
        self.assertEqual(ask_prices, [151.00, 152.00, 153.00])  # Ascending order
    
    def test_order_aggregation(self):
        """Test order aggregation at same price level"""
        # Add multiple orders at same price
        order1 = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        order2 = Order("BID002", self.symbol, OrderSide.BUY, OrderType.LIMIT, 200, self.timestamp, price=150.00)
        order3 = Order("BID003", self.symbol, OrderSide.BUY, OrderType.LIMIT, 150, self.timestamp, price=150.00)
        
        self.order_book.add_order(order1)
        self.order_book.add_order(order2)
        self.order_book.add_order(order3)
        
        # Check aggregation
        self.assertEqual(self.order_book.bids[150.00].total_volume, 450)  # 100 + 200 + 150
        self.assertEqual(self.order_book.get_best_bid(), 150.00)
        self.assertEqual(self.order_book.get_best_bid_volume(), 450)
    
    def test_market_order_execution(self):
        """Test market order execution"""
        # Set up order book
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 200, self.timestamp, price=151.00)
        
        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        
        # Execute market buy order
        market_order = Order("MKT001", self.symbol, OrderSide.BUY, OrderType.MARKET, 50, self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Check trade execution
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.volume, 50)
        self.assertEqual(trade.price, 151.00)  # Should execute at ask price
        
        # Check remaining quantity
        self.assertEqual(self.order_book.asks[151.00].total_volume, 150)  # 200 - 50
    
    def test_limit_order_matching(self):
        """Test limit order matching"""
        # Set up order book with spread
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 200, self.timestamp, price=152.00)
        
        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        
        # Add aggressive buy order that crosses spread
        aggressive_order = Order("BUY001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 150, self.timestamp, price=152.50)
        trades = self.order_book.add_order(aggressive_order)
        
        # Check trade execution
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.volume, 150)
        self.assertEqual(trade.price, 152.00)  # Should execute at existing ask price
        
        # Check remaining quantities
        self.assertEqual(self.order_book.asks[152.00].total_volume, 50)  # 200 - 150
    
    def test_partial_fill(self):
        """Test partial order fills"""
        # Set up small ask order
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 50, self.timestamp, price=151.00)
        self.order_book.add_order(ask_order)
        
        # Large market buy order
        market_order = Order("MKT001", self.symbol, OrderSide.BUY, OrderType.MARKET, 200, self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Should only fill available quantity
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.volume, 50)  # Only what was available
        
        # Ask should be completely filled
        self.assertNotIn(151.00, self.order_book.asks)
    
    def test_multiple_level_execution(self):
        """Test execution across multiple price levels"""
        # Set up multiple ask levels
        ask1 = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=151.00)
        ask2 = Order("ASK002", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=151.50)
        ask3 = Order("ASK003", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=152.00)
        
        self.order_book.add_order(ask1)
        self.order_book.add_order(ask2)
        self.order_book.add_order(ask3)
        
        # Large market buy order
        market_order = Order("MKT001", self.symbol, OrderSide.BUY, OrderType.MARKET, 250, self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Should execute across multiple levels
        self.assertEqual(len(trades), 3)
        
        # Check trade details
        self.assertEqual(trades[0].price, 151.00)
        self.assertEqual(trades[0].volume, 100)
        self.assertEqual(trades[1].price, 151.50)
        self.assertEqual(trades[1].volume, 100)
        self.assertEqual(trades[2].price, 152.00)
        self.assertEqual(trades[2].volume, 50)  # Partial fill of third level
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation"""
        # Empty book
        self.assertIsNone(self.order_book.get_spread())
        
        # Add bid only
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        self.order_book.add_order(bid_order)
        self.assertIsNone(self.order_book.get_spread())
        
        # Add ask
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=152.00)
        self.order_book.add_order(ask_order)
        
        # Check spread
        spread = self.order_book.get_spread()
        self.assertEqual(spread, 2.00)  # 152.00 - 150.00
    
    def test_mid_price_calculation(self):
        """Test mid price calculation"""
        # Empty book
        self.assertIsNone(self.order_book.get_mid_price())
        
        # Add orders
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=152.00)
        
        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        
        # Check mid price
        mid_price = self.order_book.get_mid_price()
        self.assertEqual(mid_price, 151.00)  # (150.00 + 152.00) / 2
    
    def test_order_book_depth(self):
        """Test order book depth retrieval"""
        # Add multiple levels
        for i in range(5):
            bid_price = 150.00 - i * 0.50
            ask_price = 151.00 + i * 0.50
            
            bid_order = Order(f"BID{i}", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=bid_price)
            ask_order = Order(f"ASK{i}", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=ask_price)
            
            self.order_book.add_order(bid_order)
            self.order_book.add_order(ask_order)
        
        # Test depth retrieval
        bids_3 = self.order_book.get_depth(OrderSide.BUY, 3)
        asks_3 = self.order_book.get_depth(OrderSide.SELL, 3)
        
        self.assertEqual(len(bids_3), 3)
        self.assertEqual(len(asks_3), 3)
        
        # Check ordering
        self.assertEqual(bids_3[0], (150.00, 100))  # Best bid first
        self.assertEqual(asks_3[0], (151.00, 100))  # Best ask first
    
    def test_order_book_snapshot(self):
        """Test order book snapshot creation"""
        # Add some orders
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 200, self.timestamp, price=151.00)
        
        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        
        # Create snapshot
        snapshot = self.order_book.get_snapshot()
        
        # Check snapshot
        self.assertIsInstance(snapshot, BookSnapshot)
        self.assertEqual(snapshot.symbol, self.symbol)
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_bid_volume, 100)
        self.assertEqual(snapshot.best_ask, 151.00)
        self.assertEqual(snapshot.best_ask_volume, 200)
        self.assertIsNotNone(snapshot.timestamp)
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        # Add order
        order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        self.order_book.add_order(order)
        
        # Verify order exists
        self.assertEqual(self.order_book.bids[150.00].total_volume, 100)
        
        # Cancel order
        success = self.order_book.cancel_order("BID001")
        self.assertTrue(success)
        
        # Verify order removed
        self.assertNotIn(150.00, self.order_book.bids)
    
    def test_order_modification(self):
        """Test order modification"""
        # Add order
        original_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        self.order_book.add_order(original_order)
        
        # Modify order
        modified_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 200, self.timestamp, price=150.50)
        success = self.order_book.modify_order("BID001", original_order, modified_order)
        self.assertTrue(success)
        
        # Check modification
        self.assertNotIn(150.00, self.order_book.bids)
        self.assertEqual(self.order_book.bids[150.50].total_volume, 200)
    
    def test_invalid_orders(self):
        """Test handling of invalid orders"""
        # Test negative quantity
        with self.assertRaises(ValueError):
            invalid_order = Order("INVALID", self.symbol, OrderSide.BUY, OrderType.LIMIT, -100, self.timestamp, price=150.00)
            self.order_book.add_order(invalid_order)
        
        # Test zero quantity
        with self.assertRaises(ValueError):
            invalid_order = Order("INVALID", self.symbol, OrderSide.BUY, OrderType.LIMIT, 0, self.timestamp, price=150.00)
            self.order_book.add_order(invalid_order)
        
        # Test negative price
        with self.assertRaises(ValueError):
            invalid_order = Order("INVALID", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=-150.00)
            self.order_book.add_order(invalid_order)
    
    def test_order_book_reset(self):
        """Test order book reset functionality"""
        # Add some orders
        bid_order = Order("BID001", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=150.00)
        ask_order = Order("ASK001", self.symbol, OrderSide.SELL, OrderType.LIMIT, 200, self.timestamp, price=151.00)
        
        self.order_book.add_order(bid_order)
        self.order_book.add_order(ask_order)
        
        # Verify orders exist
        self.assertEqual(len(self.order_book.bids), 1)
        self.assertEqual(len(self.order_book.asks), 1)
        
        # Reset book
        self.order_book.clear()
        
        # Verify book is empty
        self.assertEqual(len(self.order_book.bids), 0)
        self.assertEqual(len(self.order_book.asks), 0)
        self.assertIsNone(self.order_book.get_best_bid())
        self.assertIsNone(self.order_book.get_best_ask())
    
    def test_order_book_statistics(self):
        """Test order book statistics calculation"""
        # Add orders at multiple levels
        for i in range(3):
            bid_price = 150.00 - i * 0.25
            ask_price = 151.00 + i * 0.25
            quantity = 100 * (i + 1)
            
            bid_order = Order(f"BID{i}", self.symbol, OrderSide.BUY, OrderType.LIMIT, quantity, self.timestamp, price=bid_price)
            ask_order = Order(f"ASK{i}", self.symbol, OrderSide.SELL, OrderType.LIMIT, quantity, self.timestamp, price=ask_price)
            
            self.order_book.add_order(bid_order)
            self.order_book.add_order(ask_order)
        
        # Get statistics
        stats = self.order_book.get_statistics()
        
        # Check statistics
        self.assertEqual(stats['bid_levels'], 3)
        self.assertEqual(stats['ask_levels'], 3)
        self.assertEqual(stats['total_bid_volume'], 600)  # 100 + 200 + 300
        self.assertEqual(stats['total_ask_volume'], 600)
        self.assertEqual(stats['spread'], 1.00)  # 151.00 - 150.00
        self.assertEqual(stats['mid_price'], 150.50)


class TestOrderBookPerformance(unittest.TestCase):
    """Performance tests for OrderBook class"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.symbol = "AAPL"
        self.order_book = OrderBook(self.symbol)
        self.timestamp = pd.Timestamp.now()
    
    def test_large_order_book_performance(self):
        """Test performance with large number of orders"""
        import time
        
        # Add 10,000 orders
        start_time = time.time()
        
        for i in range(5000):
            bid_price = 150.00 - i * 0.01
            ask_price = 151.00 + i * 0.01
            
            bid_order = Order(f"BID{i}", self.symbol, OrderSide.BUY, OrderType.LIMIT, 100, self.timestamp, price=bid_price)
            ask_order = Order(f"ASK{i}", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=ask_price)
            
            self.order_book.add_order(bid_order)
            self.order_book.add_order(ask_order)
        
        add_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        for _ in range(1000):
            self.order_book.get_depth(OrderSide.BUY, 10)
            self.order_book.get_depth(OrderSide.SELL, 10)
            self.order_book.get_spread()
            self.order_book.get_mid_price()
        
        retrieval_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        self.assertLess(add_time, 5.0, "Adding 10,000 orders should take less than 5 seconds")
        self.assertLess(retrieval_time, 1.0, "1,000 retrievals should take less than 1 second")
    
    def test_market_order_execution_performance(self):
        """Test market order execution performance"""
        import time
        
        # Set up large order book
        for i in range(1000):
            ask_price = 151.00 + i * 0.01
            ask_order = Order(f"ASK{i}", self.symbol, OrderSide.SELL, OrderType.LIMIT, 100, self.timestamp, price=ask_price)
            self.order_book.add_order(ask_order)
        
        # Test large market order execution
        start_time = time.time()
        
        market_order = Order("MKT001", self.symbol, OrderSide.BUY, OrderType.MARKET, 50000, self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        execution_time = time.time() - start_time
        
        # Performance assertion
        self.assertLess(execution_time, 1.0, "Large market order execution should take less than 1 second")
        self.assertGreater(len(trades), 0, "Should generate trades")


if __name__ == '__main__':
    unittest.main()