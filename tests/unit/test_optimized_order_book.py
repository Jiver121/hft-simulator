"""
Comprehensive Unit Tests for OptimizedOrderBook

This module provides extensive testing for:
- OptimizedOrderBook vectorized operations
- Performance optimizations and memory pooling
- Batch processing functionality
- Compatibility with standard OrderBook interface
- Edge cases and error handling
"""

import unittest
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from src.engine.optimized_order_book import OptimizedOrderBook
from src.engine.order_types import Order, Trade, clear_order_id_registry
from src.utils.constants import OrderSide, OrderType, OrderStatus


class TestOptimizedOrderBook(unittest.TestCase):
    """Comprehensive tests for OptimizedOrderBook class"""

    def setUp(self):
        """Set up test fixtures"""
        self.symbol = 'AAPL'
        self.order_book = OptimizedOrderBook(self.symbol)
        self.timestamp = pd.Timestamp.now()
        # Clear order ID registry for clean tests
        clear_order_id_registry()

    def tearDown(self):
        """Clean up after tests"""
        clear_order_id_registry()

    def test_optimized_order_book_initialization(self):
        """Test OptimizedOrderBook initialization"""
        self.assertEqual(self.order_book.symbol, self.symbol)
        self.assertEqual(self.order_book.max_levels, 1000)
        self.assertEqual(self.order_book.num_bid_levels, 0)
        self.assertEqual(self.order_book.num_ask_levels, 0)
        self.assertEqual(self.order_book.num_orders, 0)
        
        # Check numpy arrays are initialized
        self.assertEqual(len(self.order_book.bid_prices), 1000)
        self.assertEqual(len(self.order_book.ask_prices), 1000)
        self.assertEqual(len(self.order_book.bid_volumes), 1000)
        self.assertEqual(len(self.order_book.ask_volumes), 1000)

    def test_single_order_addition_bid(self):
        """Test adding single bid order"""
        order = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT, 
                     volume=100, price=150.00, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(order)
        
        # No trades should occur
        self.assertEqual(len(trades), 0)
        
        # Check order book state
        self.assertEqual(self.order_book.num_bid_levels, 1)
        self.assertEqual(self.order_book.get_best_bid(), 150.00)
        self.assertEqual(self.order_book.get_best_bid_volume(), 100)

    def test_single_order_addition_ask(self):
        """Test adding single ask order"""
        order = Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT, 
                     volume=200, price=151.00, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(order)
        
        # No trades should occur
        self.assertEqual(len(trades), 0)
        
        # Check order book state
        self.assertEqual(self.order_book.num_ask_levels, 1)
        self.assertEqual(self.order_book.get_best_ask(), 151.00)
        self.assertEqual(self.order_book.get_best_ask_volume(), 200)

    def test_market_order_execution_buy(self):
        """Test market buy order execution"""
        # Set up ask orders
        ask1 = Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                    volume=100, price=151.00, timestamp=self.timestamp)
        ask2 = Order('ASK002', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                    volume=150, price=151.50, timestamp=self.timestamp)
        
        self.order_book.add_order(ask1)
        self.order_book.add_order(ask2)
        
        # Execute market buy order
        market_order = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                           volume=200, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(market_order)
        
        # Should generate two trades
        self.assertEqual(len(trades), 2)
        
        # First trade: 100 @ 151.00
        self.assertEqual(trades[0].volume, 100)
        self.assertEqual(trades[0].price, 151.00)
        
        # Second trade: 100 @ 151.50
        self.assertEqual(trades[1].volume, 100)
        self.assertEqual(trades[1].price, 151.50)
        
        # Check remaining ask liquidity
        self.assertEqual(self.order_book.get_best_ask(), 151.50)
        self.assertEqual(self.order_book.get_best_ask_volume(), 50)  # 150 - 100

    def test_market_order_execution_sell(self):
        """Test market sell order execution"""
        # Set up bid orders
        bid1 = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                    volume=100, price=150.00, timestamp=self.timestamp)
        bid2 = Order('BID002', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                    volume=150, price=149.50, timestamp=self.timestamp)
        
        self.order_book.add_order(bid1)
        self.order_book.add_order(bid2)
        
        # Execute market sell order
        market_order = Order('MKT001', self.symbol, OrderSide.SELL, OrderType.MARKET,
                           volume=200, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(market_order)
        
        # Should generate two trades
        self.assertEqual(len(trades), 2)
        
        # First trade: 100 @ 150.00 (best bid)
        self.assertEqual(trades[0].volume, 100)
        self.assertEqual(trades[0].price, 150.00)
        
        # Second trade: 100 @ 149.50
        self.assertEqual(trades[1].volume, 100)
        self.assertEqual(trades[1].price, 149.50)
        
        # Check remaining bid liquidity
        self.assertEqual(self.order_book.get_best_bid(), 149.50)
        self.assertEqual(self.order_book.get_best_bid_volume(), 50)  # 150 - 100

    def test_limit_order_matching(self):
        """Test limit order matching"""
        # Set up existing orders
        bid = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                   volume=100, price=150.00, timestamp=self.timestamp)
        ask = Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                   volume=150, price=151.00, timestamp=self.timestamp)
        
        self.order_book.add_order(bid)
        self.order_book.add_order(ask)
        
        # Add aggressive buy limit order
        aggressive_buy = Order('BUY001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                             volume=120, price=151.50, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(aggressive_buy)
        
        # Should match with ask at 151.00
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].volume, 120)
        self.assertEqual(trades[0].price, 151.00)
        
        # Check remaining ask volume
        self.assertEqual(self.order_book.get_best_ask_volume(), 30)  # 150 - 120

    def test_batch_order_processing(self):
        """Test batch order processing functionality"""
        orders = []
        
        # Create batch of limit orders
        for i in range(10):
            bid_order = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                            volume=100, price=150.00 - i*0.10, timestamp=self.timestamp)
            ask_order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                            volume=100, price=151.00 + i*0.10, timestamp=self.timestamp)
            orders.extend([bid_order, ask_order])
        
        # Process batch
        trades = self.order_book.process_order_batch(orders)
        
        # No trades should occur as there's no overlap
        self.assertEqual(len(trades), 0)
        
        # Check order book state
        self.assertEqual(self.order_book.num_bid_levels, 10)
        self.assertEqual(self.order_book.num_ask_levels, 10)
        self.assertEqual(self.order_book.get_best_bid(), 150.00)
        self.assertEqual(self.order_book.get_best_ask(), 151.00)

    def test_vectorized_order_processing(self):
        """Test vectorized order processing"""
        # Set up initial liquidity
        ask_orders = []
        for i in range(5):
            ask_order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                            volume=100, price=151.00 + i*0.25, timestamp=self.timestamp)
            ask_orders.append(ask_order)
        
        self.order_book.process_order_batch(ask_orders)
        
        # Create batch of market buy orders
        market_orders = []
        for i in range(3):
            market_order = Order(f'MKT{i:03d}', self.symbol, OrderSide.BUY, OrderType.MARKET,
                               volume=80, timestamp=self.timestamp + pd.Timedelta(seconds=i))
            market_orders.append(market_order)
        
        # Process vectorized
        trades = self.order_book.process_orders_vectorized(market_orders)
        
        # Should generate trades
        self.assertGreater(len(trades), 0)
        
        # Check statistics were updated
        stats = self.order_book.get_statistics()
        self.assertGreater(stats['performance_stats']['vectorized_matches'], 0)

    def test_order_aggregation_same_price(self):
        """Test order aggregation at same price level"""
        # Add multiple orders at same price
        orders = []
        for i in range(5):
            order = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                        volume=100, price=150.00, timestamp=self.timestamp)
            orders.append(order)
        
        self.order_book.process_order_batch(orders)
        
        # Should aggregate to single price level
        self.assertEqual(self.order_book.num_bid_levels, 1)
        self.assertEqual(self.order_book.get_best_bid_volume(), 500)  # 5 * 100

    def test_price_level_sorting(self):
        """Test proper price level sorting"""
        # Add bid orders in random order
        bid_prices = [149.50, 150.00, 149.00, 150.50, 149.75]
        bid_orders = []
        for i, price in enumerate(bid_prices):
            order = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                        volume=100, price=price, timestamp=self.timestamp)
            bid_orders.append(order)
        
        # Add ask orders in random order
        ask_prices = [151.50, 151.00, 152.00, 151.25, 151.75]
        ask_orders = []
        for i, price in enumerate(ask_prices):
            order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                        volume=100, price=price, timestamp=self.timestamp)
            ask_orders.append(order)
        
        all_orders = bid_orders + ask_orders
        self.order_book.process_order_batch(all_orders)
        
        # Check sorting: bids descending
        bid_depth = self.order_book.get_bids(5)
        expected_bid_prices = [150.50, 150.00, 149.75, 149.50, 149.00]
        actual_bid_prices = [price for price, volume in bid_depth]
        self.assertEqual(actual_bid_prices, expected_bid_prices)
        
        # Check sorting: asks ascending
        ask_depth = self.order_book.get_asks(5)
        expected_ask_prices = [151.00, 151.25, 151.50, 151.75, 152.00]
        actual_ask_prices = [price for price, volume in ask_depth]
        self.assertEqual(actual_ask_prices, expected_ask_prices)

    def test_order_cancellation(self):
        """Test order cancellation functionality"""
        # Add an order
        order = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                     volume=100, price=150.00, timestamp=self.timestamp)
        
        self.order_book.add_order(order)
        
        # Verify order exists
        self.assertEqual(self.order_book.get_best_bid_volume(), 100)
        
        # Cancel order
        success = self.order_book.cancel_order('BID001')
        self.assertTrue(success)
        
        # Verify order removed
        self.assertIsNone(self.order_book.get_best_bid())
        self.assertEqual(self.order_book.num_bid_levels, 0)

    def test_order_modification(self):
        """Test order modification functionality"""
        # Add original order
        original_order = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                              volume=100, price=150.00, timestamp=self.timestamp)
        
        self.order_book.add_order(original_order)
        
        # Create modified order
        modified_order = Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                              volume=200, price=150.50, timestamp=self.timestamp)
        
        # Modify order
        success = self.order_book.modify_order('BID001', original_order, modified_order)
        self.assertTrue(success)
        
        # Check modification
        self.assertEqual(self.order_book.get_best_bid(), 150.50)
        self.assertEqual(self.order_book.get_best_bid_volume(), 200)

    def test_empty_levels_cleanup(self):
        """Test cleanup of empty price levels"""
        # Add orders at different levels
        orders = []
        for i in range(3):
            order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                        volume=100, price=151.00 + i*0.25, timestamp=self.timestamp)
            orders.append(order)
        
        self.order_book.process_order_batch(orders)
        
        # Execute large market order to consume all liquidity
        market_order = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                           volume=300, timestamp=self.timestamp)
        
        trades = self.order_book.add_order(market_order)
        
        # All ask levels should be cleared
        self.assertEqual(self.order_book.num_ask_levels, 0)
        self.assertIsNone(self.order_book.get_best_ask())

    def test_performance_statistics(self):
        """Test performance statistics tracking"""
        # Add some orders and execute trades
        orders = []
        for i in range(10):
            bid = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                       volume=100, price=149.00 + i*0.10, timestamp=self.timestamp)
            ask = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=100, price=151.00 + i*0.10, timestamp=self.timestamp)
            orders.extend([bid, ask])
        
        self.order_book.process_order_batch(orders)
        
        # Execute some market orders
        market_buy = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                          volume=300, timestamp=self.timestamp)
        self.order_book.add_order(market_buy)
        
        # Get statistics
        stats = self.order_book.get_statistics()
        
        # Check basic statistics
        self.assertEqual(stats['symbol'], self.symbol)
        self.assertGreaterEqual(stats['performance_stats']['orders_processed'], 20)
        self.assertGreaterEqual(stats['performance_stats']['trades_executed'], 3)
        
        # Check memory efficiency stats
        self.assertIn('memory_efficiency', stats)
        self.assertGreater(stats['memory_efficiency']['memory_usage_mb'], 0)

    def test_memory_pooling(self):
        """Test memory pooling functionality"""
        # Get initial pool statistics
        initial_stats = self.order_book.get_pool_statistics()
        
        # Execute trades to create trade objects
        ask_order = Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                         volume=100, price=151.00, timestamp=self.timestamp)
        self.order_book.add_order(ask_order)
        
        market_order = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                           volume=100, timestamp=self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Return trades to pool
        for trade in trades:
            self.order_book.return_trade_to_pool(trade)
        
        # Get final pool statistics
        final_stats = self.order_book.get_pool_statistics()
        
        # Pool should have been used
        self.assertIn('pool_hits', final_stats)
        self.assertIn('pool_misses', final_stats)

    def test_market_data_snapshot(self):
        """Test market data snapshot generation"""
        # Add some orders
        bid_orders = [
            Order('BID001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                 volume=100, price=150.00, timestamp=self.timestamp),
            Order('BID002', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                 volume=200, price=149.50, timestamp=self.timestamp)
        ]
        
        ask_orders = [
            Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                 volume=150, price=151.00, timestamp=self.timestamp),
            Order('ASK002', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                 volume=250, price=151.50, timestamp=self.timestamp)
        ]
        
        for order in bid_orders + ask_orders:
            self.order_book.add_order(order)
        
        # Generate snapshot
        snapshot = self.order_book.get_snapshot()
        
        # Check snapshot properties
        self.assertEqual(snapshot.symbol, self.symbol)
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_ask, 151.00)
        self.assertEqual(len(snapshot.bids), 2)
        self.assertEqual(len(snapshot.asks), 2)

    def test_depth_array_retrieval(self):
        """Test depth array retrieval for vectorized processing"""
        # Add orders at multiple levels
        for i in range(5):
            bid_order = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                            volume=100 * (i + 1), price=150.00 - i*0.25, timestamp=self.timestamp)
            ask_order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                            volume=150 * (i + 1), price=151.00 + i*0.25, timestamp=self.timestamp)
            
            self.order_book.add_order(bid_order)
            self.order_book.add_order(ask_order)
        
        # Get depth arrays
        bid_prices, bid_volumes = self.order_book.get_depth_array('bid', 3)
        ask_prices, ask_volumes = self.order_book.get_depth_array('ask', 3)
        
        # Check array properties
        self.assertEqual(len(bid_prices), 3)
        self.assertEqual(len(bid_volumes), 3)
        self.assertEqual(len(ask_prices), 3)
        self.assertEqual(len(ask_volumes), 3)
        
        # Check data types
        self.assertIsInstance(bid_prices, np.ndarray)
        self.assertIsInstance(bid_volumes, np.ndarray)
        
        # Check values
        self.assertEqual(bid_prices[0], 150.00)  # Best bid
        self.assertEqual(ask_prices[0], 151.00)  # Best ask
        self.assertEqual(bid_volumes[0], 100)
        self.assertEqual(ask_volumes[0], 150)

    def test_large_order_handling(self):
        """Test handling of large orders"""
        # Create large order book
        orders = []
        for i in range(100):
            bid = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                       volume=100, price=150.00 - i*0.01, timestamp=self.timestamp)
            ask = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=100, price=151.00 + i*0.01, timestamp=self.timestamp)
            orders.extend([bid, ask])
        
        # Process in batch
        trades = self.order_book.process_order_batch(orders)
        
        # Should handle without issues
        self.assertEqual(len(trades), 0)  # No overlapping orders
        self.assertEqual(self.order_book.num_bid_levels, 100)
        self.assertEqual(self.order_book.num_ask_levels, 100)

    def test_order_book_clearing(self):
        """Test order book clearing functionality"""
        # Add some orders
        orders = []
        for i in range(5):
            bid = Order(f'BID{i:03d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                       volume=100, price=150.00 - i*0.10, timestamp=self.timestamp)
            ask = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=100, price=151.00 + i*0.10, timestamp=self.timestamp)
            orders.extend([bid, ask])
        
        self.order_book.process_order_batch(orders)
        
        # Verify orders exist
        self.assertGreater(self.order_book.num_bid_levels, 0)
        self.assertGreater(self.order_book.num_ask_levels, 0)
        
        # Clear the book
        self.order_book.clear()
        
        # Verify everything is cleared
        self.assertEqual(self.order_book.num_bid_levels, 0)
        self.assertEqual(self.order_book.num_ask_levels, 0)
        self.assertEqual(self.order_book.num_orders, 0)
        self.assertIsNone(self.order_book.get_best_bid())
        self.assertIsNone(self.order_book.get_best_ask())

    def test_order_type_variations(self):
        """Test handling different order types"""
        # Test with both OrderSide.BUY and OrderSide.BID
        buy_order = Order('BUY001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                         volume=100, price=150.00, timestamp=self.timestamp)
        bid_order = Order('BID001', self.symbol, OrderSide.BID, OrderType.LIMIT,
                         volume=200, price=149.50, timestamp=self.timestamp)
        
        # Test with both OrderSide.SELL and OrderSide.ASK
        sell_order = Order('SELL001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                          volume=150, price=151.00, timestamp=self.timestamp)
        ask_order = Order('ASK001', self.symbol, OrderSide.ASK, OrderType.LIMIT,
                         volume=250, price=151.50, timestamp=self.timestamp)
        
        # Add all orders
        for order in [buy_order, bid_order, sell_order, ask_order]:
            trades = self.order_book.add_order(order)
            self.assertEqual(len(trades), 0)  # No matches expected
        
        # Check final state
        self.assertEqual(self.order_book.num_bid_levels, 2)
        self.assertEqual(self.order_book.num_ask_levels, 2)
        self.assertEqual(self.order_book.get_best_bid(), 150.00)
        self.assertEqual(self.order_book.get_best_ask(), 151.00)

    def test_edge_case_zero_volume_cleanup(self):
        """Test cleanup when volumes reach zero"""
        # Add ask order
        ask_order = Order('ASK001', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                         volume=100, price=151.00, timestamp=self.timestamp)
        self.order_book.add_order(ask_order)
        
        # Execute exact matching volume
        market_order = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                           volume=100, timestamp=self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Should generate one trade and clean up empty level
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].volume, 100)
        self.assertEqual(self.order_book.num_ask_levels, 0)
        self.assertIsNone(self.order_book.get_best_ask())

    def test_partial_fill_scenarios(self):
        """Test various partial fill scenarios"""
        # Set up multiple ask levels
        ask_levels = [
            (151.00, 50),
            (151.25, 75),
            (151.50, 100)
        ]
        
        for i, (price, volume) in enumerate(ask_levels):
            order = Order(f'ASK{i:03d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                        volume=volume, price=price, timestamp=self.timestamp)
            self.order_book.add_order(order)
        
        # Execute market order that partially fills multiple levels
        market_order = Order('MKT001', self.symbol, OrderSide.BUY, OrderType.MARKET,
                           volume=175, timestamp=self.timestamp)
        trades = self.order_book.add_order(market_order)
        
        # Should generate 3 trades: 50@151.00, 75@151.25, 50@151.50
        self.assertEqual(len(trades), 3)
        
        # Check individual trades
        self.assertEqual(trades[0].volume, 50)
        self.assertEqual(trades[0].price, 151.00)
        
        self.assertEqual(trades[1].volume, 75)
        self.assertEqual(trades[1].price, 151.25)
        
        self.assertEqual(trades[2].volume, 50)
        self.assertEqual(trades[2].price, 151.50)
        
        # Check remaining liquidity
        self.assertEqual(self.order_book.get_best_ask(), 151.50)
        self.assertEqual(self.order_book.get_best_ask_volume(), 50)

    def test_compatibility_interface(self):
        """Test compatibility with standard OrderBook interface"""
        # Test all standard methods exist and work
        order = Order('TEST001', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                     volume=100, price=150.00, timestamp=self.timestamp)
        
        # Test process_order alias
        trades = self.order_book.process_order(order)
        self.assertEqual(len(trades), 0)
        
        # Test depth retrieval
        bids = self.order_book.get_bids(5)
        asks = self.order_book.get_asks(5)
        
        self.assertEqual(len(bids), 1)
        self.assertEqual(len(asks), 0)
        
        # Test spread and mid price
        spread = self.order_book.get_spread()
        mid_price = self.order_book.get_mid_price()
        
        self.assertIsNone(spread)  # No ask side
        self.assertIsNone(mid_price)  # No ask side

    def test_concurrent_safety_basics(self):
        """Test basic thread safety measures"""
        # This is a basic test - full concurrent testing would require threading
        order_book = OptimizedOrderBook(self.symbol, max_levels=100)
        
        # Test that locks and pools are initialized
        self.assertIsNotNone(order_book._pool_lock)
        self.assertIsNotNone(order_book._order_pool)
        self.assertIsNotNone(order_book._trade_pool)
        
        # Test pool statistics
        pool_stats = order_book.get_pool_statistics()
        expected_keys = {'trade_pool_size', 'trade_pool_max', 'order_pool_size', 
                        'order_pool_max', 'pool_hits', 'pool_misses'}
        self.assertEqual(set(pool_stats.keys()), expected_keys)


class TestOptimizedOrderBookPerformance(unittest.TestCase):
    """Performance-focused tests for OptimizedOrderBook"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.symbol = 'AAPL'
        self.order_book = OptimizedOrderBook(self.symbol)
        self.timestamp = pd.Timestamp.now()
        clear_order_id_registry()

    def tearDown(self):
        """Clean up after tests"""
        clear_order_id_registry()

    def test_batch_processing_performance(self):
        """Test performance of batch processing vs individual orders"""
        import time
        
        # Create large batch of orders
        orders = []
        for i in range(1000):
            bid = Order(f'BID{i:04d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                       volume=100, price=150.00 - i*0.001, timestamp=self.timestamp)
            ask = Order(f'ASK{i:04d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=100, price=151.00 + i*0.001, timestamp=self.timestamp)
            orders.extend([bid, ask])
        
        # Test batch processing
        start_time = time.time()
        trades = self.order_book.process_order_batch(orders)
        batch_time = time.time() - start_time
        
        # Clear and test individual processing
        self.order_book.clear()
        
        start_time = time.time()
        for order in orders:
            self.order_book.add_order(order)
        individual_time = time.time() - start_time
        
        # Batch should be competitive (not necessarily faster due to overhead, but reasonable)
        self.assertLess(batch_time, individual_time * 2,  # Allow some overhead
                       f"Batch processing ({batch_time:.4f}s) significantly slower than individual ({individual_time:.4f}s)")
        
        # Both should complete in reasonable time
        self.assertLess(batch_time, 5.0, "Batch processing should complete in under 5 seconds")
        self.assertLess(individual_time, 5.0, "Individual processing should complete in under 5 seconds")

    def test_vectorized_operations_performance(self):
        """Test performance of vectorized operations"""
        import time
        
        # Set up large order book
        setup_orders = []
        for i in range(500):
            ask = Order(f'ASK{i:04d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=100, price=151.00 + i*0.001, timestamp=self.timestamp)
            setup_orders.append(ask)
        
        self.order_book.process_order_batch(setup_orders)
        
        # Create batch of market orders for vectorized processing
        market_orders = []
        for i in range(100):
            market_order = Order(f'MKT{i:03d}', self.symbol, OrderSide.BUY, OrderType.MARKET,
                               volume=50, timestamp=self.timestamp + pd.Timedelta(microseconds=i))
            market_orders.append(market_order)
        
        # Test vectorized processing
        start_time = time.time()
        trades = self.order_book.process_orders_vectorized(market_orders)
        vectorized_time = time.time() - start_time
        
        # Should complete efficiently
        self.assertLess(vectorized_time, 2.0, "Vectorized processing should be fast")
        self.assertGreater(len(trades), 0, "Should generate trades")
        
        # Check performance statistics
        stats = self.order_book.get_statistics()
        self.assertGreater(stats['performance_stats']['vectorized_matches'], 0)

    def test_memory_efficiency(self):
        """Test memory efficiency of the optimized order book"""
        # Add many orders to test memory usage
        orders = []
        for i in range(10000):
            if i % 2 == 0:
                order = Order(f'BID{i:05d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                            volume=100, price=150.00 - i*0.0001, timestamp=self.timestamp)
            else:
                order = Order(f'ASK{i:05d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                            volume=100, price=151.00 + i*0.0001, timestamp=self.timestamp)
            orders.append(order)
        
        # Process batch
        trades = self.order_book.process_order_batch(orders)
        
        # Check memory usage
        stats = self.order_book.get_statistics()
        memory_mb = stats['memory_efficiency']['memory_usage_mb']
        
        # Memory usage should be reasonable (less than 100MB for this test)
        self.assertLess(memory_mb, 100, f"Memory usage ({memory_mb:.2f}MB) should be reasonable")
        
        # Should handle large number of orders
        self.assertEqual(self.order_book.num_bid_levels + self.order_book.num_ask_levels, 
                        min(10000, 2000))  # Limited by max_levels

    def test_cleanup_efficiency(self):
        """Test efficiency of level cleanup operations"""
        import time
        
        # Create fragmented order book
        orders = []
        for i in range(1000):
            # Create orders that will leave small volumes
            bid = Order(f'BID{i:04d}', self.symbol, OrderSide.BUY, OrderType.LIMIT,
                       volume=i % 10 + 1, price=150.00 - i*0.001, timestamp=self.timestamp)
            ask = Order(f'ASK{i:04d}', self.symbol, OrderSide.SELL, OrderType.LIMIT,
                       volume=i % 10 + 1, price=151.00 + i*0.001, timestamp=self.timestamp)
            orders.extend([bid, ask])
        
        self.order_book.process_order_batch(orders)
        
        # Execute large market orders to create empty levels
        large_buy = Order('LARGE_BUY', self.symbol, OrderSide.BUY, OrderType.MARKET,
                         volume=50000, timestamp=self.timestamp)
        
        start_time = time.time()
        trades = self.order_book.add_order(large_buy)
        cleanup_time = time.time() - start_time
        
        # Cleanup should be efficient
        self.assertLess(cleanup_time, 1.0, "Cleanup should be fast")
        
        # Should have removed empty levels
        self.assertEqual(self.order_book.num_ask_levels, 0)


if __name__ == '__main__':
    unittest.main()
