"""
Comprehensive Unit Tests for Order Types

This module provides extensive testing for the Order class including:
- Constructor variations (volume/quantity parameters)
- Order validation and edge cases
- Order helper methods
- Trade and OrderUpdate creation
- Market data point functionality
"""

import unittest
import pandas as pd
import warnings
import uuid
from typing import Dict, Any
from datetime import datetime

from src.engine.order_types import (
    Order, Trade, OrderUpdate, PriceLevel, MarketDataPoint,
    validate_order_id_uniqueness, remove_order_id_from_registry,
    clear_order_id_registry, get_order_id_registry_size
)
from src.utils.constants import OrderSide, OrderType, OrderStatus


class TestOrder(unittest.TestCase):
    """Comprehensive tests for Order class"""

    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = pd.Timestamp.now()
        self.base_order_args = {
            'symbol': 'AAPL',
            'side': OrderSide.BUY,
            'order_type': OrderType.LIMIT,
            'timestamp': self.timestamp,
            'price': 150.50
        }
        # Clear registry before each test
        clear_order_id_registry()

    def tearDown(self):
        """Clean up after tests"""
        clear_order_id_registry()

    def test_order_constructor_with_volume(self):
        """Test Order constructor with volume parameter"""
        order = Order(
            order_id='TEST001',
            volume=100,
            **self.base_order_args
        )
        
        self.assertEqual(order.volume, 100)
        self.assertEqual(order.remaining_volume, 100)
        self.assertEqual(order.filled_volume, 0)
        self.assertEqual(order.status, OrderStatus.PENDING)

    def test_order_constructor_with_quantity(self):
        """Test Order constructor with quantity parameter (keyword-only)"""
        order = Order(
            order_id='TEST002',
            quantity=200,
            **self.base_order_args
        )
        
        self.assertEqual(order.volume, 200)
        self.assertEqual(order.remaining_volume, 200)
        self.assertEqual(order.filled_volume, 0)

    def test_order_constructor_both_volume_and_quantity(self):
        """Test Order constructor when both volume and quantity provided"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            order = Order(
                order_id='TEST003',
                volume=100,
                quantity=200,
                **self.base_order_args
            )
            
            # Should use volume and warn about quantity being ignored
            self.assertEqual(order.volume, 100)
            self.assertTrue(len(w) > 0)
            self.assertIn("Both 'volume' and 'quantity' provided", str(w[0].message))

    def test_order_constructor_neither_volume_nor_quantity(self):
        """Test Order constructor with neither volume nor quantity"""
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST004',
                **self.base_order_args
            )
        
        self.assertIn("either a 'volume' or 'quantity' parameter", str(cm.exception))

    def test_order_validation_negative_volume(self):
        """Test Order validation with negative volume"""
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST005',
                volume=-100,
                **self.base_order_args
            )
        
        self.assertIn("must be positive", str(cm.exception))

    def test_order_validation_zero_volume(self):
        """Test Order validation with zero volume"""
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST006',
                volume=0,
                **self.base_order_args
            )
        
        self.assertIn("must be positive", str(cm.exception))

    def test_order_validation_negative_price(self):
        """Test Order validation with negative price"""
        args = self.base_order_args.copy()
        args['price'] = -50.00
        
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST007',
                volume=100,
                **args
            )
        
        self.assertIn("price must be positive", str(cm.exception))

    def test_order_validation_negative_filled_volume(self):
        """Test Order validation with negative filled volume"""
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST008',
                volume=100,
                filled_volume=-10,
                **self.base_order_args
            )
        
        self.assertIn("cannot be negative", str(cm.exception))

    def test_order_validation_filled_exceeds_total(self):
        """Test Order validation when filled volume exceeds total"""
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST009',
                volume=100,
                filled_volume=150,
                **self.base_order_args
            )
        
        self.assertIn("cannot exceed total volume", str(cm.exception))

    def test_limit_order_without_price(self):
        """Test limit order validation without price"""
        args = self.base_order_args.copy()
        args['order_type'] = OrderType.LIMIT
        args['price'] = None
        
        with self.assertRaises(ValueError) as cm:
            Order(
                order_id='TEST010',
                volume=100,
                **args
            )
        
        self.assertIn("Limit orders must have a price", str(cm.exception))

    def test_market_order_with_price(self):
        """Test market order with price (should be allowed)"""
        args = self.base_order_args.copy()
        args['order_type'] = OrderType.MARKET
        
        order = Order(
            order_id='TEST011',
            volume=100,
            **args
        )
        
        # Should create successfully
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.price, 150.50)

    def test_order_create_market_order_with_volume(self):
        """Test Order.create_market_order with volume"""
        order = Order.create_market_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            volume=100,
            timestamp=self.timestamp
        )
        
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.volume, 100)
        self.assertIsNone(order.price)
        self.assertEqual(order.symbol, 'AAPL')
        self.assertEqual(order.side, OrderSide.BUY)

    def test_order_create_market_order_with_quantity(self):
        """Test Order.create_market_order with quantity parameter"""
        order = Order.create_market_order(
            symbol='AAPL',
            side=OrderSide.SELL,
            quantity=200,
            timestamp=self.timestamp
        )
        
        self.assertEqual(order.volume, 200)
        self.assertEqual(order.order_type, OrderType.MARKET)

    def test_order_create_market_order_without_volume_or_quantity(self):
        """Test Order.create_market_order without volume or quantity"""
        with self.assertRaises(ValueError) as cm:
            Order.create_market_order(
                symbol='AAPL',
                side=OrderSide.BUY,
                timestamp=self.timestamp
            )
        
        self.assertIn("Either 'volume' or 'quantity' parameter must be provided", str(cm.exception))

    def test_order_create_limit_order_with_volume(self):
        """Test Order.create_limit_order with volume"""
        order = Order.create_limit_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            volume=100,
            price=150.50,
            timestamp=self.timestamp
        )
        
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.volume, 100)
        self.assertEqual(order.price, 150.50)

    def test_order_create_limit_order_with_quantity(self):
        """Test Order.create_limit_order with quantity parameter"""
        order = Order.create_limit_order(
            symbol='AAPL',
            side=OrderSide.SELL,
            quantity=200,
            price=151.00,
            timestamp=self.timestamp
        )
        
        self.assertEqual(order.volume, 200)
        self.assertEqual(order.price, 151.00)

    def test_order_create_limit_order_without_price(self):
        """Test Order.create_limit_order without price"""
        with self.assertRaises(ValueError) as cm:
            Order.create_limit_order(
                symbol='AAPL',
                side=OrderSide.BUY,
                volume=100,
                timestamp=self.timestamp
            )
        
        self.assertIn("require a 'price' parameter", str(cm.exception))

    def test_order_is_buy(self):
        """Test order.is_buy() method"""
        buy_order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        bid_order = Order('TEST002', 'AAPL', OrderSide.BID, OrderType.LIMIT, volume=100, price=150.00)
        sell_order = Order('TEST003', 'AAPL', OrderSide.SELL, OrderType.LIMIT, volume=100, price=151.00)
        
        self.assertTrue(buy_order.is_buy())
        self.assertTrue(bid_order.is_buy())
        self.assertFalse(sell_order.is_buy())

    def test_order_is_sell(self):
        """Test order.is_sell() method"""
        sell_order = Order('TEST001', 'AAPL', OrderSide.SELL, OrderType.LIMIT, volume=100, price=151.00)
        ask_order = Order('TEST002', 'AAPL', OrderSide.ASK, OrderType.LIMIT, volume=100, price=151.00)
        buy_order = Order('TEST003', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        self.assertTrue(sell_order.is_sell())
        self.assertTrue(ask_order.is_sell())
        self.assertFalse(buy_order.is_sell())

    def test_order_is_market_order(self):
        """Test order.is_market_order() method"""
        market_order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.MARKET, volume=100)
        limit_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        self.assertTrue(market_order.is_market_order())
        self.assertFalse(limit_order.is_market_order())

    def test_order_is_limit_order(self):
        """Test order.is_limit_order() method"""
        limit_order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        market_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.MARKET, volume=100)
        
        self.assertTrue(limit_order.is_limit_order())
        self.assertFalse(market_order.is_limit_order())

    def test_order_is_active(self):
        """Test order.is_active() method"""
        pending_order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        partial_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00,
                             status=OrderStatus.PARTIAL)
        filled_order = Order('TEST003', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00,
                            status=OrderStatus.FILLED)
        
        self.assertTrue(pending_order.is_active())
        self.assertTrue(partial_order.is_active())
        self.assertFalse(filled_order.is_active())

    def test_order_is_filled(self):
        """Test order.is_filled() method"""
        filled_order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00,
                            status=OrderStatus.FILLED)
        pending_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        # Test by status
        self.assertTrue(filled_order.is_filled())
        self.assertFalse(pending_order.is_filled())
        
        # Test by remaining volume
        order_with_zero_remaining = Order('TEST003', 'AAPL', OrderSide.BUY, OrderType.LIMIT, 
                                         volume=100, price=150.00, filled_volume=100)
        self.assertTrue(order_with_zero_remaining.is_filled())

    def test_order_can_match_price(self):
        """Test order.can_match_price() method"""
        buy_limit = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        sell_limit = Order('TEST002', 'AAPL', OrderSide.SELL, OrderType.LIMIT, volume=100, price=151.00)
        market_order = Order('TEST003', 'AAPL', OrderSide.BUY, OrderType.MARKET, volume=100)
        
        # Buy limit can match at or below limit price
        self.assertTrue(buy_limit.can_match_price(150.00))  # At limit
        self.assertTrue(buy_limit.can_match_price(149.50))  # Below limit
        self.assertFalse(buy_limit.can_match_price(150.50)) # Above limit
        
        # Sell limit can match at or above limit price
        self.assertTrue(sell_limit.can_match_price(151.00))  # At limit
        self.assertTrue(sell_limit.can_match_price(151.50))  # Above limit
        self.assertFalse(sell_limit.can_match_price(150.50)) # Below limit
        
        # Market orders can match at any price
        self.assertTrue(market_order.can_match_price(100.00))
        self.assertTrue(market_order.can_match_price(200.00))

    def test_order_partial_fill(self):
        """Test order.partial_fill() method"""
        order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        # Perform partial fill
        trade = order.partial_fill(30, 150.00)
        
        # Check order state
        self.assertEqual(order.filled_volume, 30)
        self.assertEqual(order.remaining_volume, 70)
        self.assertEqual(order.status, OrderStatus.PARTIAL)
        self.assertEqual(order.average_fill_price, 150.00)
        
        # Check trade
        self.assertEqual(trade.volume, 30)
        self.assertEqual(trade.price, 150.00)
        self.assertEqual(trade.symbol, 'AAPL')
        
        # Complete the fill
        trade2 = order.partial_fill(70, 150.50)
        
        # Check final state
        self.assertEqual(order.filled_volume, 100)
        self.assertEqual(order.remaining_volume, 0)
        self.assertEqual(order.status, OrderStatus.FILLED)
        
        # Check average fill price calculation
        expected_avg = (30 * 150.00 + 70 * 150.50) / 100
        self.assertAlmostEqual(order.average_fill_price, expected_avg, places=4)

    def test_order_partial_fill_invalid_volume(self):
        """Test order.partial_fill() with invalid volume"""
        order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        # Test negative volume
        with self.assertRaises(ValueError):
            order.partial_fill(-10, 150.00)
        
        # Test zero volume
        with self.assertRaises(ValueError):
            order.partial_fill(0, 150.00)
        
        # Test volume exceeding remaining
        with self.assertRaises(ValueError):
            order.partial_fill(150, 150.00)

    def test_order_cancel(self):
        """Test order.cancel() method"""
        order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        # Cancel active order
        order.cancel()
        self.assertEqual(order.status, OrderStatus.CANCELLED)
        self.assertFalse(order.is_active())
        
        # Cancel already filled order (should not change status)
        filled_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00,
                            status=OrderStatus.FILLED)
        filled_order.cancel()
        self.assertEqual(filled_order.status, OrderStatus.FILLED)  # Should remain filled

    def test_order_to_dict(self):
        """Test order.to_dict() method"""
        order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, 
                     price=150.00, timestamp=self.timestamp, metadata={'strategy': 'test'})
        
        order_dict = order.to_dict()
        
        expected_keys = {
            'order_id', 'symbol', 'side', 'order_type', 'price', 'volume',
            'filled_volume', 'remaining_volume', 'status', 'timestamp',
            'average_fill_price', 'source', 'metadata'
        }
        
        self.assertEqual(set(order_dict.keys()), expected_keys)
        self.assertEqual(order_dict['order_id'], 'TEST001')
        self.assertEqual(order_dict['symbol'], 'AAPL')
        self.assertEqual(order_dict['side'], OrderSide.BUY.value)
        self.assertEqual(order_dict['volume'], 100)
        self.assertEqual(order_dict['metadata'], {'strategy': 'test'})

    def test_order_str_and_repr(self):
        """Test order string representations"""
        order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
        
        order_str = str(order)
        self.assertIn('TEST001', order_str)
        self.assertIn('BUY', order_str)
        self.assertIn('100', order_str)
        self.assertIn('AAPL', order_str)
        self.assertIn('150.00', order_str)
        
        # repr should be same as str
        self.assertEqual(str(order), repr(order))


class TestOrderIDRegistry(unittest.TestCase):
    """Test order ID uniqueness validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        clear_order_id_registry()
    
    def tearDown(self):
        """Clean up after tests"""
        clear_order_id_registry()
    
    def test_unique_order_id_validation(self):
        """Test order ID uniqueness validation"""
        # First order should be valid
        result = validate_order_id_uniqueness('ORDER001')
        self.assertTrue(result)
        
        # Duplicate should raise error
        with self.assertRaises(ValueError):
            validate_order_id_uniqueness('ORDER001')
    
    def test_allow_duplicates_in_tests(self):
        """Test allowing duplicates in test environment"""
        # First order
        result1 = validate_order_id_uniqueness('ORDER001')
        self.assertTrue(result1)
        
        # Duplicate with allow flag should return False but not raise
        result2 = validate_order_id_uniqueness('ORDER001', allow_duplicates_in_tests=True)
        self.assertFalse(result2)
    
    def test_remove_order_id_from_registry(self):
        """Test removing order ID from registry"""
        validate_order_id_uniqueness('ORDER001')
        
        # Remove the order ID
        remove_order_id_from_registry('ORDER001')
        
        # Should be able to add again
        result = validate_order_id_uniqueness('ORDER001')
        self.assertTrue(result)
    
    def test_registry_size_tracking(self):
        """Test registry size tracking"""
        self.assertEqual(get_order_id_registry_size(), 0)
        
        validate_order_id_uniqueness('ORDER001')
        self.assertEqual(get_order_id_registry_size(), 1)
        
        validate_order_id_uniqueness('ORDER002')
        self.assertEqual(get_order_id_registry_size(), 2)
        
        remove_order_id_from_registry('ORDER001')
        self.assertEqual(get_order_id_registry_size(), 1)
        
        clear_order_id_registry()
        self.assertEqual(get_order_id_registry_size(), 0)


class TestTrade(unittest.TestCase):
    """Test Trade class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = pd.Timestamp.now()
        self.base_trade_args = {
            'trade_id': 'TRADE001',
            'symbol': 'AAPL',
            'price': 150.50,
            'volume': 100,
            'timestamp': self.timestamp
        }
    
    def test_trade_creation(self):
        """Test basic trade creation"""
        trade = Trade(**self.base_trade_args)
        
        self.assertEqual(trade.trade_id, 'TRADE001')
        self.assertEqual(trade.symbol, 'AAPL')
        self.assertEqual(trade.price, 150.50)
        self.assertEqual(trade.volume, 100)
        self.assertEqual(trade.timestamp, self.timestamp)
    
    def test_trade_validation(self):
        """Test trade parameter validation"""
        # Test negative price
        with self.assertRaises(ValueError):
            args = self.base_trade_args.copy()
            args['price'] = -150.50
            Trade(**args)
        
        # Test zero price
        with self.assertRaises(ValueError):
            args = self.base_trade_args.copy()
            args['price'] = 0.0
            Trade(**args)
        
        # Test negative volume
        with self.assertRaises(ValueError):
            args = self.base_trade_args.copy()
            args['volume'] = -100
            Trade(**args)
        
        # Test zero volume
        with self.assertRaises(ValueError):
            args = self.base_trade_args.copy()
            args['volume'] = 0
            Trade(**args)
    
    def test_trade_value_calculation(self):
        """Test trade value calculation"""
        trade = Trade(**self.base_trade_args)
        expected_value = 150.50 * 100
        self.assertEqual(trade.trade_value, expected_value)
    
    def test_trade_aggressor_detection(self):
        """Test trade aggressor side detection"""
        # Buy aggressor
        buy_trade = Trade(aggressor_side=OrderSide.BUY, **self.base_trade_args)
        self.assertTrue(buy_trade.is_buy_aggressor())
        self.assertFalse(buy_trade.is_sell_aggressor())
        
        # Sell aggressor
        sell_trade = Trade(aggressor_side=OrderSide.SELL, **self.base_trade_args)
        self.assertTrue(sell_trade.is_sell_aggressor())
        self.assertFalse(sell_trade.is_buy_aggressor())
        
        # No aggressor
        no_aggressor_trade = Trade(**self.base_trade_args)
        self.assertFalse(no_aggressor_trade.is_buy_aggressor())
        self.assertFalse(no_aggressor_trade.is_sell_aggressor())
    
    def test_trade_to_dict(self):
        """Test trade to_dict conversion"""
        trade = Trade(
            buy_order_id='BUY001',
            sell_order_id='SELL001',
            aggressor_side=OrderSide.BUY,
            trade_type='opening',
            metadata={'exchange': 'NYSE'},
            **self.base_trade_args
        )
        
        trade_dict = trade.to_dict()
        
        expected_keys = {
            'trade_id', 'symbol', 'price', 'volume', 'trade_value', 'timestamp',
            'buy_order_id', 'sell_order_id', 'aggressor_side', 'trade_type', 'metadata'
        }
        
        self.assertEqual(set(trade_dict.keys()), expected_keys)
        self.assertEqual(trade_dict['buy_order_id'], 'BUY001')
        self.assertEqual(trade_dict['sell_order_id'], 'SELL001')
        self.assertEqual(trade_dict['aggressor_side'], OrderSide.BUY.value)
        self.assertEqual(trade_dict['trade_type'], 'opening')


class TestPriceLevel(unittest.TestCase):
    """Test PriceLevel class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.price_level = PriceLevel(price=150.00)
        self.order = Order('TEST001', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=150.00)
    
    def test_price_level_creation(self):
        """Test PriceLevel creation"""
        level = PriceLevel(price=150.00)
        
        self.assertEqual(level.price, 150.00)
        self.assertEqual(level.total_volume, 0)
        self.assertEqual(level.order_count, 0)
        self.assertEqual(len(level.orders), 0)
    
    def test_add_order_to_level(self):
        """Test adding order to price level"""
        self.price_level.add_order(self.order)
        
        self.assertEqual(self.price_level.total_volume, 100)
        self.assertEqual(self.price_level.order_count, 1)
        self.assertEqual(len(self.price_level.orders), 1)
        self.assertEqual(self.price_level.orders[0], self.order)
    
    def test_add_order_wrong_price(self):
        """Test adding order with wrong price to level"""
        wrong_price_order = Order('TEST002', 'AAPL', OrderSide.BUY, OrderType.LIMIT, volume=100, price=151.00)
        
        with self.assertRaises(ValueError):
            self.price_level.add_order(wrong_price_order)
    
    def test_remove_order_from_level(self):
        """Test removing order from price level"""
        self.price_level.add_order(self.order)
        
        # Remove the order
        removed_order = self.price_level.remove_order('TEST001')
        
        self.assertEqual(removed_order, self.order)
        self.assertEqual(self.price_level.total_volume, 0)
        self.assertEqual(self.price_level.order_count, 0)
        self.assertEqual(len(self.price_level.orders), 0)
    
    def test_remove_nonexistent_order(self):
        """Test removing non-existent order"""
        result = self.price_level.remove_order('NONEXISTENT')
        self.assertIsNone(result)
    
    def test_get_order_from_level(self):
        """Test getting order from price level"""
        self.price_level.add_order(self.order)
        
        retrieved_order = self.price_level.get_order('TEST001')
        self.assertEqual(retrieved_order, self.order)
        
        # Test non-existent order
        result = self.price_level.get_order('NONEXISTENT')
        self.assertIsNone(result)
    
    def test_is_empty(self):
        """Test price level empty check"""
        # Empty level
        self.assertTrue(self.price_level.is_empty())
        
        # Add order
        self.price_level.add_order(self.order)
        self.assertFalse(self.price_level.is_empty())
        
        # Remove order
        self.price_level.remove_order('TEST001')
        self.assertTrue(self.price_level.is_empty())


class TestMarketDataPoint(unittest.TestCase):
    """Test MarketDataPoint class functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = pd.Timestamp.now()
        self.base_data_args = {
            'timestamp': self.timestamp,
            'price': 150.50,
            'volume': 100
        }
    
    def test_market_data_point_creation(self):
        """Test MarketDataPoint creation"""
        data_point = MarketDataPoint(**self.base_data_args)
        
        self.assertEqual(data_point.timestamp, self.timestamp)
        self.assertEqual(data_point.price, 150.50)
        self.assertEqual(data_point.volume, 100)
    
    def test_market_data_point_validation(self):
        """Test MarketDataPoint validation"""
        # Test negative price
        with self.assertRaises(ValueError):
            args = self.base_data_args.copy()
            args['price'] = -150.50
            MarketDataPoint(**args)
        
        # Test zero price
        with self.assertRaises(ValueError):
            args = self.base_data_args.copy()
            args['price'] = 0.0
            MarketDataPoint(**args)
        
        # Test negative volume
        with self.assertRaises(ValueError):
            args = self.base_data_args.copy()
            args['volume'] = -100
            MarketDataPoint(**args)
    
    def test_mid_price_calculation(self):
        """Test mid price calculation"""
        # With bid and ask
        data_point = MarketDataPoint(
            best_bid=150.00,
            best_ask=151.00,
            **self.base_data_args
        )
        self.assertEqual(data_point.mid_price, 150.50)
        
        # Without bid/ask
        data_point_no_quotes = MarketDataPoint(**self.base_data_args)
        self.assertIsNone(data_point_no_quotes.mid_price)
    
    def test_spread_calculation(self):
        """Test spread calculation"""
        data_point = MarketDataPoint(
            best_bid=150.00,
            best_ask=151.00,
            **self.base_data_args
        )
        self.assertEqual(data_point.spread, 1.00)
        
        # Without bid/ask
        data_point_no_quotes = MarketDataPoint(**self.base_data_args)
        self.assertIsNone(data_point_no_quotes.spread)
    
    def test_order_imbalance_calculation(self):
        """Test order imbalance calculation"""
        # More bid volume
        data_point = MarketDataPoint(
            bid_volume=300,
            ask_volume=200,
            **self.base_data_args
        )
        expected_imbalance = (300 - 200) / (300 + 200)
        self.assertEqual(data_point.order_imbalance, expected_imbalance)
        
        # More ask volume
        data_point_ask_heavy = MarketDataPoint(
            bid_volume=200,
            ask_volume=300,
            **self.base_data_args
        )
        expected_imbalance_ask = (200 - 300) / (200 + 300)
        self.assertEqual(data_point_ask_heavy.order_imbalance, expected_imbalance_ask)
        
        # Zero volumes
        data_point_zero = MarketDataPoint(
            bid_volume=0,
            ask_volume=0,
            **self.base_data_args
        )
        self.assertIsNone(data_point_zero.order_imbalance)


if __name__ == '__main__':
    unittest.main()
