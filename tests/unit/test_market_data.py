"""
Comprehensive Unit Tests for Market Data Components

This module provides extensive testing for:
- BookSnapshot normalization of price levels
- MarketData update and retrieval methods
- Market data calculations and analysis
- Edge cases and error handling
"""

import unittest
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime

from src.engine.market_data import BookSnapshot, MarketData, calculate_microprice, calculate_effective_spread
from src.engine.order_types import PriceLevel, Order, Trade
from src.utils.constants import OrderSide, OrderType


class TestBookSnapshot(unittest.TestCase):
    """Comprehensive tests for BookSnapshot class"""

    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = pd.Timestamp.now()
        self.symbol = 'AAPL'

    def test_book_snapshot_basic_creation(self):
        """Test basic BookSnapshot creation"""
        bid_level = PriceLevel(price=150.00, total_volume=100)
        ask_level = PriceLevel(price=151.00, total_volume=200)
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[bid_level],
            asks=[ask_level]
        )
        
        self.assertEqual(snapshot.symbol, self.symbol)
        self.assertEqual(snapshot.timestamp, self.timestamp)
        self.assertEqual(len(snapshot.bids), 1)
        self.assertEqual(len(snapshot.asks), 1)
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_ask, 151.00)

    def test_book_snapshot_create_from_best_quotes(self):
        """Test BookSnapshot creation from best quotes"""
        snapshot = BookSnapshot.create_from_best_quotes(
            symbol=self.symbol,
            timestamp=self.timestamp,
            best_bid=150.00,
            best_bid_volume=100,
            best_ask=151.00,
            best_ask_volume=200,
            sequence_number=1
        )
        
        self.assertEqual(snapshot.symbol, self.symbol)
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_ask, 151.00)
        self.assertEqual(snapshot.best_bid_volume, 100)
        self.assertEqual(snapshot.best_ask_volume, 200)
        self.assertEqual(snapshot.sequence_number, 1)

    def test_book_snapshot_normalization_price_level_objects(self):
        """Test BookSnapshot normalization with PriceLevel objects"""
        bid_levels = [
            PriceLevel(price=150.00, total_volume=100),
            PriceLevel(price=149.50, total_volume=200)
        ]
        ask_levels = [
            PriceLevel(price=151.00, total_volume=150),
            PriceLevel(price=151.50, total_volume=250)
        ]
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=bid_levels,
            asks=ask_levels
        )
        
        # Should have normalized and sorted properly
        self.assertEqual(snapshot.best_bid, 150.00)  # Highest bid first
        self.assertEqual(snapshot.best_ask, 151.00)  # Lowest ask first
        self.assertEqual(len(snapshot.bids), 2)
        self.assertEqual(len(snapshot.asks), 2)

    def test_book_snapshot_normalization_tuple_format(self):
        """Test BookSnapshot normalization with (price, volume) tuples"""
        bid_tuples = [(150.00, 100), (149.50, 200), (149.00, 150)]
        ask_tuples = [(151.00, 150), (151.50, 250), (152.00, 100)]
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=bid_tuples,
            asks=ask_tuples
        )
        
        # Should normalize tuples to PriceLevel objects
        self.assertIsInstance(snapshot.bids[0], PriceLevel)
        self.assertIsInstance(snapshot.asks[0], PriceLevel)
        
        # Check sorting: bids descending, asks ascending
        self.assertEqual(snapshot.bids[0].price, 150.00)  # Highest first
        self.assertEqual(snapshot.bids[1].price, 149.50)
        self.assertEqual(snapshot.bids[2].price, 149.00)  # Lowest last
        
        self.assertEqual(snapshot.asks[0].price, 151.00)  # Lowest first
        self.assertEqual(snapshot.asks[1].price, 151.50)
        self.assertEqual(snapshot.asks[2].price, 152.00)  # Highest last

    def test_book_snapshot_normalization_mixed_formats(self):
        """Test BookSnapshot normalization with mixed PriceLevel and tuple formats"""
        bid_mixed = [
            PriceLevel(price=150.00, total_volume=100),
            (149.50, 200),  # Tuple format
            PriceLevel(price=149.00, total_volume=150)
        ]
        ask_mixed = [
            (151.00, 150),  # Tuple format
            PriceLevel(price=151.50, total_volume=250),
            (152.00, 100)   # Tuple format
        ]
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=bid_mixed,
            asks=ask_mixed
        )
        
        # All should be normalized to PriceLevel objects
        for level in snapshot.bids:
            self.assertIsInstance(level, PriceLevel)
        for level in snapshot.asks:
            self.assertIsInstance(level, PriceLevel)
        
        # Check proper sorting
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_ask, 151.00)

    def test_book_snapshot_best_quotes_properties(self):
        """Test BookSnapshot best quote properties"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200)],
            asks=[(151.00, 150), (151.50, 250)]
        )
        
        self.assertEqual(snapshot.best_bid, 150.00)
        self.assertEqual(snapshot.best_ask, 151.00)
        self.assertEqual(snapshot.best_bid_volume, 100)
        self.assertEqual(snapshot.best_ask_volume, 150)

    def test_book_snapshot_empty_sides(self):
        """Test BookSnapshot with empty bid or ask sides"""
        # Only bids
        snapshot_bids_only = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[]
        )
        
        self.assertEqual(snapshot_bids_only.best_bid, 150.00)
        self.assertIsNone(snapshot_bids_only.best_ask)
        self.assertIsNone(snapshot_bids_only.spread)
        self.assertIsNone(snapshot_bids_only.mid_price)
        
        # Only asks
        snapshot_asks_only = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[],
            asks=[(151.00, 150)]
        )
        
        self.assertIsNone(snapshot_asks_only.best_bid)
        self.assertEqual(snapshot_asks_only.best_ask, 151.00)
        self.assertIsNone(snapshot_asks_only.spread)
        self.assertIsNone(snapshot_asks_only.mid_price)

    def test_book_snapshot_calculations(self):
        """Test BookSnapshot price and volume calculations"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200), (149.00, 150)],
            asks=[(151.00, 150), (151.50, 250), (152.00, 100)]
        )
        
        # Mid price
        expected_mid = (150.00 + 151.00) / 2
        self.assertEqual(snapshot.mid_price, expected_mid)
        
        # Spread
        expected_spread = 151.00 - 150.00
        self.assertEqual(snapshot.spread, expected_spread)
        
        # Spread in basis points
        expected_spread_bps = (expected_spread / expected_mid) * 10000
        self.assertAlmostEqual(snapshot.spread_bps, expected_spread_bps, places=4)
        
        # Total volumes
        self.assertEqual(snapshot.total_bid_volume, 450)  # 100 + 200 + 150
        self.assertEqual(snapshot.total_ask_volume, 500)  # 150 + 250 + 100
        
        # Order book imbalance
        expected_imbalance = (450 - 500) / (450 + 500)
        self.assertEqual(snapshot.order_book_imbalance, expected_imbalance)

    def test_book_snapshot_market_conditions(self):
        """Test BookSnapshot market condition detection"""
        # Normal market
        normal_snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        
        self.assertFalse(normal_snapshot.is_crossed())
        self.assertFalse(normal_snapshot.is_locked())
        
        # Crossed market (bid >= ask)
        crossed_snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(151.00, 100)],
            asks=[(150.00, 150)]
        )
        
        self.assertTrue(crossed_snapshot.is_crossed())
        
        # Locked market (bid == ask)
        locked_snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.50, 100)],
            asks=[(150.50, 150)]
        )
        
        self.assertTrue(locked_snapshot.is_locked())
        self.assertTrue(locked_snapshot.is_crossed())  # Locked is also crossed

    def test_book_snapshot_depth_retrieval(self):
        """Test BookSnapshot depth retrieval"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200), (149.00, 150)],
            asks=[(151.00, 150), (151.50, 250), (152.00, 100)]
        )
        
        # Get bid depth
        bid_depth = snapshot.get_depth(OrderSide.BID, 2)
        expected_bid_depth = [(150.00, 100), (149.50, 200)]
        self.assertEqual(bid_depth, expected_bid_depth)
        
        # Get ask depth
        ask_depth = snapshot.get_depth(OrderSide.ASK, 2)
        expected_ask_depth = [(151.00, 150), (151.50, 250)]
        self.assertEqual(ask_depth, expected_ask_depth)
        
        # Test with BUY/SELL aliases
        buy_depth = snapshot.get_depth(OrderSide.BUY, 1)
        sell_depth = snapshot.get_depth(OrderSide.SELL, 1)
        
        self.assertEqual(buy_depth, [(150.00, 100)])
        self.assertEqual(sell_depth, [(151.00, 150)])

    def test_book_snapshot_volume_at_price(self):
        """Test getting volume at specific price"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200)],
            asks=[(151.00, 150), (151.50, 250)]
        )
        
        # Test exact price matches
        self.assertEqual(snapshot.get_volume_at_price(150.00, OrderSide.BID), 100)
        self.assertEqual(snapshot.get_volume_at_price(149.50, OrderSide.BID), 200)
        self.assertEqual(snapshot.get_volume_at_price(151.00, OrderSide.ASK), 150)
        self.assertEqual(snapshot.get_volume_at_price(151.50, OrderSide.ASK), 250)
        
        # Test non-existent prices
        self.assertEqual(snapshot.get_volume_at_price(148.00, OrderSide.BID), 0)
        self.assertEqual(snapshot.get_volume_at_price(153.00, OrderSide.ASK), 0)

    def test_book_snapshot_market_impact(self):
        """Test market impact calculation"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200), (149.00, 150)],
            asks=[(151.00, 150), (151.50, 250), (152.00, 100)]
        )
        
        # Buy order (consumes ask liquidity)
        avg_price, total_cost = snapshot.get_market_impact(OrderSide.BUY, 300)
        
        # Should consume: 150@151.00 + 150@151.50 = total 300
        expected_cost = 150 * 151.00 + 150 * 151.50
        expected_avg = expected_cost / 300
        
        self.assertAlmostEqual(total_cost, expected_cost, places=2)
        self.assertAlmostEqual(avg_price, expected_avg, places=4)
        
        # Sell order (consumes bid liquidity)
        avg_price_sell, total_cost_sell = snapshot.get_market_impact(OrderSide.SELL, 250)
        
        # Should consume: 100@150.00 + 150@149.50 = total 250
        expected_cost_sell = 100 * 150.00 + 150 * 149.50
        expected_avg_sell = expected_cost_sell / 250
        
        self.assertAlmostEqual(total_cost_sell, expected_cost_sell, places=2)
        self.assertAlmostEqual(avg_price_sell, expected_avg_sell, places=4)

    def test_book_snapshot_market_impact_insufficient_liquidity(self):
        """Test market impact with insufficient liquidity"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        
        # Request more than available
        avg_price, total_cost = snapshot.get_market_impact(OrderSide.BUY, 200)
        
        # Should consume all 150 at 151.00, then 50 more at 151.00 (last price)
        expected_cost = 200 * 151.00
        expected_avg = 151.00
        
        self.assertEqual(total_cost, expected_cost)
        self.assertEqual(avg_price, expected_avg)

    def test_book_snapshot_to_dict(self):
        """Test BookSnapshot to_dict conversion"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200)],
            asks=[(151.00, 150), (151.50, 250)],
            last_trade_price=150.75,
            last_trade_volume=50,
            sequence_number=123
        )
        
        snapshot_dict = snapshot.to_dict(depth=2)
        
        # Check basic fields
        self.assertEqual(snapshot_dict['symbol'], self.symbol)
        self.assertEqual(snapshot_dict['sequence_number'], 123)
        self.assertEqual(snapshot_dict['best_bid'], 150.00)
        self.assertEqual(snapshot_dict['best_ask'], 151.00)
        self.assertEqual(snapshot_dict['last_trade_price'], 150.75)
        
        # Check depth levels
        self.assertEqual(snapshot_dict['bid_price_1'], 150.00)
        self.assertEqual(snapshot_dict['bid_volume_1'], 100)
        self.assertEqual(snapshot_dict['bid_price_2'], 149.50)
        self.assertEqual(snapshot_dict['bid_volume_2'], 200)
        
        self.assertEqual(snapshot_dict['ask_price_1'], 151.00)
        self.assertEqual(snapshot_dict['ask_volume_1'], 150)
        self.assertEqual(snapshot_dict['ask_price_2'], 151.50)
        self.assertEqual(snapshot_dict['ask_volume_2'], 250)

    def test_book_snapshot_string_representation(self):
        """Test BookSnapshot string representation"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        
        snapshot_str = str(snapshot)
        
        self.assertIn(self.symbol, snapshot_str)
        self.assertIn('150.00', snapshot_str)
        self.assertIn('151.00', snapshot_str)
        self.assertIn('100', snapshot_str)
        self.assertIn('150', snapshot_str)


class TestMarketData(unittest.TestCase):
    """Comprehensive tests for MarketData class"""

    def setUp(self):
        """Set up test fixtures"""
        self.symbol = 'AAPL'
        self.timestamp = pd.Timestamp.now()
        self.market_data = MarketData(symbol=self.symbol)

    def test_market_data_initialization(self):
        """Test MarketData initialization"""
        self.assertEqual(self.market_data.symbol, self.symbol)
        self.assertIsNone(self.market_data.current_snapshot)
        self.assertEqual(len(self.market_data.snapshots_history), 0)
        self.assertEqual(len(self.market_data.trades_history), 0)
        self.assertEqual(self.market_data.daily_volume, 0)
        self.assertEqual(self.market_data.daily_trade_count, 0)
        self.assertEqual(self.market_data.update_count, 0)

    def test_market_data_update_snapshot(self):
        """Test MarketData snapshot updates"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)],
            last_trade_price=150.50,
            last_trade_volume=50
        )
        
        # Update snapshot
        self.market_data.update_snapshot(snapshot)
        
        # Check update
        self.assertEqual(self.market_data.current_snapshot, snapshot)
        self.assertEqual(self.market_data.last_update_timestamp, self.timestamp)
        self.assertEqual(self.market_data.update_count, 1)
        self.assertEqual(len(self.market_data.snapshots_history), 1)
        
        # Check daily stats update
        self.assertEqual(self.market_data.daily_close, 150.50)
        self.assertEqual(self.market_data.daily_open, 150.50)
        self.assertEqual(self.market_data.daily_high, 150.50)
        self.assertEqual(self.market_data.daily_low, 150.50)
        self.assertEqual(self.market_data.daily_volume, 50)

    def test_market_data_snapshot_history_limit(self):
        """Test MarketData snapshot history maintains limit"""
        # Add many snapshots
        for i in range(1500):  # More than the 1000 limit
            snapshot = BookSnapshot(
                symbol=self.symbol,
                timestamp=self.timestamp + pd.Timedelta(seconds=i),
                bids=[(150.00 + i*0.01, 100)],
                asks=[(151.00 + i*0.01, 150)]
            )
            self.market_data.update_snapshot(snapshot)
        
        # Should maintain limit
        self.assertEqual(len(self.market_data.snapshots_history), 1000)
        self.assertEqual(self.market_data.update_count, 1500)

    def test_market_data_add_trade(self):
        """Test MarketData trade addition"""
        trade_data = {
            'trade_id': 'T001',
            'price': 150.75,
            'volume': 100,
            'timestamp': self.timestamp,
            'side': 'buy'
        }
        
        self.market_data.add_trade(trade_data)
        
        # Check trade addition
        self.assertEqual(len(self.market_data.trades_history), 1)
        self.assertEqual(self.market_data.daily_trade_count, 1)
        self.assertEqual(self.market_data.trades_history[0], trade_data)
        
        # Check daily stats update
        self.assertEqual(self.market_data.daily_close, 150.75)
        self.assertEqual(self.market_data.daily_volume, 100)

    def test_market_data_trade_history_limit(self):
        """Test MarketData trade history maintains limit"""
        # Add many trades
        for i in range(1500):  # More than the 1000 limit
            trade_data = {
                'trade_id': f'T{i:03d}',
                'price': 150.00 + i * 0.01,
                'volume': 100,
                'timestamp': self.timestamp + pd.Timedelta(seconds=i)
            }
            self.market_data.add_trade(trade_data)
        
        # Should maintain limit
        self.assertEqual(len(self.market_data.trades_history), 1000)
        self.assertEqual(self.market_data.daily_trade_count, 1500)

    def test_market_data_daily_stats_updates(self):
        """Test MarketData daily statistics updates"""
        # Add trades with different prices
        trades = [
            {'price': 150.00, 'volume': 100},
            {'price': 152.00, 'volume': 200},  # New high
            {'price': 149.00, 'volume': 150},  # New low
            {'price': 151.00, 'volume': 300}   # Final price
        ]
        
        for i, trade in enumerate(trades):
            trade['trade_id'] = f'T{i:03d}'
            trade['timestamp'] = self.timestamp + pd.Timedelta(seconds=i)
            self.market_data.add_trade(trade)
        
        # Check daily statistics
        self.assertEqual(self.market_data.daily_open, 150.00)    # First trade
        self.assertEqual(self.market_data.daily_close, 151.00)   # Last trade
        self.assertEqual(self.market_data.daily_high, 152.00)    # Highest price
        self.assertEqual(self.market_data.daily_low, 149.00)     # Lowest price
        self.assertEqual(self.market_data.daily_volume, 750)     # Total volume
        self.assertEqual(self.market_data.daily_trade_count, 4)

    def test_market_data_current_price_from_snapshot(self):
        """Test MarketData current price from snapshot"""
        # Snapshot with mid price
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        
        self.market_data.update_snapshot(snapshot)
        expected_mid_price = (150.00 + 151.00) / 2
        self.assertEqual(self.market_data.current_price, expected_mid_price)
        
        # Snapshot with last trade price (no mid price)
        snapshot_no_quotes = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[],
            asks=[],
            last_trade_price=150.75
        )
        
        self.market_data.update_snapshot(snapshot_no_quotes)
        self.assertEqual(self.market_data.current_price, 150.75)

    def test_market_data_daily_return_calculation(self):
        """Test MarketData daily return calculation"""
        # No trades yet
        self.assertIsNone(self.market_data.daily_return)
        
        # Add opening trade
        self.market_data.add_trade({
            'trade_id': 'T001',
            'price': 150.00,
            'volume': 100,
            'timestamp': self.timestamp
        })
        
        # Add closing trade
        self.market_data.add_trade({
            'trade_id': 'T002',
            'price': 153.00,  # 2% gain
            'volume': 100,
            'timestamp': self.timestamp + pd.Timedelta(hours=1)
        })
        
        expected_return = ((153.00 - 150.00) / 150.00) * 100
        self.assertAlmostEqual(self.market_data.daily_return, expected_return, places=4)

    def test_market_data_daily_range_calculation(self):
        """Test MarketData daily range calculation"""
        # Add trades with different prices
        self.market_data.add_trade({'price': 150.00, 'volume': 100, 'trade_id': 'T1', 'timestamp': self.timestamp})
        self.market_data.add_trade({'price': 155.00, 'volume': 100, 'trade_id': 'T2', 'timestamp': self.timestamp})
        self.market_data.add_trade({'price': 148.00, 'volume': 100, 'trade_id': 'T3', 'timestamp': self.timestamp})
        
        expected_range = 155.00 - 148.00
        self.assertEqual(self.market_data.daily_range, expected_range)

    def test_market_data_recent_snapshots(self):
        """Test MarketData recent snapshots retrieval"""
        # Add multiple snapshots
        for i in range(15):
            snapshot = BookSnapshot(
                symbol=self.symbol,
                timestamp=self.timestamp + pd.Timedelta(seconds=i),
                bids=[(150.00 + i*0.01, 100)],
                asks=[(151.00 + i*0.01, 150)]
            )
            self.market_data.update_snapshot(snapshot)
        
        # Get recent snapshots
        recent_5 = self.market_data.get_recent_snapshots(5)
        recent_10 = self.market_data.get_recent_snapshots(10)
        
        self.assertEqual(len(recent_5), 5)
        self.assertEqual(len(recent_10), 10)
        
        # Should be the most recent ones
        self.assertEqual(recent_5[-1].bids[0].price, 150.14)  # Last snapshot

    def test_market_data_recent_trades(self):
        """Test MarketData recent trades retrieval"""
        # Add multiple trades
        for i in range(15):
            trade_data = {
                'trade_id': f'T{i:03d}',
                'price': 150.00 + i * 0.01,
                'volume': 100,
                'timestamp': self.timestamp + pd.Timedelta(seconds=i)
            }
            self.market_data.add_trade(trade_data)
        
        # Get recent trades
        recent_5 = self.market_data.get_recent_trades(5)
        recent_10 = self.market_data.get_recent_trades(10)
        
        self.assertEqual(len(recent_5), 5)
        self.assertEqual(len(recent_10), 10)
        
        # Should be the most recent ones
        self.assertEqual(recent_5[-1]['trade_id'], 'T014')  # Last trade

    def test_market_data_price_history(self):
        """Test MarketData price history extraction"""
        # Add snapshots with mid prices
        for i in range(10):
            snapshot = BookSnapshot(
                symbol=self.symbol,
                timestamp=self.timestamp + pd.Timedelta(seconds=i),
                bids=[(150.00 + i*0.01, 100)],
                asks=[(151.00 + i*0.01, 150)]
            )
            self.market_data.update_snapshot(snapshot)
        
        price_history = self.market_data.get_price_history(5)
        
        self.assertEqual(len(price_history), 5)
        # Should contain mid prices from recent snapshots
        expected_last_mid = (150.09 + 151.09) / 2
        self.assertAlmostEqual(price_history[-1], expected_last_mid, places=4)

    def test_market_data_spread_history(self):
        """Test MarketData spread history extraction"""
        # Add snapshots with different spreads
        for i in range(10):
            bid_price = 150.00 + i*0.01
            ask_price = bid_price + 0.05 + i*0.001  # Varying spread
            
            snapshot = BookSnapshot(
                symbol=self.symbol,
                timestamp=self.timestamp + pd.Timedelta(seconds=i),
                bids=[(bid_price, 100)],
                asks=[(ask_price, 150)]
            )
            self.market_data.update_snapshot(snapshot)
        
        spread_history = self.market_data.get_spread_history(5)
        
        self.assertEqual(len(spread_history), 5)
        # Last spread should be around 0.059 (0.05 + 9*0.001)
        self.assertAlmostEqual(spread_history[-1], 0.059, places=3)

    def test_market_data_volume_profile(self):
        """Test MarketData volume profile extraction"""
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100), (149.50, 200), (149.00, 150)],
            asks=[(151.00, 150), (151.50, 250), (152.00, 100)]
        )
        
        self.market_data.update_snapshot(snapshot)
        
        # Get bid volume profile
        bid_profile = self.market_data.get_volume_profile(OrderSide.BID, 3)
        expected_bid_profile = {150.00: 100, 149.50: 200, 149.00: 150}
        self.assertEqual(bid_profile, expected_bid_profile)
        
        # Get ask volume profile
        ask_profile = self.market_data.get_volume_profile(OrderSide.ASK, 2)
        expected_ask_profile = {151.00: 150, 151.50: 250}
        self.assertEqual(ask_profile, expected_ask_profile)

    def test_market_data_vwap_calculation(self):
        """Test MarketData VWAP calculation"""
        # Add trades for VWAP calculation
        trades = [
            {'price': 150.00, 'volume': 100},
            {'price': 151.00, 'volume': 200},
            {'price': 149.00, 'volume': 300}
        ]
        
        for i, trade in enumerate(trades):
            trade['trade_id'] = f'T{i:03d}'
            trade['timestamp'] = self.timestamp + pd.Timedelta(seconds=i)
            self.market_data.add_trade(trade)
        
        vwap = self.market_data.calculate_vwap()
        
        # VWAP = (150*100 + 151*200 + 149*300) / (100 + 200 + 300)
        expected_vwap = (150*100 + 151*200 + 149*300) / 600
        self.assertAlmostEqual(vwap, expected_vwap, places=4)

    def test_market_data_vwap_no_trades(self):
        """Test MarketData VWAP calculation with no trades"""
        vwap = self.market_data.calculate_vwap()
        self.assertIsNone(vwap)

    def test_market_data_to_dict(self):
        """Test MarketData to_dict conversion"""
        # Add some data
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        self.market_data.update_snapshot(snapshot)
        
        self.market_data.add_trade({
            'trade_id': 'T001',
            'price': 150.50,
            'volume': 200,
            'timestamp': self.timestamp
        })
        
        market_data_dict = self.market_data.to_dict()
        
        # Check basic fields
        self.assertEqual(market_data_dict['symbol'], self.symbol)
        self.assertEqual(market_data_dict['current_price'], 150.50)
        self.assertEqual(market_data_dict['daily_volume'], 200)
        self.assertEqual(market_data_dict['daily_trade_count'], 1)
        self.assertEqual(market_data_dict['update_count'], 1)
        
        # Should include current snapshot
        self.assertIsNotNone(market_data_dict['current_snapshot'])
        
        # Should include VWAP
        self.assertIsNotNone(market_data_dict['vwap'])

    def test_market_data_string_representation(self):
        """Test MarketData string representation"""
        # Add some data
        self.market_data.daily_volume = 1000
        self.market_data.update_count = 5
        
        snapshot = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(150.00, 100)],
            asks=[(151.00, 150)]
        )
        self.market_data.update_snapshot(snapshot)
        
        market_data_str = str(self.market_data)
        
        self.assertIn(self.symbol, market_data_str)
        self.assertIn('150.50', market_data_str)  # Mid price
        self.assertIn('1,000', market_data_str)    # Volume (formatted with comma)
        self.assertIn('6', market_data_str)       # Update count (5 + 1 from snapshot)


class TestMarketDataUtilities(unittest.TestCase):
    """Test market data utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.timestamp = pd.Timestamp.now()
        self.snapshot = BookSnapshot(
            symbol='AAPL',
            timestamp=self.timestamp,
            bids=[(150.00, 300)],
            asks=[(151.00, 200)]
        )

    def test_calculate_microprice_default(self):
        """Test microprice calculation with default parameters"""
        microprice = calculate_microprice(self.snapshot)
        
        # With alpha=0.5: microprice = (0.5 * ask * bid_vol + 0.5 * bid * ask_vol) / (bid_vol + ask_vol)
        # = (0.5 * 151 * 300 + 0.5 * 150 * 200) / (300 + 200)
        expected = (0.5 * 151 * 300 + 0.5 * 150 * 200) / 500
        self.assertAlmostEqual(microprice, expected, places=4)

    def test_calculate_microprice_custom_alpha(self):
        """Test microprice calculation with custom alpha"""
        alpha = 0.3
        microprice = calculate_microprice(self.snapshot, alpha)
        
        expected = (alpha * 151 * 300 + (1-alpha) * 150 * 200) / 500
        self.assertAlmostEqual(microprice, expected, places=4)

    def test_calculate_microprice_no_quotes(self):
        """Test microprice calculation with missing quotes"""
        empty_snapshot = BookSnapshot(
            symbol='AAPL',
            timestamp=self.timestamp,
            bids=[],
            asks=[]
        )
        
        microprice = calculate_microprice(empty_snapshot)
        self.assertIsNone(microprice)

    def test_calculate_microprice_zero_volumes(self):
        """Test microprice calculation with zero volumes"""
        zero_vol_snapshot = BookSnapshot(
            symbol='AAPL',
            timestamp=self.timestamp,
            bids=[(150.00, 0)],
            asks=[(151.00, 0)]
        )
        
        microprice = calculate_microprice(zero_vol_snapshot)
        # Should fall back to mid price
        expected_mid = (150.00 + 151.00) / 2
        self.assertEqual(microprice, expected_mid)

    def test_calculate_effective_spread(self):
        """Test effective spread calculation"""
        trade_price = 150.75  # Between bid and ask
        
        effective_spread = calculate_effective_spread(trade_price, self.snapshot)
        
        # Mid price = (150.00 + 151.00) / 2 = 150.50
        # Effective spread = 2 * |150.75 - 150.50| = 2 * 0.25 = 0.50
        expected = 2 * abs(trade_price - 150.50)
        self.assertEqual(effective_spread, expected)

    def test_calculate_effective_spread_no_mid_price(self):
        """Test effective spread calculation without mid price"""
        empty_snapshot = BookSnapshot(
            symbol='AAPL',
            timestamp=self.timestamp,
            bids=[],
            asks=[]
        )
        
        effective_spread = calculate_effective_spread(150.75, empty_snapshot)
        self.assertIsNone(effective_spread)


if __name__ == '__main__':
    unittest.main()
