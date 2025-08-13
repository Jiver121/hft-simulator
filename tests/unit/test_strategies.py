"""
Unit Tests for Trading Strategies

This module contains comprehensive unit tests for all trading strategies,
testing strategy logic, signal generation, and risk management.

Educational Notes:
- Strategy tests verify correct signal generation and decision making
- Risk management components are tested for proper limit enforcement
- Performance tracking and metrics calculation are validated
- Edge cases and market conditions are thoroughly tested
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from src.strategies.liquidity_taking import LiquidityTakingStrategy, LiquidityTakingConfig
from src.strategies.strategy_utils import RiskManager
from src.engine.order_book import OrderBook
from src.engine.order_types import Order, OrderSide, OrderType, Trade, PriceLevel
from src.engine.market_data import BookSnapshot
from src.performance.portfolio import Portfolio


class TestBaseStrategy(unittest.TestCase):
    """Test cases for BaseStrategy class"""

    def setUp(self):
        """Set up test fixtures"""
        self.symbol = "AAPL"
        self.initial_capital = 100000.0

        # Create mock portfolio
        self.portfolio = Mock(spec=Portfolio)
        self.portfolio.current_cash = self.initial_capital
        self.portfolio.total_value = self.initial_capital

        # Create concrete strategy for testing
        class TestStrategy(BaseStrategy):
            def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> StrategyResult:
                return StrategyResult()

        self.strategy = TestStrategy(
            strategy_name="TestStrategy",
            symbol=self.symbol,
        )

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.strategy_name, "TestStrategy")
        self.assertEqual(self.strategy.symbol, self.symbol)
        self.assertTrue(self.strategy.is_active)

    def test_start_stop_strategy(self):
        """Test strategy start/stop functionality"""
        # Stop strategy
        self.strategy.set_active(False)
        self.assertFalse(self.strategy.is_active)

        # Start strategy
        self.strategy.set_active(True)
        self.assertTrue(self.strategy.is_active)

    def test_position_tracking(self):
        """Test position tracking functionality"""
        # Initially no positions
        self.assertEqual(self.strategy.current_position, 0)

        # Add position
        self.strategy.current_position = 100
        self.assertEqual(self.strategy.current_position, 100)

    def test_pnl_calculation(self):
        """Test P&L calculation"""
        # Set up position and prices
        self.strategy.current_position = 100
        self.strategy.average_price = 150.0

        # Calculate P&L with current price
        current_price = 155.0
        pnl = self.strategy._calculate_unrealized_pnl(current_price)
        expected_pnl = 100 * (155.0 - 150.0)
        self.assertEqual(pnl, expected_pnl)

    def test_risk_limits(self):
        """Test risk limit enforcement"""
        # Set position limit
        self.strategy.max_position_size = 1000
        
        order = self.strategy.create_order(OrderSide.BUY, 500)

        # Test within limits
        self.assertTrue(self.strategy.check_risk_limits(order, 150.0))

        # Test exceeding limits
        order.volume = 1500
        self.assertFalse(self.strategy.check_risk_limits(order, 150.0)[0])


class TestMarketMakingStrategy(unittest.TestCase):
    """Test cases for MarketMakingStrategy class"""

    def setUp(self):
        """Set up test fixtures"""
        self.symbol = "AAPL"
        self.config = MarketMakingConfig(
            target_spread=0.02,
            inventory_target=0,
        )

        # Create mock portfolio
        self.portfolio = Mock(spec=Portfolio)
        self.portfolio.current_cash = 100000.0
        self.portfolio.total_value = 100000.0

        self.strategy = MarketMakingStrategy(
            symbol=self.symbol,
            config=self.config
        )

        # Create test market data
        self.timestamp = pd.Timestamp.now()
        self.market_data = self._create_book_snapshot()

    def _create_book_snapshot(self) -> BookSnapshot:
        """Create test book snapshot"""
        return BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[PriceLevel(price=149.95, total_volume=500), PriceLevel(price=149.90, total_volume=300), PriceLevel(price=149.85, total_volume=200)],
            asks=[PriceLevel(price=150.05, total_volume=500), PriceLevel(price=150.10, total_volume=300), PriceLevel(price=150.15, total_volume=200)]
        )

    def test_fair_value_estimation(self):
        """Test fair value estimation"""
        fair_value = self.strategy._estimate_fair_value(self.market_data)

        # Should be close to mid price
        expected_mid = (149.95 + 150.05) / 2
        self.assertAlmostEqual(fair_value, expected_mid, places=2)

    def test_spread_calculation(self):
        """Test spread calculation"""
        fair_value = 150.0
        spread = self.strategy._calculate_optimal_spread(self.market_data, fair_value)

        # Should be at least the target spread
        self.assertGreaterEqual(spread, self.config.spread_target)

    def test_inventory_skewing(self):
        """Test inventory skewing logic"""
        fair_value = 150.0
        spread = 0.10

        # Test with no inventory
        self.strategy.current_position = 0
        bid_price, ask_price, _, _ = self.strategy._calculate_quotes(fair_value, spread)

        expected_bid = fair_value - spread / 2
        expected_ask = fair_value + spread / 2

        self.assertAlmostEqual(bid_price, expected_bid, places=2)
        self.assertAlmostEqual(ask_price, expected_ask, places=2)

        # Test with long inventory (should skew quotes down)
        self.strategy.current_position = 500
        bid_price_long, ask_price_long, _, _ = self.strategy._calculate_quotes(fair_value, spread)

        self.assertLess(bid_price_long, expected_bid)
        self.assertLess(ask_price_long, expected_ask)

    def test_signal_generation(self):
        """Test signal generation"""
        self.strategy.set_active(True)

        # Generate signals
        result = self.strategy.on_market_update(self.market_data, self.timestamp)

        # Should generate both bid and ask orders
        self.assertEqual(len(result.orders), 2)

        bid_order = next((o for o in result.orders if o.side == OrderSide.BUY), None)
        ask_order = next((o for o in result.orders if o.side == OrderSide.SELL), None)

        self.assertIsNotNone(bid_order)
        self.assertIsNotNone(ask_order)

        # Bid should be below fair value, ask above
        fair_value = self.strategy._estimate_fair_value(self.market_data)
        self.assertLess(bid_order.price, fair_value)
        self.assertGreater(ask_order.price, fair_value)

    def test_adverse_selection_protection(self):
        """Test adverse selection protection"""
        # Simulate rapid price movement
        self.strategy.recent_trades = [
            Trade("T1", self.symbol, 150.10, 100, self.timestamp, "BUY1", "SELL1"),
            Trade("T2", self.symbol, 150.15, 200, self.timestamp, "BUY2", "SELL2"),
            Trade("T3", self.symbol, 150.20, 150, self.timestamp, "BUY3", "SELL3")
        ]

        # Should detect adverse selection
        adverse_selection = self.strategy._detect_adverse_selection(self.market_data)
        self.assertTrue(adverse_selection)

        # Should widen spreads
        fair_value = 150.0
        spread_normal = self.strategy._calculate_optimal_spread(self.market_data, fair_value)

        # Simulate adverse selection condition
        self.strategy.adverse_selection_detected = True
        spread_adverse = self.strategy._calculate_optimal_spread(self.market_data, fair_value)

        self.assertGreater(spread_adverse, spread_normal)

    def test_position_limits(self):
        """Test position limit enforcement"""
        # Set large position
        self.strategy.current_position = 900  # Close to limit of 1000

        # Generate signals
        result = self.strategy.on_market_update(self.market_data, self.timestamp)

        # Should still generate ask order but smaller bid order
        bid_order = next((o for o in result.orders if o.side == OrderSide.BUY), None)
        ask_order = next((o for o in result.orders if o.side == OrderSide.SELL), None)

        if bid_order:
            self.assertLessEqual(bid_order.volume, 100)  # Reduced size

        self.assertIsNotNone(ask_order)  # Should still offer to sell


class TestLiquidityTakingStrategy(unittest.TestCase):
    """Test cases for LiquidityTakingStrategy class"""

    def setUp(self):
        """Set up test fixtures"""
        self.symbol = "AAPL"
        self.config = LiquidityTakingConfig(
            signal_threshold=0.3,  # Higher threshold for proper weak signal testing  
            confidence_threshold=0.6,
            mean_reversion_window=20,
        )

        # Create mock portfolio
        self.portfolio = Mock(spec=Portfolio)
        self.portfolio.current_cash = 100000.0
        self.portfolio.total_value = 100000.0

        self.strategy = LiquidityTakingStrategy(
            symbol=self.symbol,
            config=self.config,
            max_order_size=500  # Allow larger orders for strong signals
        )

        # Create test market data
        self.timestamp = pd.Timestamp.now()
        self.market_data = self._create_book_snapshot()

        # Set up price history for signal calculation
        self.strategy.price_history = [149.0, 149.5, 150.0]

    def _create_book_snapshot(self) -> BookSnapshot:
        """Create test book snapshot"""
        return BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(149.95, 1000), (149.90, 800), (149.85, 600)],
            asks=[(150.05, 1000), (150.10, 800), (150.15, 600)]
        )

    def test_momentum_signal_generation(self):
        """Test momentum signal generation"""
        # Set up strong upward momentum
        self.strategy.price_history = [148.0, 149.0, 150.0, 151.0]

        momentum_signal = self.strategy._calculate_momentum_signal(self.market_data)

        # Should generate positive momentum signal
        self.assertGreater(momentum_signal, self.config.momentum_threshold)

    def test_mean_reversion_signal_generation(self):
        """Test mean reversion signal generation"""
        # Set up price deviation from moving average
        self.strategy.price_history = [150.0] * 10 + [152.0]  # Price spike

        mean_reversion_signal = self.strategy._calculate_mean_reversion_signal(self.market_data)

        # Should generate negative mean reversion signal (price too high)
        self.assertLess(mean_reversion_signal, -self.config.mean_reversion_threshold)

    def test_volume_signal_generation(self):
        """Test volume-based signal generation"""
        # High volume should strengthen signals
        volume_signal = self.strategy._calculate_volume_signal(self.market_data)

        # Volume is 1500, threshold is 1000, so should be positive
        self.assertGreater(volume_signal, 0)

    def test_order_flow_analysis(self):
        """Test order flow analysis"""
        # Set up recent trades with buying pressure
        self.strategy.trade_history = [
            Trade("T1", self.symbol, 150.05, 100, self.timestamp, "BUY1", "SELL1"),  # Buy at ask
            Trade("T2", self.symbol, 150.05, 200, self.timestamp, "BUY2", "SELL2"),  # Buy at ask
            Trade("T3", self.symbol, 149.95, 50, self.timestamp, "BUY3", "SELL3")   # Sell at bid
        ]

        order_flow_signal = self.strategy._calculate_order_flow_signal(self.market_data)

        # Should detect buying pressure
        self.assertGreater(order_flow_signal, 0)

    def test_signal_combination(self):
        """Test signal combination logic"""
        # Mock individual signals
        with patch.object(self.strategy, '_calculate_momentum_signal', return_value=0.015):
            with patch.object(self.strategy, '_calculate_mean_reversion_signal', return_value=-0.005):
                with patch.object(self.strategy, '_calculate_volume_signal', return_value=0.5):
                    with patch.object(self.strategy, '_calculate_order_flow_signal', return_value=0.3):

                        combined_signal = self.strategy._combine_signals(self.market_data)

                        # Should be positive (momentum + volume + order flow > mean reversion)
                        self.assertGreater(combined_signal, 0)

    def test_signal_generation_buy(self):
        """Test buy signal generation"""
        self.strategy.set_active(True)

        # Mock strong buy signal
        with patch.object(self.strategy, '_combine_signals', return_value=0.8):
            result = self.strategy.on_market_update(self.market_data, self.timestamp)

            # Should generate buy order
            self.assertEqual(len(result.orders), 1)
            order = result.orders[0]
            self.assertEqual(order.side, OrderSide.BUY)
            self.assertEqual(order.order_type, OrderType.MARKET)

    def test_signal_generation_sell(self):
        """Test sell signal generation"""
        self.strategy.set_active(True)

        # Mock strong sell signal
        with patch.object(self.strategy, '_combine_signals', return_value=-0.8):
            result = self.strategy.on_market_update(self.market_data, self.timestamp)

            # Should generate sell order
            self.assertEqual(len(result.orders), 1)
            order = result.orders[0]
            self.assertEqual(order.side, OrderSide.SELL)
            self.assertEqual(order.order_type, OrderType.MARKET)

    def test_no_signal_generation(self):
        """Test no signal generation when signals are weak"""
        self.strategy.set_active(True)

        # Mock weak signal
        with patch.object(self.strategy, '_combine_signals', return_value=0.1):
            result = self.strategy.on_market_update(self.market_data, self.timestamp)

            # Should not generate any orders
            self.assertEqual(len(result.orders), 0)

    def test_position_sizing(self):
        """Test position sizing logic"""
        # Test with no existing position
        self.strategy.current_position = 0
        size = self.strategy._calculate_position_size(0.5)  # Medium signal strength

        self.assertGreater(size, 0)
        self.assertLessEqual(size, self.config.position_limit)

        # Test with existing position near limit
        self.strategy.current_position = 450
        size_limited = self.strategy._calculate_position_size(0.5)

        self.assertLessEqual(size_limited, 50)  # Should be limited

    def test_execution_timing(self):
        """Test execution timing optimization"""
        # Test with tight spread (good timing)
        tight_spread_data = self._create_book_snapshot()

        timing_score = self.strategy._evaluate_execution_timing(tight_spread_data)
        self.assertGreater(timing_score, 0.5)  # Good timing

        # Test with wide spread (poor timing)
        wide_spread_data = BookSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamp,
            bids=[(149.90, 1000)],
            asks=[(150.10, 1000)]
        )

        timing_score_wide = self.strategy._evaluate_execution_timing(wide_spread_data)
        self.assertLess(timing_score_wide, timing_score)  # Worse timing


class TestStrategyUtils(unittest.TestCase):
    """Test cases for strategy utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager()

    def test_risk_manager_position_limits(self):
        """Test risk manager position limits"""
        order = Order(
            order_id="test",
            symbol="AAPL",
            side=OrderSide.BUY,
            volume=1000,
            price=150.0,
            order_type=OrderType.LIMIT,
            timestamp=pd.Timestamp.now()
        )
        # Test within limits
        self.assertTrue(self.risk_manager.check_risk_limits("AAPL", order, 150.0)[0])

        # Test exceeding limits
        order.volume = 100000
        self.assertFalse(self.risk_manager.check_risk_limits("AAPL", order, 150.0)[0])

    def test_risk_manager_drawdown_limits(self):
        """Test risk manager drawdown limits"""
        # Test acceptable drawdown
        self.risk_manager.current_drawdown = 0.01
        order = Order(
            order_id="test",
            symbol="AAPL",
            side=OrderSide.BUY,
            volume=100,
            price=150.0,
            order_type=OrderType.LIMIT,
            timestamp=pd.Timestamp.now()
        )
        self.assertTrue(self.risk_manager.check_risk_limits("AAPL", order, 150.0)[0])

        # Test excessive drawdown
        self.risk_manager.current_drawdown = 0.10
        self.assertFalse(self.risk_manager.check_risk_limits("AAPL", order, 150.0)[0])


if __name__ == '__main__':
    unittest.main()