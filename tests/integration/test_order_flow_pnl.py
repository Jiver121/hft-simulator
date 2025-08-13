"""
Order Flow and P&L Validation Integration Tests.

This module provides comprehensive testing for order flow from strategy signals
to execution and detailed P&L calculation validation across different scenarios.
"""

import pytest
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import Mock

from src.engine.order_book import OrderBook
from src.engine.order_types import Order, OrderSide, OrderType
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestOrderFlowPnL:
    """Comprehensive tests for order flow and P&L calculations."""

    @pytest.fixture
    def pnl_test_environment(self):
        """Set up environment specifically for P&L testing."""
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        risk_manager = RiskManager(portfolio)
        
        # Strategies with known parameters for predictable testing
        market_maker = MarketMakingStrategy(
            target_spread=0.02,  # 2% spread
            max_position=1000,
            inventory_target=0
        )
        
        liquidity_taker = LiquidityTakingStrategy(
            signal_threshold=0.01,  # 1% threshold
            max_order_size=100
        )
        
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        
        return {
            'order_book': order_book,
            'portfolio': portfolio,
            'risk_manager': risk_manager,
            'market_maker': market_maker,
            'liquidity_taker': liquidity_taker,
            'execution_simulator': execution_simulator
        }

    def test_complete_order_lifecycle(self, pnl_test_environment):
        """Test complete order lifecycle from signal to settlement."""
        env = pnl_test_environment
        symbol = 'BTCUSDT'
        base_price = 50000.0
        
        # Phase 1: Initialize market conditions
        initial_market_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': base_price - 5.0,
            'ask_price': base_price + 5.0,
            'bid_size': 10.0,
            'ask_size': 10.0,
            'last_price': base_price,
            'volume': 1.0
        }
        
        env['order_book'].process_market_data(initial_market_data)
        
        # Phase 2: Strategy signal generation
        mm_signals = env['market_maker'].generate_signals(initial_market_data)
        assert len(mm_signals) > 0, "Market maker should generate signals"
        
        # Phase 3: Order creation and validation
        orders_created = []
        for i, signal in enumerate(mm_signals):
            order = Order(
                order_id=f"lifecycle_test_{i}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.LIMIT,
                quantity=signal['quantity'],
                price=signal['price']
            )
            
            # Validate order parameters
            assert order.quantity > 0, "Order quantity must be positive"
            assert order.price > 0, "Order price must be positive"
            assert order.symbol == symbol, "Order symbol must match"
            
            orders_created.append(order)
        
        # Phase 4: Risk management validation
        risk_approved_orders = []
        risk_rejected_orders = []
        
        for order in orders_created:
            risk_result = env['risk_manager'].validate_order(order)
            
            assert 'approved' in risk_result, "Risk result must contain approval status"
            assert 'reason' in risk_result, "Risk result must contain reason"
            
            if risk_result['approved']:
                risk_approved_orders.append(order)
            else:
                risk_rejected_orders.append((order, risk_result['reason']))
        
        logger.info(f"Risk management: {len(risk_approved_orders)} approved, {len(risk_rejected_orders)} rejected")
        
        # Phase 5: Order execution
        executed_fills = []
        execution_failures = []
        
        for order in risk_approved_orders:
            fill = env['execution_simulator'].execute_order(order)
            
            if fill:
                # Validate fill structure
                assert 'order_id' in fill, "Fill must contain order_id"
                assert 'symbol' in fill, "Fill must contain symbol"
                assert 'side' in fill, "Fill must contain side"
                assert 'quantity' in fill, "Fill must contain quantity"
                assert 'price' in fill, "Fill must contain price"
                assert 'timestamp' in fill, "Fill must contain timestamp"
                
                # Validate fill values
                assert fill['quantity'] > 0, "Fill quantity must be positive"
                assert fill['price'] > 0, "Fill price must be positive"
                assert fill['quantity'] <= order.quantity, "Fill quantity cannot exceed order quantity"
                
                executed_fills.append(fill)
            else:
                execution_failures.append(order.order_id)
        
        logger.info(f"Execution: {len(executed_fills)} fills, {len(execution_failures)} failures")
        
        # Phase 6: Portfolio updates
        initial_portfolio_value = env['portfolio'].get_summary()['total_value']
        
        for fill in executed_fills:
            env['portfolio'].update_position(fill)
        
        # Phase 7: Final validation
        final_portfolio_summary = env['portfolio'].get_summary()
        
        # Portfolio should be updated
        if len(executed_fills) > 0:
            assert final_portfolio_summary['total_value'] != initial_portfolio_value, \
                "Portfolio value should change after trades"
        
        # Trade history should match executed fills
        trade_history = env['portfolio'].get_trade_history()
        assert len(trade_history) == len(executed_fills), \
            "Trade history should match executed fills"
        
        # All fills should be recorded correctly
        recorded_fill_ids = {trade['order_id'] for trade in trade_history}
        executed_fill_ids = {fill['order_id'] for fill in executed_fills}
        assert recorded_fill_ids == executed_fill_ids, \
            "All executed fills should be in trade history"

    def test_pnl_calculation_accuracy(self, pnl_test_environment):
        """Test accuracy of P&L calculations with known trade sequences."""
        env = pnl_test_environment
        symbol = 'BTCUSDT'
        
        # Define test trades with known P&L outcomes
        test_trades = [
            # Trade 1: Buy 1 BTC at $50,000
            {
                'side': OrderSide.BUY,
                'quantity': 1.0,
                'price': 50000.0,
                'expected_cost': 50000.0,
                'expected_commission': 50.0  # 0.1% commission
            },
            # Trade 2: Sell 0.5 BTC at $51,000 (profit)
            {
                'side': OrderSide.SELL,
                'quantity': 0.5,
                'price': 51000.0,
                'expected_revenue': 25500.0,
                'expected_commission': 25.5,
                'expected_pnl': 500.0 - 25.5  # Profit minus commission
            },
            # Trade 3: Buy 0.3 BTC at $52,000
            {
                'side': OrderSide.BUY,
                'quantity': 0.3,
                'price': 52000.0,
                'expected_cost': 15600.0,
                'expected_commission': 15.6
            },
            # Trade 4: Sell 0.8 BTC at $49,000 (loss)
            {
                'side': OrderSide.SELL,
                'quantity': 0.8,
                'price': 49000.0,
                'expected_revenue': 39200.0,
                'expected_commission': 39.2
            }
        ]
        
        calculated_pnl = 0.0
        position_tracker = []  # Track positions for FIFO calculation
        
        for i, trade in enumerate(test_trades):
            # Create and execute order
            order = Order(
                order_id=f"pnl_accuracy_{i}",
                symbol=symbol,
                side=trade['side'],
                order_type=OrderType.MARKET,
                quantity=trade['quantity'],
                price=trade['price']
            )
            
            # Mock fill to ensure exact prices
            fill = {
                'order_id': order.order_id,
                'symbol': symbol,
                'side': trade['side'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'timestamp': time.time() + i,
                'commission': trade['quantity'] * trade['price'] * 0.001  # 0.1% commission
            }
            
            # Update portfolio
            env['portfolio'].update_position(fill)
            
            # Manual P&L calculation for validation
            if trade['side'] == OrderSide.BUY:
                position_tracker.append({
                    'quantity': trade['quantity'],
                    'price': trade['price']
                })
            else:  # SELL
                remaining_to_sell = trade['quantity']
                trade_pnl = 0.0
                
                # FIFO calculation
                while remaining_to_sell > 0 and position_tracker:
                    oldest_position = position_tracker[0]
                    
                    if oldest_position['quantity'] <= remaining_to_sell:
                        # Sell entire oldest position
                        sell_quantity = oldest_position['quantity']
                        trade_pnl += (trade['price'] - oldest_position['price']) * sell_quantity
                        remaining_to_sell -= sell_quantity
                        position_tracker.pop(0)
                    else:
                        # Partially sell oldest position
                        trade_pnl += (trade['price'] - oldest_position['price']) * remaining_to_sell
                        oldest_position['quantity'] -= remaining_to_sell
                        remaining_to_sell = 0
                
                calculated_pnl += trade_pnl - fill['commission']
        
        # Get portfolio P&L
        portfolio_summary = env['portfolio'].get_summary()
        portfolio_pnl = portfolio_summary.get('realized_pnl', 0.0)
        
        # Allow for small floating-point differences
        pnl_difference = abs(portfolio_pnl - calculated_pnl)
        assert pnl_difference < 0.01, f"P&L calculation mismatch: portfolio={portfolio_pnl}, calculated={calculated_pnl}, diff={pnl_difference}"
        
        # Test P&L attribution
        pnl_attribution = env['portfolio'].get_pnl_attribution()
        assert symbol in pnl_attribution, "P&L attribution should include the symbol"
        
        symbol_attribution = pnl_attribution[symbol]
        assert 'trading_pnl' in symbol_attribution, "Should have trading P&L"
        assert 'commission_cost' in symbol_attribution, "Should have commission costs"
        
        # Total commissions should match
        expected_total_commission = sum(
            trade['quantity'] * trade['price'] * 0.001
            for trade in test_trades
        )
        
        total_commission = abs(symbol_attribution['commission_cost'])  # Commissions are negative
        commission_difference = abs(total_commission - expected_total_commission)
        assert commission_difference < 0.01, f"Commission mismatch: expected={expected_total_commission}, actual={total_commission}"

    def test_multi_asset_pnl_separation(self, pnl_test_environment):
        """Test P&L calculation accuracy across multiple assets."""
        env = pnl_test_environment
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Define trades for each symbol
        trades_by_symbol = {
            'BTCUSDT': [
                {'side': OrderSide.BUY, 'quantity': 1.0, 'price': 50000.0},
                {'side': OrderSide.SELL, 'quantity': 1.0, 'price': 51000.0}  # $1000 profit
            ],
            'ETHUSDT': [
                {'side': OrderSide.BUY, 'quantity': 10.0, 'price': 3000.0},
                {'side': OrderSide.SELL, 'quantity': 10.0, 'price': 2900.0}  # $1000 loss
            ]
        }
        
        expected_pnl_by_symbol = {}
        
        # Execute trades for each symbol
        for symbol, trades in trades_by_symbol.items():
            symbol_pnl = 0.0
            
            for i, trade in enumerate(trades):
                order = Order(
                    order_id=f"multi_asset_{symbol}_{i}",
                    symbol=symbol,
                    side=trade['side'],
                    order_type=OrderType.MARKET,
                    quantity=trade['quantity'],
                    price=trade['price']
                )
                
                fill = {
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'side': trade['side'],
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'timestamp': time.time() + i,
                    'commission': trade['quantity'] * trade['price'] * 0.001
                }
                
                env['portfolio'].update_position(fill)
                
                # Calculate expected P&L (simplified for this test)
                if trade['side'] == OrderSide.SELL:
                    # Assume FIFO and previous buy at the first trade price
                    buy_price = trades[0]['price']
                    symbol_pnl += (trade['price'] - buy_price) * trade['quantity'] - fill['commission']
                else:
                    symbol_pnl -= fill['commission']  # Just commission cost for buys
            
            expected_pnl_by_symbol[symbol] = symbol_pnl
        
        # Validate P&L separation by asset
        pnl_attribution = env['portfolio'].get_pnl_attribution()
        
        for symbol in symbols:
            assert symbol in pnl_attribution, f"P&L attribution missing for {symbol}"
            
            symbol_data = pnl_attribution[symbol]
            actual_pnl = symbol_data.get('trading_pnl', 0.0) + symbol_data.get('commission_cost', 0.0)
            expected_pnl = expected_pnl_by_symbol[symbol]
            
            pnl_diff = abs(actual_pnl - expected_pnl)
            assert pnl_diff < 1.0, f"P&L mismatch for {symbol}: expected={expected_pnl}, actual={actual_pnl}"
        
        # Cross-check: total P&L should equal sum of individual symbols
        total_portfolio_pnl = env['portfolio'].get_summary().get('realized_pnl', 0.0)
        sum_of_symbol_pnl = sum(expected_pnl_by_symbol.values())
        
        total_diff = abs(total_portfolio_pnl - sum_of_symbol_pnl)
        assert total_diff < 1.0, f"Total P&L mismatch: portfolio={total_portfolio_pnl}, sum={sum_of_symbol_pnl}"

    def test_complex_order_flow_scenarios(self, pnl_test_environment):
        """Test complex order flow scenarios with partial fills and cancellations."""
        env = pnl_test_environment
        symbol = 'BTCUSDT'
        base_price = 50000.0
        
        # Initialize market with sufficient depth
        env['order_book'].process_market_data({
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': base_price - 10,
            'ask_price': base_price + 10,
            'bid_size': 100.0,
            'ask_size': 100.0
        })
        
        # Scenario 1: Large order that should be partially filled
        large_order = Order(
            order_id="large_order_test",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50.0,  # Larger than typical market depth
            price=base_price + 5
        )
        
        fill_large = env['execution_simulator'].execute_order(large_order)
        
        if fill_large:
            # Validate partial fill behavior
            assert fill_large['quantity'] <= large_order.quantity, \
                "Fill quantity should not exceed order quantity"
            
            if fill_large['quantity'] < large_order.quantity:
                logger.info(f"Partial fill: {fill_large['quantity']}/{large_order.quantity}")
            
            env['portfolio'].update_position(fill_large)
        
        # Scenario 2: Out-of-market orders that might not fill
        out_of_market_order = Order(
            order_id="out_of_market_test",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=base_price - 1000  # Far below market
        )
        
        fill_out_of_market = env['execution_simulator'].execute_order(out_of_market_order)
        
        # This order should not fill
        if not fill_out_of_market:
            logger.info("Out-of-market order correctly did not fill")
        else:
            env['portfolio'].update_position(fill_out_of_market)
        
        # Scenario 3: Rapid sequence of small orders
        small_orders_fills = []
        
        for i in range(10):
            small_order = Order(
                order_id=f"small_order_{i}",
                symbol=symbol,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=0.1,
                price=base_price
            )
            
            fill = env['execution_simulator'].execute_order(small_order)
            if fill:
                small_orders_fills.append(fill)
                env['portfolio'].update_position(fill)
        
        # Validate order flow integrity
        portfolio_summary = env['portfolio'].get_summary()
        trade_history = env['portfolio'].get_trade_history()
        
        # All fills should be recorded
        expected_fills = len([f for f in [fill_large, fill_out_of_market] if f]) + len(small_orders_fills)
        actual_fills = len(trade_history)
        
        assert actual_fills == expected_fills, \
            f"Trade history mismatch: expected {expected_fills}, actual {actual_fills}"
        
        # Portfolio should be in consistent state
        assert isinstance(portfolio_summary['total_value'], (int, float)), \
            "Portfolio total value should be numeric"
        
        # Position sizes should be consistent with fills
        positions = portfolio_summary.get('positions', {})
        if symbol in positions:
            position_size = positions[symbol]['quantity']
            
            # Calculate expected position from trade history
            expected_position = 0.0
            for trade in trade_history:
                if trade['side'] == OrderSide.BUY:
                    expected_position += trade['quantity']
                else:
                    expected_position -= trade['quantity']
            
            position_diff = abs(position_size - expected_position)
            assert position_diff < 0.0001, \
                f"Position size mismatch: expected {expected_position}, actual {position_size}"

    def test_high_frequency_order_flow(self, pnl_test_environment):
        """Test order flow under high-frequency trading conditions."""
        env = pnl_test_environment
        symbol = 'BTCUSDT'
        
        # Initialize stable market conditions
        base_price = 50000.0
        env['order_book'].process_market_data({
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': base_price - 1,
            'ask_price': base_price + 1,
            'bid_size': 1000.0,
            'ask_size': 1000.0
        })
        
        # Generate high-frequency orders
        hf_orders = []
        for i in range(100):  # 100 rapid orders
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            price = base_price + np.random.uniform(-5, 5)
            quantity = np.random.uniform(0.01, 1.0)
            
            order = Order(
                order_id=f"hf_order_{i}",
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price
            )
            
            hf_orders.append(order)
        
        # Execute all orders rapidly
        start_time = time.time()
        executed_fills = []
        
        for order in hf_orders:
            # Risk check
            risk_result = env['risk_manager'].validate_order(order)
            
            if risk_result['approved']:
                fill = env['execution_simulator'].execute_order(order)
                if fill:
                    executed_fills.append(fill)
                    env['portfolio'].update_position(fill)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance validation
        orders_per_second = len(hf_orders) / execution_time
        logger.info(f"High-frequency execution: {orders_per_second:.0f} orders/second")
        
        assert orders_per_second > 100, f"High-frequency execution too slow: {orders_per_second} orders/second"
        
        # Validate system integrity after high-frequency trading
        portfolio_summary = env['portfolio'].get_summary()
        trade_history = env['portfolio'].get_trade_history()
        
        assert len(trade_history) == len(executed_fills), \
            "All executed fills should be in trade history"
        
        # P&L calculations should still be accurate
        if len(executed_fills) > 0:
            pnl_attribution = env['portfolio'].get_pnl_attribution()
            assert symbol in pnl_attribution, "P&L attribution should include the symbol"
        
        # Risk metrics should still be calculable
        risk_metrics = env['risk_manager'].calculate_risk_metrics()
        assert 'max_drawdown' in risk_metrics, "Risk metrics should be calculable"
        
        # Final portfolio state should be valid
        assert portfolio_summary['total_value'] > 0, "Portfolio value should remain positive"

    def test_commission_and_fee_accuracy(self, pnl_test_environment):
        """Test accuracy of commission and fee calculations."""
        env = pnl_test_environment
        symbol = 'BTCUSDT'
        
        # Define commission structure for testing
        commission_rate = 0.001  # 0.1%
        
        # Test trades with known commission amounts
        test_trades = [
            {'quantity': 1.0, 'price': 50000.0, 'expected_commission': 50.0},
            {'quantity': 0.5, 'price': 60000.0, 'expected_commission': 30.0},
            {'quantity': 2.0, 'price': 45000.0, 'expected_commission': 90.0},
        ]
        
        total_expected_commission = 0.0
        
        for i, trade in enumerate(test_trades):
            order = Order(
                order_id=f"commission_test_{i}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=trade['quantity'],
                price=trade['price']
            )
            
            fill = {
                'order_id': order.order_id,
                'symbol': symbol,
                'side': OrderSide.BUY,
                'quantity': trade['quantity'],
                'price': trade['price'],
                'timestamp': time.time() + i,
                'commission': trade['quantity'] * trade['price'] * commission_rate
            }
            
            env['portfolio'].update_position(fill)
            total_expected_commission += trade['expected_commission']
            
            # Validate individual commission calculation
            calculated_commission = trade['quantity'] * trade['price'] * commission_rate
            assert abs(calculated_commission - trade['expected_commission']) < 0.01, \
                f"Commission calculation error for trade {i}"
        
        # Validate total commission tracking
        pnl_attribution = env['portfolio'].get_pnl_attribution()
        symbol_data = pnl_attribution[symbol]
        total_actual_commission = abs(symbol_data['commission_cost'])  # Commissions are negative
        
        commission_diff = abs(total_actual_commission - total_expected_commission)
        assert commission_diff < 0.01, \
            f"Total commission mismatch: expected={total_expected_commission}, actual={total_actual_commission}"
        
        # Test commission impact on P&L
        portfolio_summary = env['portfolio'].get_summary()
        cash_balance = portfolio_summary['cash_balance']
        
        # Cash balance should be reduced by total commission
        initial_balance = 100000.0  # From fixture
        total_trade_value = sum(trade['quantity'] * trade['price'] for trade in test_trades)
        expected_cash = initial_balance - total_trade_value - total_expected_commission
        
        cash_diff = abs(cash_balance - expected_cash)
        assert cash_diff < 0.01, \
            f"Cash balance mismatch: expected={expected_cash}, actual={cash_balance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
