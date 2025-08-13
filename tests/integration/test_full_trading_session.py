"""
Integration tests for complete trading sessions.

This module implements end-to-end testing that simulates full trading sessions
from strategy signal generation to execution, portfolio updates, and P&L calculations.
"""

import asyncio
import pytest
import time
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from src.engine.order_book import OrderBook
from src.engine.order_types import Order, OrderSide, OrderType
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.realtime.data_feeds import MockDataFeed
from src.realtime.trading_system import RealTimeTradingSystem
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestFullTradingSessions:
    """Test complete trading sessions with multiple strategies and assets."""

    @pytest.fixture
    def setup_trading_environment(self):
        """Set up a complete trading environment for testing."""
        # Core components
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        risk_manager = RiskManager(portfolio)
        
        # Strategies
        market_maker = MarketMakingStrategy(
            target_spread=0.01,
            max_position=1000,
            inventory_target=0
        )
        
        liquidity_taker = LiquidityTakingStrategy(
            signal_threshold=0.005,
            max_order_size=500
        )
        
        # Execution components
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        
        # Data feed
        data_feed = MockDataFeed(symbols=['BTCUSDT', 'ETHUSDT'])
        
        # Trading system (temporarily disabled due to constructor mismatch)
        # trading_system = RealTimeTradingSystem(
        #     order_book=order_book,
        #     strategies=[market_maker, liquidity_taker],
        #     portfolio=portfolio,
        #     risk_manager=risk_manager,
        #     data_feed=data_feed
        # )
        trading_system = None
        
        return {
            'order_book': order_book,
            'portfolio': portfolio,
            'risk_manager': risk_manager,
            'market_maker': market_maker,
            'liquidity_taker': liquidity_taker,
            'execution_simulator': execution_simulator,
            'data_feed': data_feed,
            'trading_system': trading_system
        }

    def test_single_asset_trading_session(self, setup_trading_environment):
        """Test a complete trading session for a single asset."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        
        # Initialize order book with market data
        env['order_book'].process_market_data({
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': 50000.0,
            'ask_price': 50001.0,
            'bid_size': 10.0,
            'ask_size': 10.0
        })
        
        # Test market making strategy signal generation
        market_data = {
            'symbol': symbol,
            'price': 50000.5,
            'timestamp': time.time(),
            'volume': 1.0
        }
        
        mm_signals = env['market_maker'].generate_signals(market_data)
        assert len(mm_signals) >= 1, "Market maker should generate at least one signal"
        
        # Execute market making orders
        for signal in mm_signals:
            order = Order(
                order_id=f"mm_{signal['timestamp']}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.LIMIT,
                quantity=signal['quantity'],
                price=signal['price']
            )
            
            fill = env['execution_simulator'].execute_order(order)
            if fill:
                env['portfolio'].update_position(fill)
        
        # Simulate price movement and liquidity taking
        updated_market_data = {
            'symbol': symbol,
            'price': 50010.0,  # Price moved up
            'timestamp': time.time() + 1,
            'volume': 2.0
        }
        
        lt_signals = env['liquidity_taker'].generate_signals(updated_market_data)
        
        # Execute liquidity taking orders
        for signal in lt_signals:
            order = Order(
                order_id=f"lt_{signal['timestamp']}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.MARKET,
                quantity=signal['quantity']
            )
            
            fill = env['execution_simulator'].execute_order(order)
            if fill:
                env['portfolio'].update_position(fill)
        
        # Validate portfolio state
        portfolio_summary = env['portfolio'].get_summary()
        assert 'total_pnl' in portfolio_summary
        assert 'positions' in portfolio_summary
        
        # Validate risk metrics
        risk_metrics = env['risk_manager'].calculate_risk_metrics()
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics

    def test_multi_asset_trading_session(self, setup_trading_environment):
        """Test trading session with multiple assets."""
        env = setup_trading_environment
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        # Initialize order books for multiple symbols
        market_data_updates = [
            {
                'symbol': 'BTCUSDT',
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'bid_size': 10.0,
                'ask_size': 10.0
            },
            {
                'symbol': 'ETHUSDT',
                'timestamp': time.time(),
                'bid_price': 3000.0,
                'ask_price': 3001.0,
                'bid_size': 50.0,
                'ask_size': 50.0
            }
        ]
        
        for data in market_data_updates:
            env['order_book'].process_market_data(data)
        
        # Test concurrent trading on multiple assets
        orders_executed = []
        
        for symbol in symbols:
            # Generate and execute market making orders
            market_data = {
                'symbol': symbol,
                'price': 50000.0 if symbol == 'BTCUSDT' else 3000.0,
                'timestamp': time.time(),
                'volume': 1.0
            }
            
            signals = env['market_maker'].generate_signals(market_data)
            
            for signal in signals:
                order = Order(
                    order_id=f"multi_{symbol}_{len(orders_executed)}",
                    symbol=symbol,
                    side=signal['side'],
                    order_type=OrderType.LIMIT,
                    quantity=signal['quantity'],
                    price=signal['price']
                )
                
                fill = env['execution_simulator'].execute_order(order)
                if fill:
                    env['portfolio'].update_position(fill)
                    orders_executed.append(order)
        
        # Validate multi-asset portfolio
        portfolio_summary = env['portfolio'].get_summary()
        assert len(portfolio_summary['positions']) <= len(symbols)
        
        # Check risk metrics across multiple assets
        risk_metrics = env['risk_manager'].calculate_risk_metrics()
        assert 'portfolio_var' in risk_metrics or 'var_95' in risk_metrics

    @pytest.mark.asyncio
    async def test_order_flow_end_to_end(self, setup_trading_environment):
        """Test complete order flow from signal to execution."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        
        # Step 1: Market data arrives
        market_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': 50000.0,
            'ask_price': 50001.0,
            'bid_size': 10.0,
            'ask_size': 10.0,
            'last_price': 50000.5,
            'volume': 1.0
        }
        
        # Step 2: Update order book
        env['order_book'].process_market_data(market_data)
        
        # Step 3: Strategy generates signals
        signals = env['market_maker'].generate_signals(market_data)
        assert len(signals) > 0, "Strategy should generate signals"
        
        # Step 4: Convert signals to orders
        orders = []
        for i, signal in enumerate(signals):
            order = Order(
                order_id=f"test_order_{i}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.LIMIT,
                quantity=signal['quantity'],
                price=signal['price']
            )
            orders.append(order)
        
        # Step 5: Risk management checks
        for order in orders:
            risk_check = env['risk_manager'].validate_order(order)
            assert risk_check['approved'], f"Order should pass risk checks: {risk_check}"
        
        # Step 6: Execute orders
        fills = []
        for order in orders:
            fill = env['execution_simulator'].execute_order(order)
            if fill:
                fills.append(fill)
        
        assert len(fills) > 0, "At least one order should be filled"
        
        # Step 7: Update portfolio
        for fill in fills:
            env['portfolio'].update_position(fill)
        
        # Step 8: Validate final state
        portfolio_summary = env['portfolio'].get_summary()
        assert portfolio_summary['total_value'] != 100000.0  # Should have changed
        
        # Check that orders are properly recorded
        trade_history = env['portfolio'].get_trade_history()
        assert len(trade_history) == len(fills)

    def test_portfolio_pnl_calculations(self, setup_trading_environment):
        """Test comprehensive P&L calculations and updates."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        initial_balance = env['portfolio'].cash_balance
        
        # Execute a series of trades with known outcomes
        test_trades = [
            {'side': OrderSide.BUY, 'quantity': 1.0, 'price': 50000.0},
            {'side': OrderSide.SELL, 'quantity': 0.5, 'price': 50100.0},  # Profit
            {'side': OrderSide.BUY, 'quantity': 0.5, 'price': 50200.0},
            {'side': OrderSide.SELL, 'quantity': 1.0, 'price': 49900.0},  # Loss
        ]
        
        expected_pnl = 0.0
        
        for i, trade in enumerate(test_trades):
            order = Order(
                order_id=f"pnl_test_{i}",
                symbol=symbol,
                side=trade['side'],
                order_type=OrderType.MARKET,
                quantity=trade['quantity'],
                price=trade['price']
            )
            
            # Mock the fill for precise P&L testing
            fill = {
                'order_id': order.order_id,
                'symbol': symbol,
                'side': trade['side'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'timestamp': time.time() + i,
                'commission': trade['quantity'] * trade['price'] * 0.001  # 0.1% commission
            }
            
            env['portfolio'].update_position(fill)
            
            # Calculate expected P&L manually for validation
            if trade['side'] == OrderSide.SELL:
                # Assuming FIFO for simplicity
                if i == 1:  # First sell at profit
                    expected_pnl += (50100.0 - 50000.0) * 0.5 - fill['commission']
                elif i == 3:  # Second sell at loss
                    # Selling 0.5 at 50000 cost and 0.5 at 50200 cost
                    expected_pnl += (49900.0 - 50000.0) * 0.5 + (49900.0 - 50200.0) * 0.5 - fill['commission']
        
        # Validate P&L calculations
        portfolio_summary = env['portfolio'].get_summary()
        realized_pnl = portfolio_summary.get('realized_pnl', 0.0)
        
        # Allow for small numerical differences
        assert abs(realized_pnl - expected_pnl) < 1.0, f"P&L mismatch: expected {expected_pnl}, got {realized_pnl}"
        
        # Test P&L attribution
        pnl_attribution = env['portfolio'].get_pnl_attribution()
        assert symbol in pnl_attribution
        assert 'trading_pnl' in pnl_attribution[symbol]
        assert 'commission_cost' in pnl_attribution[symbol]

    def test_multi_strategy_interaction(self, setup_trading_environment):
        """Test interactions between multiple strategies sharing resources."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        
        # Set up market conditions that would trigger both strategies
        market_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': 50000.0,
            'ask_price': 50002.0,  # Wide spread for market making
            'bid_size': 10.0,
            'ask_size': 10.0,
            'last_price': 50001.0,
            'volume': 10.0,
            'price_change': 0.01  # Strong signal for liquidity taking
        }
        
        env['order_book'].process_market_data(market_data)
        
        # Generate signals from both strategies
        mm_signals = env['market_maker'].generate_signals(market_data)
        lt_signals = env['liquidity_taker'].generate_signals(market_data)
        
        # Both strategies should generate signals
        assert len(mm_signals) > 0, "Market maker should generate signals"
        assert len(lt_signals) > 0, "Liquidity taker should generate signals"
        
        # Test resource allocation (position limits)
        all_orders = []
        
        # Create orders from market maker
        for i, signal in enumerate(mm_signals):
            order = Order(
                order_id=f"mm_multi_{i}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.LIMIT,
                quantity=signal['quantity'],
                price=signal['price']
            )
            all_orders.append(('mm', order))
        
        # Create orders from liquidity taker
        for i, signal in enumerate(lt_signals):
            order = Order(
                order_id=f"lt_multi_{i}",
                symbol=symbol,
                side=signal['side'],
                order_type=OrderType.MARKET,
                quantity=signal['quantity']
            )
            all_orders.append(('lt', order))
        
        # Execute orders and check for proper resource sharing
        total_position = 0.0
        strategy_fills = {'mm': [], 'lt': []}
        
        for strategy_name, order in all_orders:
            # Check risk limits before execution
            risk_check = env['risk_manager'].validate_order(order)
            
            if risk_check['approved']:
                fill = env['execution_simulator'].execute_order(order)
                if fill:
                    env['portfolio'].update_position(fill)
                    strategy_fills[strategy_name].append(fill)
                    
                    # Update total position
                    if fill['side'] == OrderSide.BUY:
                        total_position += fill['quantity']
                    else:
                        total_position -= fill['quantity']
        
        # Validate that risk limits were respected
        assert abs(total_position) <= env['risk_manager'].max_position_size
        
        # Both strategies should have some activity
        assert len(strategy_fills['mm']) > 0 or len(strategy_fills['lt']) > 0

    @pytest.mark.asyncio
    async def test_high_load_conditions(self, setup_trading_environment):
        """Test system behavior under high load conditions."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        
        # Generate high-frequency market data updates
        num_updates = 1000
        base_price = 50000.0
        
        start_time = time.time()
        processed_updates = 0
        
        for i in range(num_updates):
            # Simulate price volatility
            price_change = (i % 20 - 10) * 0.1  # Price oscillates
            current_price = base_price + price_change
            
            market_data = {
                'symbol': symbol,
                'timestamp': time.time(),
                'bid_price': current_price - 0.5,
                'ask_price': current_price + 0.5,
                'bid_size': 10.0 + (i % 5),
                'ask_size': 10.0 + (i % 5),
                'last_price': current_price,
                'volume': 1.0 + (i % 3)
            }
            
            try:
                # Process market data update
                env['order_book'].process_market_data(market_data)
                
                # Generate strategy signals (every 10th update to avoid overload)
                if i % 10 == 0:
                    mm_signals = env['market_maker'].generate_signals(market_data)
                    
                    # Execute some orders
                    for j, signal in enumerate(mm_signals[:2]):  # Limit to 2 orders
                        order = Order(
                            order_id=f"load_test_{i}_{j}",
                            symbol=symbol,
                            side=signal['side'],
                            order_type=OrderType.LIMIT,
                            quantity=min(signal['quantity'], 1.0),  # Small orders
                            price=signal['price']
                        )
                        
                        if env['risk_manager'].validate_order(order)['approved']:
                            fill = env['execution_simulator'].execute_order(order)
                            if fill:
                                env['portfolio'].update_position(fill)
                
                processed_updates += 1
                
                # Add small delay to simulate realistic conditions
                await asyncio.sleep(0.001)  # 1ms delay
                
            except Exception as e:
                logger.error(f"Error processing update {i}: {e}")
                break
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processed_updates >= num_updates * 0.95  # At least 95% success rate
        assert processing_time < 60.0  # Should complete within 60 seconds
        
        # System should still be in valid state
        portfolio_summary = env['portfolio'].get_summary()
        assert isinstance(portfolio_summary['total_value'], (int, float))
        assert portfolio_summary['total_value'] > 0
        
        # Risk metrics should still be calculable
        risk_metrics = env['risk_manager'].calculate_risk_metrics()
        assert 'max_drawdown' in risk_metrics

    def test_data_flow_validation(self, setup_trading_environment):
        """Test data flow from market data feed to strategy decisions."""
        env = setup_trading_environment
        
        # Mock data feed with specific test data
        test_data_sequence = [
            {
                'symbol': 'BTCUSDT',
                'timestamp': time.time(),
                'price': 50000.0,
                'volume': 1.0,
                'bid_price': 49999.0,
                'ask_price': 50001.0
            },
            {
                'symbol': 'BTCUSDT',
                'timestamp': time.time() + 1,
                'price': 50010.0,  # Price increase
                'volume': 2.0,
                'bid_price': 50009.0,
                'ask_price': 50011.0
            },
            {
                'symbol': 'BTCUSDT',
                'timestamp': time.time() + 2,
                'price': 49990.0,  # Price decrease
                'volume': 1.5,
                'bid_price': 49989.0,
                'ask_price': 49991.0
            }
        ]
        
        strategy_decisions = []
        
        # Process each data point and track strategy responses
        for data in test_data_sequence:
            # Step 1: Data arrives at order book
            env['order_book'].process_market_data(data)
            order_book_state = env['order_book'].get_state()
            
            # Validate order book updated correctly
            assert abs(order_book_state['bid_price'] - data['bid_price']) < 0.01
            assert abs(order_book_state['ask_price'] - data['ask_price']) < 0.01
            
            # Step 2: Strategy processes the data
            mm_signals = env['market_maker'].generate_signals(data)
            lt_signals = env['liquidity_taker'].generate_signals(data)
            
            # Step 3: Record strategy decisions
            strategy_decisions.append({
                'timestamp': data['timestamp'],
                'price': data['price'],
                'mm_signals': len(mm_signals),
                'lt_signals': len(lt_signals),
                'mm_decisions': mm_signals,
                'lt_decisions': lt_signals
            })
        
        # Validate data flow integrity
        assert len(strategy_decisions) == len(test_data_sequence)
        
        # Check that strategies responded to price changes appropriately
        price_changes = []
        for i in range(1, len(test_data_sequence)):
            price_change = test_data_sequence[i]['price'] - test_data_sequence[i-1]['price']
            price_changes.append(price_change)
        
        # At least some strategy decisions should have been made
        total_signals = sum(d['mm_signals'] + d['lt_signals'] for d in strategy_decisions)
        assert total_signals > 0, "Strategies should generate some signals"
        
        # Validate signal consistency with market conditions
        for decision in strategy_decisions:
            for signal in decision['mm_decisions']:
                assert 'side' in signal
                assert 'quantity' in signal
                assert 'price' in signal
                assert signal['quantity'] > 0
                assert signal['price'] > 0

    def test_system_resilience(self, setup_trading_environment):
        """Test system resilience to various error conditions."""
        env = setup_trading_environment
        symbol = 'BTCUSDT'
        
        # Test 1: Invalid market data
        invalid_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': -1.0,  # Invalid price
            'ask_price': 'invalid',  # Invalid type
            'volume': None  # Invalid volume
        }
        
        # System should handle invalid data gracefully
        try:
            env['order_book'].process_market_data(invalid_data)
        except Exception:
            pass  # Expected to fail, but shouldn't crash the system
        
        # Test 2: Order with invalid parameters
        invalid_order = Order(
            order_id="invalid_test",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=-1.0,  # Invalid quantity
            price=0.0  # Invalid price
        )
        
        # Risk manager should reject invalid orders
        risk_check = env['risk_manager'].validate_order(invalid_order)
        assert not risk_check['approved'], "Invalid order should be rejected"
        
        # Test 3: System should continue operating after errors
        valid_data = {
            'symbol': symbol,
            'timestamp': time.time(),
            'bid_price': 50000.0,
            'ask_price': 50001.0,
            'volume': 1.0
        }
        
        # This should work normally after error conditions
        env['order_book'].process_market_data(valid_data)
        signals = env['market_maker'].generate_signals(valid_data)
        
        # System should still generate valid signals
        assert isinstance(signals, list)
        
        # Test 4: Portfolio should handle extreme positions
        extreme_order = Order(
            order_id="extreme_test",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1000000.0,  # Very large order
            price=50000.0
        )
        
        # Risk manager should prevent extreme positions
        risk_check = env['risk_manager'].validate_order(extreme_order)
        assert not risk_check['approved'], "Extreme order should be rejected by risk management"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
