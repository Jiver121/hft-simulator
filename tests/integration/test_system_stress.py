"""
System stress tests for the HFT trading system.

This module tests system behavior under extreme load conditions, concurrent operations,
and stress scenarios including memory pressure and high-frequency data.
"""

import asyncio
import pytest
import time
import threading
import concurrent.futures
import psutil
import gc
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import numpy as np

from src.engine.order_book import OrderBook
from src.engine.order_types import Order, OrderSide, OrderType
from src.strategies.market_making import MarketMakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.realtime.data_feeds import MockDataFeed
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSystemStress:
    """Stress tests for the complete trading system."""

    @pytest.fixture
    def stress_environment(self):
        """Set up environment for stress testing."""
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=1000000.0)  # Larger balance for stress tests
        risk_manager = RiskManager(portfolio)
        
        # Configure for higher throughput
        market_maker = MarketMakingStrategy(
            target_spread=0.01,
            max_position=10000,
            inventory_target=0
        )
        
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        data_feed = MockDataFeed(symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'])
        
        return {
            'order_book': order_book,
            'portfolio': portfolio,
            'risk_manager': risk_manager,
            'strategy': market_maker,
            'execution_simulator': execution_simulator,
            'data_feed': data_feed
        }

    def test_high_frequency_data_processing(self, stress_environment):
        """Test processing of high-frequency market data."""
        env = stress_environment
        symbol = 'BTCUSDT'
        
        # Generate high-frequency data (10,000 updates)
        num_updates = 10000
        base_price = 50000.0
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        processed_count = 0
        errors = []
        
        for i in range(num_updates):
            try:
                # Generate realistic market data with micro-movements
                price_change = np.random.normal(0, 1.0)  # Small random changes
                current_price = base_price + price_change
                
                market_data = {
                    'symbol': symbol,
                    'timestamp': time.time() + i * 0.001,  # 1ms intervals
                    'bid_price': current_price - 0.5,
                    'ask_price': current_price + 0.5,
                    'bid_size': 10.0 + np.random.uniform(-2, 2),
                    'ask_size': 10.0 + np.random.uniform(-2, 2),
                    'last_price': current_price,
                    'volume': np.random.uniform(0.1, 2.0)
                }
                
                # Process the data
                env['order_book'].process_market_data(market_data)
                processed_count += 1
                
                # Periodic memory check
                if i % 1000 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    if current_memory > start_memory * 2:  # Memory growth > 100%
                        logger.warning(f"Memory usage doubled at iteration {i}: {current_memory}MB")
                
            except Exception as e:
                errors.append(f"Error at iteration {i}: {e}")
                if len(errors) > 100:  # Stop if too many errors
                    break
        
        end_time = time.time()
        processing_time = end_time - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Performance assertions
        assert processed_count >= num_updates * 0.95, f"Processed {processed_count}/{num_updates} updates"
        assert processing_time < 30.0, f"Processing took too long: {processing_time}s"
        assert len(errors) < num_updates * 0.01, f"Too many errors: {len(errors)}"
        
        # Memory efficiency check
        memory_growth = end_memory - start_memory
        assert memory_growth < 500, f"Memory growth too high: {memory_growth}MB"
        
        # Throughput calculation
        throughput = processed_count / processing_time
        logger.info(f"Market data processing throughput: {throughput:.0f} updates/second")
        assert throughput > 1000, f"Throughput too low: {throughput} updates/second"

    def test_concurrent_order_processing(self, stress_environment):
        """Test concurrent order processing from multiple threads."""
        env = stress_environment
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        # Initialize order books for all symbols
        for symbol in symbols:
            env['order_book'].process_market_data({
                'symbol': symbol,
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'bid_size': 100.0,
                'ask_size': 100.0
            })
        
        def create_orders(thread_id: int, num_orders: int) -> List[Dict]:
            """Create orders for a specific thread."""
            orders_created = []
            
            for i in range(num_orders):
                symbol = symbols[i % len(symbols)]
                side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
                
                order = Order(
                    order_id=f"stress_{thread_id}_{i}",
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=np.random.uniform(0.1, 10.0),
                    price=50000.0 + np.random.uniform(-100, 100)
                )
                
                try:
                    # Risk check
                    risk_result = env['risk_manager'].validate_order(order)
                    if risk_result['approved']:
                        fill = env['execution_simulator'].execute_order(order)
                        if fill:
                            env['portfolio'].update_position(fill)
                            orders_created.append({
                                'thread_id': thread_id,
                                'order_id': order.order_id,
                                'fill': fill
                            })
                except Exception as e:
                    logger.error(f"Thread {thread_id} order {i} error: {e}")
            
            return orders_created
        
        # Run concurrent order processing
        num_threads = 10
        orders_per_thread = 100
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(create_orders, thread_id, orders_per_thread)
                for thread_id in range(num_threads)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Aggregate results
        total_orders = sum(len(result) for result in results)
        unique_order_ids = set()
        
        for result in results:
            for order_info in result:
                unique_order_ids.add(order_info['order_id'])
        
        # Assertions
        assert len(unique_order_ids) == total_orders, "Duplicate order IDs detected"
        assert total_orders > 0, "No orders were processed"
        assert processing_time < 10.0, f"Concurrent processing took too long: {processing_time}s"
        
        # Portfolio should be in consistent state
        portfolio_summary = env['portfolio'].get_summary()
        assert isinstance(portfolio_summary['total_value'], (int, float))
        
        logger.info(f"Concurrent processing: {total_orders} orders in {processing_time:.2f}s")

    @pytest.mark.asyncio
    async def test_async_system_load(self, stress_environment):
        """Test system under async load with multiple coroutines."""
        env = stress_environment
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
        # Initialize all symbols
        for symbol in symbols:
            env['order_book'].process_market_data({
                'symbol': symbol,
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'bid_size': 100.0,
                'ask_size': 100.0
            })
        
        async def market_data_producer(symbol: str, num_updates: int):
            """Produce market data updates for a symbol."""
            updates_sent = 0
            
            for i in range(num_updates):
                try:
                    market_data = {
                        'symbol': symbol,
                        'timestamp': time.time(),
                        'bid_price': 50000.0 + np.random.uniform(-50, 50),
                        'ask_price': 50001.0 + np.random.uniform(-50, 50),
                        'bid_size': 10.0 + np.random.uniform(-5, 5),
                        'ask_size': 10.0 + np.random.uniform(-5, 5),
                        'volume': np.random.uniform(0.1, 5.0)
                    }
                    
                    env['order_book'].process_market_data(market_data)
                    updates_sent += 1
                    
                    await asyncio.sleep(0.001)  # 1ms delay
                    
                except Exception as e:
                    logger.error(f"Market data error for {symbol}: {e}")
            
            return updates_sent
        
        async def strategy_processor(strategy_id: int, num_decisions: int):
            """Process strategy decisions."""
            decisions_made = 0
            
            for i in range(num_decisions):
                try:
                    symbol = symbols[i % len(symbols)]
                    
                    market_data = {
                        'symbol': symbol,
                        'price': 50000.0 + np.random.uniform(-10, 10),
                        'timestamp': time.time(),
                        'volume': np.random.uniform(0.1, 2.0)
                    }
                    
                    signals = env['strategy'].generate_signals(market_data)
                    
                    for signal in signals:
                        order = Order(
                            order_id=f"async_{strategy_id}_{i}_{len(signals)}",
                            symbol=symbol,
                            side=signal['side'],
                            order_type=OrderType.LIMIT,
                            quantity=signal['quantity'],
                            price=signal['price']
                        )
                        
                        if env['risk_manager'].validate_order(order)['approved']:
                            fill = env['execution_simulator'].execute_order(order)
                            if fill:
                                env['portfolio'].update_position(fill)
                    
                    decisions_made += 1
                    await asyncio.sleep(0.002)  # 2ms delay
                    
                except Exception as e:
                    logger.error(f"Strategy {strategy_id} error: {e}")
            
            return decisions_made
        
        # Run async load test
        start_time = time.time()
        
        # Create tasks for market data and strategy processing
        tasks = []
        
        # Market data producers
        for symbol in symbols:
            tasks.append(market_data_producer(symbol, 200))
        
        # Strategy processors
        for strategy_id in range(5):
            tasks.append(strategy_processor(strategy_id, 100))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check results
        successful_results = [r for r in results if isinstance(r, int)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(errors) == 0, f"Async processing errors: {errors}"
        assert len(successful_results) > 0, "No successful async operations"
        assert processing_time < 15.0, f"Async processing took too long: {processing_time}s"
        
        # System should still be responsive
        portfolio_summary = env['portfolio'].get_summary()
        assert isinstance(portfolio_summary, dict)
        
        logger.info(f"Async load test completed: {sum(successful_results)} operations in {processing_time:.2f}s")

    def test_memory_pressure_handling(self, stress_environment):
        """Test system behavior under memory pressure."""
        env = stress_environment
        
        # Create large number of orders and market data to stress memory
        large_objects = []
        symbol = 'BTCUSDT'
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Generate many market data updates
            for i in range(5000):
                market_data = {
                    'symbol': symbol,
                    'timestamp': time.time() + i,
                    'bid_price': 50000.0 + np.random.uniform(-100, 100),
                    'ask_price': 50001.0 + np.random.uniform(-100, 100),
                    'bid_size': np.random.uniform(1, 100),
                    'ask_size': np.random.uniform(1, 100),
                    'volume': np.random.uniform(0.1, 10.0),
                    # Add extra data to increase memory usage
                    'extra_data': [np.random.random() for _ in range(100)]
                }
                
                env['order_book'].process_market_data(market_data)
                large_objects.append(market_data)
                
                # Check memory usage periodically
                if i % 500 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_growth = current_memory - start_memory
                    
                    if memory_growth > 1000:  # More than 1GB growth
                        logger.warning(f"High memory usage detected: {current_memory}MB")
                        break
            
            # Generate many orders
            for i in range(2000):
                order = Order(
                    order_id=f"memory_stress_{i}",
                    symbol=symbol,
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=np.random.uniform(0.1, 10.0),
                    price=50000.0 + np.random.uniform(-50, 50)
                )
                
                if env['risk_manager'].validate_order(order)['approved']:
                    fill = env['execution_simulator'].execute_order(order)
                    if fill:
                        env['portfolio'].update_position(fill)
            
            # Force garbage collection
            gc.collect()
            
            # Check final memory usage
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_growth = end_memory - start_memory
            
            # System should still be functional
            portfolio_summary = env['portfolio'].get_summary()
            assert isinstance(portfolio_summary['total_value'], (int, float))
            
            # Memory growth should be reasonable
            assert total_growth < 2000, f"Excessive memory growth: {total_growth}MB"
            
            logger.info(f"Memory pressure test: {total_growth}MB growth")
            
        finally:
            # Clean up large objects
            large_objects.clear()
            gc.collect()

    def test_extreme_market_conditions(self, stress_environment):
        """Test system behavior under extreme market conditions."""
        env = stress_environment
        symbol = 'BTCUSDT'
        
        # Test scenarios
        scenarios = [
            # Flash crash scenario
            {
                'name': 'flash_crash',
                'price_changes': [-0.1, -0.15, -0.2, -0.1, 0.05, 0.1, 0.15, 0.05],
                'volume_multipliers': [1, 5, 10, 8, 3, 2, 1, 1]
            },
            # Extreme volatility
            {
                'name': 'extreme_volatility',
                'price_changes': [0.05, -0.08, 0.12, -0.07, 0.09, -0.11, 0.06, -0.04],
                'volume_multipliers': [2, 8, 6, 7, 5, 9, 3, 2]
            },
            # Market manipulation scenario
            {
                'name': 'manipulation',
                'price_changes': [0.02, 0.02, 0.02, 0.02, -0.08, 0.01, 0.01, 0.01],
                'volume_multipliers': [1, 1, 1, 1, 20, 1, 1, 1]
            }
        ]
        
        base_price = 50000.0
        orders_processed = {'total': 0, 'filled': 0, 'rejected': 0}
        
        for scenario in scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            current_price = base_price
            
            for i, (price_change, volume_mult) in enumerate(zip(
                scenario['price_changes'], 
                scenario['volume_multipliers']
            )):
                # Apply price change
                current_price *= (1 + price_change)
                
                # Create extreme market data
                market_data = {
                    'symbol': symbol,
                    'timestamp': time.time() + i,
                    'bid_price': current_price * 0.999,
                    'ask_price': current_price * 1.001,
                    'bid_size': 10.0 * volume_mult,
                    'ask_size': 10.0 * volume_mult,
                    'last_price': current_price,
                    'volume': 1.0 * volume_mult,
                    'price_change': price_change
                }
                
                try:
                    # Process market data
                    env['order_book'].process_market_data(market_data)
                    
                    # Generate strategy signals
                    signals = env['strategy'].generate_signals(market_data)
                    
                    # Process orders
                    for signal in signals:
                        orders_processed['total'] += 1
                        
                        order = Order(
                            order_id=f"{scenario['name']}_{i}_{orders_processed['total']}",
                            symbol=symbol,
                            side=signal['side'],
                            order_type=OrderType.LIMIT,
                            quantity=signal['quantity'],
                            price=signal['price']
                        )
                        
                        # Risk check
                        risk_result = env['risk_manager'].validate_order(order)
                        
                        if risk_result['approved']:
                            fill = env['execution_simulator'].execute_order(order)
                            if fill:
                                env['portfolio'].update_position(fill)
                                orders_processed['filled'] += 1
                        else:
                            orders_processed['rejected'] += 1
                
                except Exception as e:
                    logger.error(f"Error in {scenario['name']} step {i}: {e}")
        
        # Validate system survived extreme conditions
        portfolio_summary = env['portfolio'].get_summary()
        risk_metrics = env['risk_manager'].calculate_risk_metrics()
        
        assert isinstance(portfolio_summary['total_value'], (int, float))
        assert 'max_drawdown' in risk_metrics
        
        # Some orders should have been processed
        assert orders_processed['total'] > 0, "No orders generated during extreme conditions"
        
        # Risk management should have rejected some orders
        rejection_rate = orders_processed['rejected'] / orders_processed['total'] if orders_processed['total'] > 0 else 0
        logger.info(f"Order rejection rate during extreme conditions: {rejection_rate:.2%}")
        
        # System should still be operational
        test_order = Order(
            order_id="final_test",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=current_price
        )
        
        risk_check = env['risk_manager'].validate_order(test_order)
        assert isinstance(risk_check, dict), "Risk manager should still be operational"

    def test_resource_cleanup(self, stress_environment):
        """Test proper resource cleanup after intensive operations."""
        env = stress_environment
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform intensive operations
        for iteration in range(10):
            # Generate many market updates
            for i in range(1000):
                market_data = {
                    'symbol': 'BTCUSDT',
                    'timestamp': time.time() + i + iteration * 1000,
                    'bid_price': 50000.0 + np.random.uniform(-100, 100),
                    'ask_price': 50001.0 + np.random.uniform(-100, 100),
                    'volume': np.random.uniform(0.1, 10.0)
                }
                
                env['order_book'].process_market_data(market_data)
            
            # Generate many orders
            for i in range(500):
                order = Order(
                    order_id=f"cleanup_test_{iteration}_{i}",
                    symbol='BTCUSDT',
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=np.random.uniform(0.1, 5.0),
                    price=50000.0 + np.random.uniform(-25, 25)
                )
                
                if env['risk_manager'].validate_order(order)['approved']:
                    fill = env['execution_simulator'].execute_order(order)
                    if fill:
                        env['portfolio'].update_position(fill)
            
            # Force cleanup
            gc.collect()
            
            # Check memory doesn't grow unboundedly
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            if memory_growth > 500:  # More than 500MB growth
                logger.warning(f"Memory growth in iteration {iteration}: {memory_growth}MB")
        
        # Final cleanup and verification
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        # System should still be functional
        portfolio_summary = env['portfolio'].get_summary()
        assert isinstance(portfolio_summary, dict)
        
        # Memory growth should be bounded
        assert total_growth < 1000, f"Excessive total memory growth: {total_growth}MB"
        
        logger.info(f"Resource cleanup test completed: {total_growth}MB total growth")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
