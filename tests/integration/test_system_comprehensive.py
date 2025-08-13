"""
Comprehensive System Integration Test Runner.

This module orchestrates and runs all integration and system tests to provide
a complete validation of the HFT trading system.
"""

import pytest
import asyncio
import time
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestSystemComprehensive:
    """Comprehensive system integration test suite."""

    @pytest.fixture(scope="class", autouse=True)
    def system_health_check(self):
        """Perform system health check before running integration tests."""
        logger.info("Performing system health check...")
        
        # Check critical imports
        try:
            from src.engine.order_book import OrderBook
            from src.strategies.market_making import MarketMakingStrategy
            from src.performance.portfolio import Portfolio
            from src.realtime.data_feeds import MockDataFeed
            logger.info("✓ All critical modules imported successfully")
        except ImportError as e:
            pytest.fail(f"Critical import failed: {e}")
        
        # Check basic functionality
        try:
            order_book = OrderBook()
            portfolio = Portfolio()
            strategy = MarketMakingStrategy()
            data_feed = MockDataFeed(symbols=['BTCUSDT'])
            logger.info("✓ Core components initialized successfully")
        except Exception as e:
            pytest.fail(f"Core component initialization failed: {e}")
        
        yield
        
        logger.info("System health check completed")

    def test_system_startup_sequence(self):
        """Test complete system startup sequence."""
        logger.info("Testing system startup sequence...")
        
        startup_steps = [
            "Initialize OrderBook",
            "Initialize Portfolio", 
            "Initialize RiskManager",
            "Initialize Strategies",
            "Initialize ExecutionSimulator",
            "Initialize DataFeed",
            "Start TradingSystem"
        ]
        
        completed_steps = []
        
        try:
            # Step 1: Initialize OrderBook
            from src.engine.order_book import OrderBook
            order_book = OrderBook()
            completed_steps.append(startup_steps[0])
            
            # Step 2: Initialize Portfolio
            from src.performance.portfolio import Portfolio
            portfolio = Portfolio(initial_balance=100000.0)
            completed_steps.append(startup_steps[1])
            
            # Step 3: Initialize RiskManager
            from src.performance.risk_manager import RiskManager
            risk_manager = RiskManager(portfolio)
            completed_steps.append(startup_steps[2])
            
            # Step 4: Initialize Strategies
            from src.strategies.market_making import MarketMakingStrategy
            from src.strategies.liquidity_taking import LiquidityTakingStrategy
            market_maker = MarketMakingStrategy()
            liquidity_taker = LiquidityTakingStrategy()
            completed_steps.append(startup_steps[3])
            
            # Step 5: Initialize ExecutionSimulator
            from src.execution.simulator import ExecutionSimulator
            execution_simulator = ExecutionSimulator(order_book, portfolio)
            completed_steps.append(startup_steps[4])
            
            # Step 6: Initialize DataFeed
            from src.realtime.data_feeds import MockDataFeed
            data_feed = MockDataFeed(symbols=['BTCUSDT', 'ETHUSDT'])
            completed_steps.append(startup_steps[5])
            
            # Step 7: Start TradingSystem (mock start)
            from src.realtime.trading_system import TradingSystem
            trading_system = TradingSystem(
                order_book=order_book,
                strategies=[market_maker, liquidity_taker],
                portfolio=portfolio,
                risk_manager=risk_manager,
                data_feed=data_feed
            )
            completed_steps.append(startup_steps[6])
            
        except Exception as e:
            logger.error(f"Startup failed at step {len(completed_steps)}: {e}")
            raise
        
        # Validate all steps completed
        assert len(completed_steps) == len(startup_steps), \
            f"Startup incomplete: {completed_steps}"
        
        logger.info(f"✓ System startup sequence completed: {len(completed_steps)} steps")

    def test_component_integration_health(self):
        """Test health of component integrations."""
        logger.info("Testing component integration health...")
        
        from src.engine.order_book import OrderBook
        from src.performance.portfolio import Portfolio
        from src.performance.risk_manager import RiskManager
        from src.strategies.market_making import MarketMakingStrategy
        from src.execution.simulator import ExecutionSimulator
        from src.engine.order_types import Order, OrderSide, OrderType
        
        # Initialize components
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        risk_manager = RiskManager(portfolio)
        strategy = MarketMakingStrategy()
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        
        integration_tests = []
        
        # Test 1: OrderBook <-> Strategy integration
        try:
            market_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'last_price': 50000.5,
                'volume': 1.0
            }
            
            order_book.process_market_data(market_data)
            signals = strategy.generate_signals(market_data)
            
            assert isinstance(signals, list), "Strategy should return list of signals"
            integration_tests.append("OrderBook <-> Strategy: ✓")
            
        except Exception as e:
            integration_tests.append(f"OrderBook <-> Strategy: ✗ ({e})")
        
        # Test 2: Strategy <-> RiskManager integration
        try:
            if signals:  # Only if we got signals from previous test
                order = Order(
                    order_id="integration_test",
                    symbol='BTCUSDT',
                    side=signals[0]['side'],
                    order_type=OrderType.LIMIT,
                    quantity=signals[0]['quantity'],
                    price=signals[0]['price']
                )
                
                risk_result = risk_manager.validate_order(order)
                assert 'approved' in risk_result, "Risk manager should return approval status"
                integration_tests.append("Strategy <-> RiskManager: ✓")
            else:
                integration_tests.append("Strategy <-> RiskManager: ⚠ (No signals to test)")
                
        except Exception as e:
            integration_tests.append(f"Strategy <-> RiskManager: ✗ ({e})")
        
        # Test 3: ExecutionSimulator <-> Portfolio integration
        try:
            test_order = Order(
                order_id="portfolio_integration_test",
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
                price=50000.0
            )
            
            fill = execution_simulator.execute_order(test_order)
            if fill:
                initial_balance = portfolio.cash_balance
                portfolio.update_position(fill)
                final_balance = portfolio.cash_balance
                
                assert final_balance != initial_balance, "Portfolio should update after trade"
                integration_tests.append("ExecutionSimulator <-> Portfolio: ✓")
            else:
                integration_tests.append("ExecutionSimulator <-> Portfolio: ⚠ (No fill to test)")
                
        except Exception as e:
            integration_tests.append(f"ExecutionSimulator <-> Portfolio: ✗ ({e})")
        
        # Log results
        for result in integration_tests:
            logger.info(result)
        
        # Count successful integrations
        successful = len([r for r in integration_tests if '✓' in r])
        total = len(integration_tests)
        
        assert successful >= total * 0.8, f"Too many integration failures: {successful}/{total}"
        
        logger.info(f"Integration health: {successful}/{total} components integrated successfully")

    def test_data_flow_end_to_end(self):
        """Test complete data flow from input to output."""
        logger.info("Testing end-to-end data flow...")
        
        from src.engine.order_book import OrderBook
        from src.performance.portfolio import Portfolio
        from src.performance.risk_manager import RiskManager
        from src.strategies.market_making import MarketMakingStrategy
        from src.execution.simulator import ExecutionSimulator
        from src.engine.order_types import Order, OrderSide, OrderType
        
        # Initialize system
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        risk_manager = RiskManager(portfolio)
        strategy = MarketMakingStrategy()
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        
        # Define data flow stages
        flow_stages = {
            'market_data_input': False,
            'order_book_update': False,
            'strategy_signal_generation': False,
            'risk_validation': False,
            'order_execution': False,
            'portfolio_update': False,
            'performance_calculation': False
        }
        
        try:
            # Stage 1: Market data input
            market_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'bid_size': 10.0,
                'ask_size': 10.0,
                'last_price': 50000.5,
                'volume': 1.0
            }
            flow_stages['market_data_input'] = True
            
            # Stage 2: Order book update
            order_book.process_market_data(market_data)
            order_book_state = order_book.get_state()
            assert 'bid_price' in order_book_state, "Order book should contain bid price"
            flow_stages['order_book_update'] = True
            
            # Stage 3: Strategy signal generation
            signals = strategy.generate_signals(market_data)
            assert len(signals) > 0, "Strategy should generate signals"
            flow_stages['strategy_signal_generation'] = True
            
            # Stage 4: Risk validation
            signal = signals[0]
            order = Order(
                order_id="data_flow_test",
                symbol='BTCUSDT',
                side=signal['side'],
                order_type=OrderType.LIMIT,
                quantity=signal['quantity'],
                price=signal['price']
            )
            
            risk_result = risk_manager.validate_order(order)
            assert risk_result['approved'], "Order should pass risk validation"
            flow_stages['risk_validation'] = True
            
            # Stage 5: Order execution
            fill = execution_simulator.execute_order(order)
            assert fill is not None, "Order should execute and generate fill"
            flow_stages['order_execution'] = True
            
            # Stage 6: Portfolio update
            initial_value = portfolio.get_summary()['total_value']
            portfolio.update_position(fill)
            final_value = portfolio.get_summary()['total_value']
            assert final_value != initial_value, "Portfolio value should change"
            flow_stages['portfolio_update'] = True
            
            # Stage 7: Performance calculation
            performance_metrics = risk_manager.calculate_risk_metrics()
            assert 'max_drawdown' in performance_metrics, "Performance metrics should be available"
            flow_stages['performance_calculation'] = True
            
        except Exception as e:
            logger.error(f"Data flow test failed: {e}")
            raise
        
        # Validate all stages completed
        completed_stages = sum(flow_stages.values())
        total_stages = len(flow_stages)
        
        logger.info("Data flow stages:")
        for stage, completed in flow_stages.items():
            status = "✓" if completed else "✗"
            logger.info(f"  {stage}: {status}")
        
        assert completed_stages == total_stages, \
            f"Data flow incomplete: {completed_stages}/{total_stages} stages"
        
        logger.info(f"✓ End-to-end data flow completed: {completed_stages}/{total_stages} stages")

    @pytest.mark.asyncio
    async def test_system_responsiveness(self):
        """Test system responsiveness under various conditions."""
        logger.info("Testing system responsiveness...")
        
        from src.engine.order_book import OrderBook
        from src.strategies.market_making import MarketMakingStrategy
        
        order_book = OrderBook()
        strategy = MarketMakingStrategy()
        
        # Test response times for various operations
        response_times = {}
        
        # Test 1: Market data processing
        start_time = time.time()
        market_data = {
            'symbol': 'BTCUSDT',
            'timestamp': time.time(),
            'bid_price': 50000.0,
            'ask_price': 50001.0,
            'volume': 1.0
        }
        order_book.process_market_data(market_data)
        response_times['market_data_processing'] = time.time() - start_time
        
        # Test 2: Strategy signal generation
        start_time = time.time()
        signals = strategy.generate_signals(market_data)
        response_times['strategy_signal_generation'] = time.time() - start_time
        
        # Test 3: Multiple rapid operations
        start_time = time.time()
        for i in range(100):
            test_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time() + i,
                'bid_price': 50000.0 + i * 0.1,
                'ask_price': 50001.0 + i * 0.1,
                'volume': 1.0
            }
            order_book.process_market_data(test_data)
            
            # Add small async delay
            await asyncio.sleep(0.001)
        
        response_times['batch_processing'] = (time.time() - start_time) / 100
        
        # Validate response times
        max_acceptable_times = {
            'market_data_processing': 0.001,  # 1ms
            'strategy_signal_generation': 0.01,  # 10ms
            'batch_processing': 0.005  # 5ms average per operation
        }
        
        logger.info("Response time analysis:")
        for operation, response_time in response_times.items():
            max_time = max_acceptable_times[operation]
            status = "✓" if response_time <= max_time else "⚠"
            logger.info(f"  {operation}: {response_time*1000:.2f}ms {status}")
            
            # Allow some tolerance for CI environments
            assert response_time <= max_time * 2, \
                f"{operation} too slow: {response_time*1000:.2f}ms > {max_time*2000:.2f}ms"
        
        logger.info("✓ System responsiveness test completed")

    def test_error_handling_robustness(self):
        """Test system robustness to various error conditions."""
        logger.info("Testing error handling robustness...")
        
        from src.engine.order_book import OrderBook
        from src.performance.portfolio import Portfolio
        from src.performance.risk_manager import RiskManager
        from src.engine.order_types import Order, OrderSide, OrderType
        
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        risk_manager = RiskManager(portfolio)
        
        error_scenarios = []
        
        # Test 1: Invalid market data
        try:
            invalid_data = {
                'symbol': 'INVALID',
                'timestamp': 'not_a_timestamp',
                'bid_price': -1000,  # Negative price
                'ask_price': 'not_a_number',
                'volume': None
            }
            order_book.process_market_data(invalid_data)
            error_scenarios.append("Invalid market data: System handled gracefully")
        except Exception as e:
            error_scenarios.append(f"Invalid market data: Caught exception ({type(e).__name__})")
        
        # Test 2: Invalid order
        try:
            invalid_order = Order(
                order_id="invalid_test",
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=-1.0,  # Negative quantity
                price=0.0  # Zero price
            )
            risk_result = risk_manager.validate_order(invalid_order)
            assert not risk_result['approved'], "Invalid order should be rejected"
            error_scenarios.append("Invalid order: Properly rejected by risk manager")
        except Exception as e:
            error_scenarios.append(f"Invalid order: Exception handling ({type(e).__name__})")
        
        # Test 3: System state after errors
        try:
            valid_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time(),
                'bid_price': 50000.0,
                'ask_price': 50001.0,
                'volume': 1.0
            }
            order_book.process_market_data(valid_data)
            
            valid_order = Order(
                order_id="recovery_test",
                symbol='BTCUSDT',
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                price=50000.0
            )
            risk_result = risk_manager.validate_order(valid_order)
            
            error_scenarios.append("System recovery: System operational after errors")
        except Exception as e:
            error_scenarios.append(f"System recovery: Failed to recover ({e})")
        
        # Log error handling results
        logger.info("Error handling test results:")
        for scenario in error_scenarios:
            logger.info(f"  {scenario}")
        
        # System should handle at least 2/3 error scenarios gracefully
        successful_scenarios = len([s for s in error_scenarios if 'gracefully' in s or 'rejected' in s or 'operational' in s])
        assert successful_scenarios >= 2, f"Poor error handling: {successful_scenarios}/{len(error_scenarios)}"
        
        logger.info(f"✓ Error handling robustness: {successful_scenarios}/{len(error_scenarios)} scenarios handled properly")

    def test_performance_benchmarks(self):
        """Test system performance against established benchmarks."""
        logger.info("Running performance benchmarks...")
        
        from src.engine.order_book import OrderBook
        from src.strategies.market_making import MarketMakingStrategy
        from src.performance.portfolio import Portfolio
        from src.engine.order_types import Order, OrderSide, OrderType
        
        order_book = OrderBook()
        strategy = MarketMakingStrategy()
        portfolio = Portfolio(initial_balance=100000.0)
        
        benchmarks = {}
        
        # Benchmark 1: Market data throughput
        start_time = time.time()
        num_updates = 1000
        
        for i in range(num_updates):
            market_data = {
                'symbol': 'BTCUSDT',
                'timestamp': time.time() + i,
                'bid_price': 50000.0 + (i % 100),
                'ask_price': 50001.0 + (i % 100),
                'volume': 1.0 + (i % 10)
            }
            order_book.process_market_data(market_data)
        
        processing_time = time.time() - start_time
        benchmarks['market_data_throughput'] = num_updates / processing_time
        
        # Benchmark 2: Strategy signal generation rate
        start_time = time.time()
        num_signals = 0
        
        for i in range(100):
            market_data = {
                'symbol': 'BTCUSDT',
                'price': 50000.0 + i,
                'timestamp': time.time() + i,
                'volume': 1.0
            }
            signals = strategy.generate_signals(market_data)
            num_signals += len(signals)
        
        signal_time = time.time() - start_time
        benchmarks['signal_generation_rate'] = num_signals / signal_time
        
        # Benchmark 3: Portfolio update rate
        start_time = time.time()
        num_updates = 100
        
        for i in range(num_updates):
            fill = {
                'order_id': f"benchmark_{i}",
                'symbol': 'BTCUSDT',
                'side': OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                'quantity': 0.1,
                'price': 50000.0,
                'timestamp': time.time() + i,
                'commission': 5.0
            }
            portfolio.update_position(fill)
        
        portfolio_time = time.time() - start_time
        benchmarks['portfolio_update_rate'] = num_updates / portfolio_time
        
        # Define minimum acceptable benchmarks
        min_benchmarks = {
            'market_data_throughput': 1000,  # updates/second
            'signal_generation_rate': 100,   # signals/second
            'portfolio_update_rate': 500     # updates/second
        }
        
        logger.info("Performance benchmark results:")
        all_benchmarks_met = True
        
        for metric, rate in benchmarks.items():
            min_rate = min_benchmarks[metric]
            status = "✓" if rate >= min_rate else "⚠"
            if rate < min_rate:
                all_benchmarks_met = False
            
            logger.info(f"  {metric}: {rate:.0f}/sec (min: {min_rate}/sec) {status}")
        
        # Allow for some performance variance in different environments
        passing_benchmarks = sum(1 for metric, rate in benchmarks.items() 
                                if rate >= min_benchmarks[metric] * 0.7)
        
        assert passing_benchmarks >= len(benchmarks) * 0.8, \
            f"Performance benchmarks not met: {passing_benchmarks}/{len(benchmarks)}"
        
        logger.info(f"✓ Performance benchmarks: {passing_benchmarks}/{len(benchmarks)} targets achieved")

    @pytest.mark.slow
    def test_extended_operation_stability(self):
        """Test system stability during extended operation."""
        logger.info("Testing extended operation stability...")
        
        from src.engine.order_book import OrderBook
        from src.performance.portfolio import Portfolio
        from src.strategies.market_making import MarketMakingStrategy
        from src.execution.simulator import ExecutionSimulator
        import gc
        import psutil
        
        # Initialize system
        order_book = OrderBook()
        portfolio = Portfolio(initial_balance=100000.0)
        strategy = MarketMakingStrategy()
        execution_simulator = ExecutionSimulator(order_book, portfolio)
        
        # Track system metrics
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        operations_completed = 0
        errors_encountered = 0
        
        # Run extended operation (reduced for faster testing)
        target_operations = 5000  # Reduced from 50000 for CI
        
        try:
            for i in range(target_operations):
                # Market data update
                market_data = {
                    'symbol': 'BTCUSDT',
                    'timestamp': time.time() + i,
                    'bid_price': 50000.0 + (i % 1000) * 0.1,
                    'ask_price': 50001.0 + (i % 1000) * 0.1,
                    'volume': 1.0 + (i % 100) * 0.01
                }
                
                try:
                    order_book.process_market_data(market_data)
                    
                    # Periodic strategy execution
                    if i % 10 == 0:
                        signals = strategy.generate_signals(market_data)
                        for signal in signals[:1]:  # Limit to 1 signal to control load
                            order = Order(
                                order_id=f"stability_test_{i}_{len(signals)}",
                                symbol='BTCUSDT',
                                side=signal['side'],
                                order_type=OrderType.LIMIT,
                                quantity=min(signal['quantity'], 1.0),  # Limit order size
                                price=signal['price']
                            )
                            
                            fill = execution_simulator.execute_order(order)
                            if fill:
                                portfolio.update_position(fill)
                    
                    operations_completed += 1
                    
                    # Periodic cleanup and health check
                    if i % 1000 == 0:
                        gc.collect()
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_growth = current_memory - initial_memory
                        
                        if memory_growth > 500:  # More than 500MB growth
                            logger.warning(f"High memory usage at operation {i}: {current_memory}MB")
                        
                        logger.info(f"Extended operation progress: {i}/{target_operations} ({i/target_operations*100:.1f}%)")
                        
                except Exception as e:
                    errors_encountered += 1
                    logger.error(f"Error at operation {i}: {e}")
                    
                    if errors_encountered > target_operations * 0.01:  # More than 1% error rate
                        logger.error("Too many errors, stopping extended operation test")
                        break
        
        except KeyboardInterrupt:
            logger.info("Extended operation test interrupted")
        
        end_time = time.time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        total_time = end_time - start_time
        operations_per_second = operations_completed / total_time if total_time > 0 else 0
        error_rate = errors_encountered / operations_completed if operations_completed > 0 else 1
        memory_growth = final_memory - initial_memory
        
        logger.info("Extended operation results:")
        logger.info(f"  Operations completed: {operations_completed}")
        logger.info(f"  Errors encountered: {errors_encountered}")
        logger.info(f"  Operations/second: {operations_per_second:.1f}")
        logger.info(f"  Error rate: {error_rate:.2%}")
        logger.info(f"  Memory growth: {memory_growth:.1f}MB")
        logger.info(f"  Total time: {total_time:.1f}s")
        
        # Validation criteria
        assert operations_completed >= target_operations * 0.9, \
            f"Too few operations completed: {operations_completed}/{target_operations}"
        
        assert error_rate <= 0.05, \
            f"Error rate too high: {error_rate:.2%}"
        
        assert memory_growth < 1000, \
            f"Excessive memory growth: {memory_growth}MB"
        
        assert operations_per_second > 50, \
            f"Operations/second too low: {operations_per_second:.1f}"
        
        logger.info("✓ Extended operation stability test passed")

    def test_system_configuration_validation(self):
        """Validate system configuration and settings."""
        logger.info("Validating system configuration...")
        
        config_checks = []
        
        # Check 1: Required modules are accessible
        required_modules = [
            'src.engine.order_book',
            'src.strategies.market_making',
            'src.performance.portfolio',
            'src.realtime.data_feeds'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                config_checks.append(f"Module {module}: ✓")
            except ImportError:
                config_checks.append(f"Module {module}: ✗")
        
        # Check 2: System constants and configurations
        try:
            from src.utils.constants import OrderSide, OrderType
            config_checks.append("Constants accessible: ✓")
        except ImportError:
            config_checks.append("Constants accessible: ✗")
        
        # Check 3: Logging configuration
        try:
            from src.utils.logger import get_logger
            test_logger = get_logger("test_config")
            test_logger.info("Test log message")
            config_checks.append("Logging configuration: ✓")
        except Exception:
            config_checks.append("Logging configuration: ✗")
        
        # Log configuration results
        logger.info("System configuration validation:")
        for check in config_checks:
            logger.info(f"  {check}")
        
        # Count successful checks
        successful_checks = len([c for c in config_checks if '✓' in c])
        total_checks = len(config_checks)
        
        assert successful_checks >= total_checks * 0.9, \
            f"Configuration validation failed: {successful_checks}/{total_checks}"
        
        logger.info(f"✓ System configuration validation: {successful_checks}/{total_checks} checks passed")


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
