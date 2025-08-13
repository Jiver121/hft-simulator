#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for HFT Simulator

This test suite validates the complete system integration by testing:
- Order flow from strategy to execution
- P&L calculation accuracy
- Position tracking
- Multi-symbol/multi-strategy operations
- Performance metrics computation
- Data processing pipeline
- Memory usage and performance benchmarks
"""

import unittest
import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
import tempfile
import shutil

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import HFT Simulator components
from src.execution.simulator import ExecutionSimulator, BacktestResult
from src.execution.fill_models import RealisticFillModel, PerfectFillModel
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot
from src.data.ingestion import DataIngestion
from src.data.preprocessor import DataPreprocessor
from src.performance.portfolio import Portfolio, Position
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.utils.helpers import Timer

# Import main components for end-to-end testing
from main import BatchBacktester, BacktestConfig, create_sample_data

warnings.filterwarnings('ignore')

# Configure test-specific logging to avoid Windows file locking issues
import logging
def get_test_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class SimpleTestStrategy(BaseStrategy):
    """Simple test strategy for integration testing"""
    
    def __init__(self, symbol: str, order_frequency: int = 10):
        super().__init__("test_strategy", symbol)
        self.order_frequency = order_frequency
        self.tick_count = 0
        self.orders_generated = 0
        
        # Set data mode to backtest
        self.set_data_mode("backtest")
        
    def on_market_update(self, snapshot: 'BookSnapshot', timestamp: pd.Timestamp) -> 'StrategyResult':
        """Generate predictable orders for testing"""
        result = StrategyResult(timestamp=timestamp)
        
        if not snapshot or not snapshot.mid_price:
            return result
        
        self.tick_count += 1
        
        # Generate order every N ticks
        if self.tick_count % self.order_frequency == 0:
            if snapshot.best_bid and snapshot.best_ask:
                # Alternate between buy and sell orders with market orders for better execution
                if self.orders_generated % 2 == 0:
                    # Market buy order - should execute immediately
                    order = self.create_order(
                        side=OrderSide.BUY,
                        volume=100,
                        price=None,  # Market orders must have price=None
                        order_type=OrderType.MARKET,
                        reason="Test buy order"
                    )
                else:
                    # Market sell order - should execute immediately
                    order = self.create_order(
                        side=OrderSide.SELL,
                        volume=100,
                        price=None,  # Market orders must have price=None
                        order_type=OrderType.MARKET,
                        reason="Test sell order"
                    )
                
                result.add_order(order, f"Test order #{self.orders_generated}")
                self.submit_order(order)
                self.orders_generated += 1
        
        return result


class IntegrationTestSuite(unittest.TestCase):
    """Comprehensive integration test suite"""
    
    def setUp(self):
        """Set up test environment"""
        self.logger = get_test_logger(f"{__name__}.test")
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="hft_test_"))
        self.test_output_dir = Path(tempfile.mkdtemp(prefix="hft_output_"))
        
        # Generate test data - use valid symbols (uppercase letters only)
        self.test_symbols = ["TESTA", "TESTB", "TESTC"]
        self.generate_test_data()
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            if self.test_data_dir.exists():
                shutil.rmtree(self.test_data_dir)
            if self.test_output_dir.exists():
                shutil.rmtree(self.test_output_dir)
        except Exception as e:
            self.logger.warning(f"Failed to clean up test directories: {e}")
    
    def generate_test_data(self):
        """Generate synthetic test data"""
        base_prices = {"TESTA": 100.0, "TESTB": 50.0, "TESTC": 200.0}
        
        for symbol in self.test_symbols:
            np.random.seed(hash(symbol) % 1000)
            
            # Generate time series
            n_ticks = 1000
            base_time = pd.Timestamp('2024-01-01 09:30:00')
            timestamps = [base_time + pd.Timedelta(seconds=i) for i in range(n_ticks)]
            
            # Generate realistic price movements
            current_price = base_prices[symbol]
            data = []
            
            for i, timestamp in enumerate(timestamps):
                # Random walk with mean reversion
                price_change = np.random.normal(0, 0.002 * current_price)
                current_price += price_change
                current_price = max(current_price * 0.95, min(current_price * 1.05, current_price))
                
                # Generate bid/ask spread
                spread = max(0.01, current_price * 0.0005)  # 5 bps spread
                bid_price = round(current_price - spread/2, 2)
                ask_price = round(current_price + spread/2, 2)
                
                # Generate volume
                volume = np.random.randint(100, 1000)
                
                data.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'volume': volume,
                    'bid': bid_price,
                    'ask': ask_price,
                    'bid_volume': np.random.randint(100, 500),
                    'ask_volume': np.random.randint(100, 500)
                })
            
            # Save to CSV
            df = pd.DataFrame(data)
            csv_file = self.test_data_dir / f"{symbol}_data.csv"
            df.to_csv(csv_file, index=False)
    
    def test_01_basic_order_flow(self):
        """Test basic order flow from strategy to execution"""
        self.logger.info("Testing basic order flow...")
        
        symbol = "TESTA"
        csv_file = self.test_data_dir / f"{symbol}_data.csv"
        
        # Create simulator
        simulator = ExecutionSimulator(
            symbol=symbol,
            fill_model=PerfectFillModel(),  # Use perfect fills for predictable testing
            initial_cash=100000.0
        )
        
        # Create test strategy
        strategy = SimpleTestStrategy(symbol, order_frequency=50)  # Generate orders every 50 ticks
        
        # Load test data
        data = pd.read_csv(csv_file)
        
        # Run backtest
        result = simulator.run_backtest(
            data_source=data,
            strategy=strategy
        )
        
        # Validate results
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.symbol, symbol)
        self.assertGreater(len(result.orders), 0, "No orders were generated")
        self.assertGreater(len(result.trades), 0, "No trades were executed")
        
        # Check that orders led to trades (with perfect fill model)
        self.logger.info(f"Generated {len(result.orders)} orders, executed {len(result.trades)} trades")
        self.assertGreater(result.fill_rate, 0.0, "Fill rate should be positive")
        
        # Check basic metrics calculation
        self.assertIsNotNone(result.total_pnl)
        self.assertIsNotNone(result.win_rate)
        self.assertIsNotNone(result.max_drawdown)
    
    def test_02_pnl_calculation_accuracy(self):
        """Test P&L calculation accuracy with known trades"""
        self.logger.info("Testing P&L calculation accuracy...")
        
        symbol = "TESTA"
        
        # Create simulator
        simulator = ExecutionSimulator(
            symbol=symbol,
            fill_model=PerfectFillModel(),
            initial_cash=100000.0
        )
        
        # Create manual test data with known price movements
        test_data = []
        base_time = pd.Timestamp('2024-01-01 09:30:00')
        
        # Simple price sequence: 100, 101, 102, 101, 100
        prices = [100.0, 101.0, 102.0, 101.0, 100.0]
        
        for i, price in enumerate(prices):
            spread = 0.02
            test_data.append({
                'timestamp': base_time + pd.Timedelta(seconds=i),
                'symbol': symbol,
                'price': price,
                'volume': 100,
                'bid': price - spread/2,
                'ask': price + spread/2,
                'bid_volume': 100,
                'ask_volume': 100
            })
        
        df = pd.DataFrame(test_data)
        
        # Create strategy that makes predictable trades
        class PredictablePnLStrategy(SimpleTestStrategy):
            def on_market_update(self, snapshot, timestamp):
                result = StrategyResult(timestamp=timestamp)
                
                if not snapshot or not snapshot.mid_price:
                    return result
                
                # Buy at 100, sell at 102 for known P&L
                if abs(snapshot.mid_price - 100.0) < 0.01 and self.orders_generated == 0:
                    # Buy order
                    order = self.create_order(
                        side=OrderSide.BUY,
                        volume=100,
                        price=snapshot.best_ask,
                        order_type=OrderType.LIMIT,
                        reason="Known buy"
                    )
                    result.add_order(order, "Buy at 100")
                    self.submit_order(order)
                    self.orders_generated += 1
                elif abs(snapshot.mid_price - 102.0) < 0.01 and self.orders_generated == 1:
                    # Sell order
                    order = self.create_order(
                        side=OrderSide.SELL,
                        volume=100,
                        price=snapshot.best_bid,
                        order_type=OrderType.LIMIT,
                        reason="Known sell"
                    )
                    result.add_order(order, "Sell at 102")
                    self.submit_order(order)
                    self.orders_generated += 1
                
                return result
        
        strategy = PredictablePnLStrategy(symbol)
        
        # Run backtest
        result = simulator.run_backtest(
            data_source=df,
            strategy=strategy
        )
        
        # Expected P&L: Buy 100 shares at ~100.01, sell at ~101.99 = ~199 profit
        expected_pnl = 100 * (101.99 - 100.01)  # Approximate
        
        self.logger.info(f"Calculated P&L: ${result.total_pnl:.2f}, Expected: ~${expected_pnl:.2f}")
        
        # Check if P&L is in reasonable range (allowing for small execution differences)
        self.assertGreater(result.total_pnl, expected_pnl * 0.8, "P&L is too low")
        self.assertLess(result.total_pnl, expected_pnl * 1.2, "P&L is too high")
    
    def test_03_position_tracking(self):
        """Test position tracking accuracy"""
        self.logger.info("Testing position tracking...")
        
        symbol = "TESTA"
        csv_file = self.test_data_dir / f"{symbol}_data.csv"
        
        simulator = ExecutionSimulator(
            symbol=symbol,
            fill_model=PerfectFillModel(),
            initial_cash=100000.0
        )
        
        strategy = SimpleTestStrategy(symbol, order_frequency=20)
        data = pd.read_csv(csv_file)
        
        result = simulator.run_backtest(
            data_source=data,
            strategy=strategy
        )
        
        # Manually calculate position from trades
        calculated_position = 0
        for trade in result.trades:
            if trade.is_buy_aggressor():
                calculated_position += trade.volume
            else:
                calculated_position -= trade.volume
        
        # Get final position from simulator
        final_state = simulator.get_current_state()
        actual_position = final_state['current_position']
        
        self.logger.info(f"Calculated position: {calculated_position}, Actual position: {actual_position}")
        self.assertEqual(calculated_position, actual_position, "Position tracking mismatch")
    
    def test_04_multi_symbol_backtest(self):
        """Test multi-symbol backtesting"""
        self.logger.info("Testing multi-symbol backtesting...")
        
        config = BacktestConfig()
        config.strategy_type = "market_making"
        config.initial_capital = 100000.0
        
        backtester = BatchBacktester(config)
        
        # Run batch backtest on all test symbols
        results = backtester.run_batch_backtest(
            data_path=str(self.test_data_dir),
            output_path=str(self.test_output_dir)
        )
        
        # Validate results
        self.assertEqual(len(results), len(self.test_symbols), "Should have results for all symbols")
        
        for result in results:
            self.assertIsInstance(result, BacktestResult)
            self.assertIn(result.symbol, self.test_symbols)
            self.assertIsNotNone(result.total_pnl)
    
    def test_05_strategy_comparison(self):
        """Test strategy comparison functionality"""
        self.logger.info("Testing strategy comparison...")
        
        symbol = "TESTA"
        csv_file = self.test_data_dir / f"{symbol}_data.csv"
        data = pd.read_csv(csv_file)
        
        # Test market making strategy
        mm_config = BacktestConfig()
        mm_config.strategy_type = "market_making"
        mm_backtester = BatchBacktester(mm_config)
        
        mm_result = mm_backtester.run_single_backtest({
            'file_path': str(csv_file),
            'symbol': symbol
        })
        
        # Test momentum strategy  
        momentum_config = BacktestConfig()
        momentum_config.strategy_type = "momentum"
        momentum_backtester = BatchBacktester(momentum_config)
        
        momentum_result = momentum_backtester.run_single_backtest({
            'file_path': str(csv_file),
            'symbol': symbol
        })
        
        # Both strategies should produce results
        self.assertIsNotNone(mm_result)
        self.assertIsNotNone(momentum_result)
        
        # Results should have different characteristics
        self.logger.info(f"Market Making - Orders: {len(mm_result.orders)}, Trades: {len(mm_result.trades)}")
        self.logger.info(f"Momentum - Orders: {len(momentum_result.orders)}, Trades: {len(momentum_result.trades)}")
    
    def test_06_performance_metrics(self):
        """Test performance metrics calculation"""
        self.logger.info("Testing performance metrics calculation...")
        
        symbol = "TESTA"
        csv_file = self.test_data_dir / f"{symbol}_data.csv"
        
        simulator = ExecutionSimulator(
            symbol=symbol,
            fill_model=RealisticFillModel(),
            initial_cash=100000.0
        )
        
        strategy = SimpleTestStrategy(symbol, order_frequency=30)
        data = pd.read_csv(csv_file)
        
        result = simulator.run_backtest(
            data_source=data,
            strategy=strategy
        )
        
        # Validate all key metrics are calculated
        metrics_to_check = [
            'total_pnl', 'total_trades', 'total_volume', 'win_rate',
            'fill_rate', 'max_drawdown', 'duration'
        ]
        
        for metric in metrics_to_check:
            self.assertTrue(hasattr(result, metric), f"Missing metric: {metric}")
            value = getattr(result, metric)
            self.assertIsNotNone(value, f"Metric {metric} is None")
        
        # Validate metric ranges
        self.assertGreaterEqual(result.win_rate, 0.0)
        self.assertLessEqual(result.win_rate, 1.0)
        self.assertGreaterEqual(result.fill_rate, 0.0)
        self.assertLessEqual(result.fill_rate, 1.0)
        self.assertGreaterEqual(result.max_drawdown, 0.0)
    
    def test_07_data_validation(self):
        """Test data validation and preprocessing"""
        self.logger.info("Testing data validation...")
        
        # Test data ingestion
        data_ingestion = DataIngestion()
        csv_file = self.test_data_dir / "TESTA_data.csv"
        
        data = data_ingestion.load_csv(csv_file)
        self.assertFalse(data.empty, "Data should not be empty")
        
        # Test data preprocessing
        preprocessor = DataPreprocessor(tick_size=0.01)
        processed_data = preprocessor.process_tick_data(data)
        
        self.assertFalse(processed_data.empty, "Processed data should not be empty")
        
        # Test data validation
        simulator = ExecutionSimulator("TESTA", initial_cash=100000.0)
        validation_report = simulator.validate_data_integrity(processed_data)
        
        self.assertTrue(validation_report['is_valid'], f"Data validation failed: {validation_report['issues']}")
    
    def test_08_memory_and_performance(self):
        """Test memory usage and performance benchmarks"""
        self.logger.info("Testing memory and performance...")
        
        # Create larger dataset for performance testing
        large_symbol = "PERFTEST"
        n_ticks = 10000
        
        # Generate large dataset
        base_time = pd.Timestamp('2024-01-01 09:30:00')
        large_data = []
        current_price = 100.0
        
        for i in range(n_ticks):
            current_price += np.random.normal(0, 0.01)
            spread = 0.02
            
            large_data.append({
                'timestamp': base_time + pd.Timedelta(milliseconds=i),
                'symbol': large_symbol,
                'price': round(current_price, 2),
                'volume': np.random.randint(100, 1000),
                'bid': round(current_price - spread/2, 2),
                'ask': round(current_price + spread/2, 2),
                'bid_volume': 100,
                'ask_volume': 100
            })
        
        df = pd.DataFrame(large_data)
        
        # Test performance
        simulator = ExecutionSimulator(
            symbol=large_symbol,
            fill_model=RealisticFillModel(),
            initial_cash=100000.0
        )
        
        strategy = SimpleTestStrategy(large_symbol, order_frequency=100)
        
        with Timer() as timer:
            result = simulator.run_backtest(
                data_source=df,
                strategy=strategy
            )
        
        processing_time = timer.elapsed()
        ticks_per_second = n_ticks / (processing_time / 1000)
        
        self.logger.info(f"Processed {n_ticks:,} ticks in {processing_time:.1f}ms")
        self.logger.info(f"Performance: {ticks_per_second:.0f} ticks/second")
        
        # Performance benchmarks
        self.assertGreater(ticks_per_second, 1000, "Performance should be > 1000 ticks/second")
        self.assertIsNotNone(result)
    
    def test_09_error_handling(self):
        """Test error handling and edge cases"""
        self.logger.info("Testing error handling...")
        
        # Test with empty data
        empty_data = pd.DataFrame()
        simulator = ExecutionSimulator("TEST", initial_cash=100000.0)
        
        with self.assertRaises(ValueError):
            simulator.run_backtest(
                data_source=empty_data,
                strategy=SimpleTestStrategy("TEST")
            )
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')],
            'symbol': ['TEST'],
            'price': [-100],  # Invalid negative price
            'volume': [100],
            'bid': [99],
            'ask': [101]
        })
        
        # Should handle gracefully
        result = simulator.run_backtest(
            data_source=invalid_data,
            strategy=SimpleTestStrategy("TEST")
        )
        
        # Should complete without crashing
        self.assertIsInstance(result, BacktestResult)
    
    def test_10_progress_tracking(self):
        """Test progress tracking functionality"""
        self.logger.info("Testing progress tracking...")
        
        symbol = "TESTA"
        csv_file = self.test_data_dir / f"{symbol}_data.csv"
        
        simulator = ExecutionSimulator(
            symbol=symbol,
            fill_model=RealisticFillModel(),
            initial_cash=100000.0
        )
        
        # Configure progress tracking
        simulator.set_config(
            show_progress_bar=True,
            verbose_progress=True,
            progress_update_frequency=50
        )
        
        strategy = SimpleTestStrategy(symbol, order_frequency=25)
        data = pd.read_csv(csv_file)
        
        # Run backtest with progress tracking
        result = simulator.run_backtest(
            data_source=data,
            strategy=strategy
        )
        
        # Check that progress tracking variables were updated
        self.assertGreater(simulator.orders_generated_count, 0, "Orders counter should be > 0")
        self.assertGreater(simulator.trades_executed_count, 0, "Trades counter should be > 0")
        self.assertGreater(simulator.ticks_processed_count, 0, "Ticks counter should be > 0")
        
        self.logger.info(f"Progress tracking: {simulator.orders_generated_count} orders, "
                        f"{simulator.trades_executed_count} trades, "
                        f"{simulator.ticks_processed_count} ticks")


def run_integration_tests(test_type: Optional[str] = None, generate_report: bool = False) -> Dict[str, Any]:
    """
    Run integration tests with optional filtering
    
    Args:
        test_type: Type of tests to run ('basic', 'comprehensive', 'performance', 'stress')
        generate_report: Whether to generate detailed test report
    
    Returns:
        Test results summary
    """
    logger = get_test_logger("integration_tests")
    
    # Configure test suite based on type
    if test_type == 'basic':
        test_methods = [
            'test_01_basic_order_flow',
            'test_02_pnl_calculation_accuracy',
            'test_03_position_tracking'
        ]
    elif test_type == 'performance':
        test_methods = [
            'test_08_memory_and_performance'
        ]
    elif test_type == 'stress':
        test_methods = [
            'test_04_multi_symbol_backtest',
            'test_08_memory_and_performance',
            'test_09_error_handling'
        ]
    elif test_type == 'comprehensive':
        test_methods = None  # Run all tests
    else:
        test_methods = None  # Default to all tests
    
    # Create test suite
    if test_methods:
        suite = unittest.TestSuite()
        for method_name in test_methods:
            suite.addTest(IntegrationTestSuite(method_name))
    else:
        suite = unittest.TestLoader().loadTestsFromTestCase(IntegrationTestSuite)
    
    # Run tests
    logger.info(f"Running {test_type or 'all'} integration tests...")
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    test_result = runner.run(suite)
    execution_time = time.time() - start_time
    
    # Compile results
    results_summary = {
        'test_type': test_type or 'comprehensive',
        'execution_time_seconds': execution_time,
        'tests_run': test_result.testsRun,
        'failures': len(test_result.failures),
        'errors': len(test_result.errors),
        'success_rate': (test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun if test_result.testsRun > 0 else 0,
        'timestamp': datetime.now().isoformat(),
        'details': {
            'failures': [(str(test), error) for test, error in test_result.failures],
            'errors': [(str(test), error) for test, error in test_result.errors]
        }
    }
    
    # Generate report if requested
    if generate_report:
        report_file = Path("test_results") / f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Test report saved to: {report_file}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("INTEGRATION TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Test Type: {results_summary['test_type'].title()}")
    logger.info(f"Tests Run: {results_summary['tests_run']}")
    logger.info(f"Passed: {results_summary['tests_run'] - results_summary['failures'] - results_summary['errors']}")
    logger.info(f"Failed: {results_summary['failures']}")
    logger.info(f"Errors: {results_summary['errors']}")
    logger.info(f"Success Rate: {results_summary['success_rate']:.1%}")
    logger.info(f"Execution Time: {results_summary['execution_time_seconds']:.2f} seconds")
    logger.info(f"{'='*60}")
    
    return results_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run HFT Simulator Integration Tests')
    parser.add_argument('--type', choices=['basic', 'comprehensive', 'performance', 'stress'], 
                       help='Type of tests to run')
    parser.add_argument('--generate-report', action='store_true', 
                       help='Generate detailed test report')
    
    args = parser.parse_args()
    
    try:
        results = run_integration_tests(args.type, args.generate_report)
        exit_code = 0 if results['success_rate'] == 1.0 else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"Integration tests failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
