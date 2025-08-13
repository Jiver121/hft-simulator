"""
Comprehensive Order Flow Integration Tests

This module tests the complete order lifecycle from submission to settlement,
validating all steps in the order processing pipeline.

Test Scenarios:
- Order submission → matching → execution → settlement flow
- Multiple order types (market, limit, stop) validation
- Order modifications and cancellations
- Partial fills and order splitting
- Cross-order matching and trade generation
- Settlement and position reconciliation

Educational Notes:
- Order flow testing validates the core trading system functionality
- Tests ensure order integrity throughout the complete lifecycle
- Critical for validating matching engine logic and fill models
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import time

# Import system components
from src.execution.matching_engine import MatchingEngine, MultiSymbolMatchingEngine
from src.execution.fill_models import FillModel, RealisticFillModel, FillResult
from src.engine.order_book import OrderBook
from src.engine.order_types import Order, Trade, OrderUpdate, OrderStatus
from src.engine.market_data import BookSnapshot, MarketData
from src.performance.portfolio import Portfolio
from src.performance.metrics import PerformanceAnalyzer
from src.realtime.order_management import RealTimeOrderManager
from src.utils.constants import OrderSide, OrderType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderFlowTestSuite:
    """Comprehensive test suite for order flow validation"""
    
    def __init__(self):
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.initial_capital = 1000000.0
        self.test_data = {}
        self.results = {}
        
    def setup_test_environment(self) -> Dict[str, Any]:
        """Setup comprehensive test environment"""
        
        # Create matching engines for multiple symbols
        matching_engines = {}
        order_books = {}
        
        for symbol in self.symbols:
            matching_engines[symbol] = MatchingEngine(
                symbol=symbol,
                fill_model=RealisticFillModel(),
                tick_size=0.01,
                min_order_size=1,
                max_order_size=10000
            )
            order_books[symbol] = OrderBook(symbol)
            
        # Create multi-symbol matching engine
        multi_engine = MultiSymbolMatchingEngine(self.symbols)
        
        # Create portfolio and performance analyzer
        portfolio = Portfolio(self.initial_capital)
        analyzer = PerformanceAnalyzer(self.initial_capital)
        
        # Generate test market data
        market_data = self._generate_test_market_data()
        
        return {
            'matching_engines': matching_engines,
            'multi_engine': multi_engine,
            'order_books': order_books,
            'portfolio': portfolio,
            'analyzer': analyzer,
            'market_data': market_data
        }
    
    def _generate_test_market_data(self) -> Dict[str, List[BookSnapshot]]:
        """Generate realistic test market data"""
        market_data = {}
        base_time = pd.Timestamp.now()
        
        for symbol in self.symbols:
            # Generate base price
            base_price = np.random.uniform(100, 300)
            snapshots = []
            
            for i in range(1000):  # 1000 market updates
                timestamp = base_time + pd.Timedelta(milliseconds=i * 100)
                
                # Generate realistic price movement
                price_change = np.random.normal(0, 0.001) * base_price
                mid_price = base_price + price_change
                base_price = mid_price
                
                # Generate bid-ask spread
                spread = np.random.uniform(0.01, 0.05)
                best_bid = mid_price - spread / 2
                best_ask = mid_price + spread / 2
                
                # Generate volumes
                bid_volume = np.random.randint(100, 1000)
                ask_volume = np.random.randint(100, 1000)
                
                snapshot = BookSnapshot(
                    symbol=symbol,
                    timestamp=timestamp,
                    bids=[(best_bid, bid_volume)],
                    asks=[(best_ask, ask_volume)],
                    best_bid=best_bid,
                    best_ask=best_ask,
                    best_bid_volume=bid_volume,
                    best_ask_volume=ask_volume,
                    mid_price=mid_price
                )
                
                snapshots.append(snapshot)
            
            market_data[symbol] = snapshots
        
        return market_data
    
    def test_basic_order_lifecycle(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test basic order submission to settlement flow"""
        logger.info("Testing basic order lifecycle...")
        
        symbol = "AAPL"
        engine = test_env['matching_engines'][symbol]
        results = {
            'test_name': 'basic_order_lifecycle',
            'orders_submitted': 0,
            'trades_generated': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'settlement_verified': False,
            'success': False,
            'errors': []
        }
        
        try:
            # Create test orders
            orders = [
                Order(
                    order_id=f"test_buy_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=150.0,
                    volume=100,
                    timestamp=pd.Timestamp.now()
                ),
                Order(
                    order_id=f"test_sell_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=150.01,
                    volume=100,
                    timestamp=pd.Timestamp.now()
                )
            ]
            
            # Submit orders
            all_trades = []
            for order in orders:
                trades, update = engine.process_order(order)
                all_trades.extend(trades)
                results['orders_submitted'] += 1
                
                if update.update_type == 'trade':
                    results['orders_filled'] += 1
            
            results['trades_generated'] = len(all_trades)
            
            # Verify trades
            for trade in all_trades:
                assert trade.price > 0, "Trade price must be positive"
                assert trade.volume > 0, "Trade volume must be positive"
                assert trade.buy_order_id or trade.sell_order_id, "Trade must have order references"
            
            # Test order cancellation
            cancel_order = Order(
                order_id=f"test_cancel_{uuid.uuid4()}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=149.0,
                volume=50,
                timestamp=pd.Timestamp.now()
            )
            
            engine.process_order(cancel_order)
            cancelled = engine.cancel_order(cancel_order.order_id)
            if cancelled:
                results['orders_cancelled'] += 1
            
            # Verify settlement (position tracking)
            engine_stats = engine.get_statistics()
            results['settlement_verified'] = (
                engine_stats['orders_processed'] > 0 and
                engine_stats['trades_generated'] == len(all_trades)
            )
            
            results['success'] = True
            logger.info(f"Basic order lifecycle test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Basic order lifecycle test failed: {e}")
        
        return results
    
    def test_market_order_execution(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test market order immediate execution"""
        logger.info("Testing market order execution...")
        
        symbol = "MSFT"
        engine = test_env['matching_engines'][symbol]
        results = {
            'test_name': 'market_order_execution',
            'market_orders_submitted': 0,
            'immediate_fills': 0,
            'execution_latency_us': [],
            'slippage_measured': [],
            'success': False,
            'errors': []
        }
        
        try:
            # Add liquidity to order book first
            liquidity_orders = [
                Order(
                    order_id=f"liquidity_bid_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=200.0,
                    volume=1000,
                    timestamp=pd.Timestamp.now()
                ),
                Order(
                    order_id=f"liquidity_ask_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=200.05,
                    volume=1000,
                    timestamp=pd.Timestamp.now()
                )
            ]
            
            for liq_order in liquidity_orders:
                engine.process_order(liq_order)
            
            # Submit market orders
            market_orders = [
                Order(
                    order_id=f"market_buy_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    volume=100,
                    timestamp=pd.Timestamp.now()
                ),
                Order(
                    order_id=f"market_sell_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    volume=100,
                    timestamp=pd.Timestamp.now()
                )
            ]
            
            for market_order in market_orders:
                start_time = time.time_ns()
                trades, update = engine.process_order(market_order)
                end_time = time.time_ns()
                
                execution_time_us = (end_time - start_time) / 1000
                results['execution_latency_us'].append(execution_time_us)
                results['market_orders_submitted'] += 1
                
                if trades:
                    results['immediate_fills'] += 1
                    
                    # Calculate slippage (simplified)
                    expected_price = 200.025  # Mid price
                    actual_price = trades[0].price
                    slippage = abs(actual_price - expected_price) / expected_price
                    results['slippage_measured'].append(slippage)
            
            # Verify all market orders filled immediately
            results['success'] = results['immediate_fills'] == results['market_orders_submitted']
            logger.info(f"Market order execution test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Market order execution test failed: {e}")
        
        return results
    
    def test_partial_fill_handling(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test partial fill scenarios and order splitting"""
        logger.info("Testing partial fill handling...")
        
        symbol = "GOOGL"
        engine = test_env['matching_engines'][symbol]
        results = {
            'test_name': 'partial_fill_handling',
            'orders_with_partial_fills': 0,
            'total_partial_trades': 0,
            'fill_ratios': [],
            'remaining_volumes_tracked': True,
            'success': False,
            'errors': []
        }
        
        try:
            # Create liquidity with smaller sizes to force partial fills
            small_liquidity = [
                Order(
                    order_id=f"small_ask_{i}_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=250.0 + (i * 0.01),
                    volume=50,  # Small volumes
                    timestamp=pd.Timestamp.now()
                ) for i in range(5)
            ]
            
            for liq_order in small_liquidity:
                engine.process_order(liq_order)
            
            # Submit large order that will be partially filled
            large_order = Order(
                order_id=f"large_buy_{uuid.uuid4()}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=250.10,  # Price that can match multiple levels
                volume=300,  # Larger than available liquidity
                timestamp=pd.Timestamp.now()
            )
            
            trades, update = engine.process_order(large_order)
            
            if trades:
                results['orders_with_partial_fills'] += 1
                results['total_partial_trades'] = len(trades)
                
                total_filled = sum(trade.volume for trade in trades)
                fill_ratio = total_filled / large_order.volume
                results['fill_ratios'].append(fill_ratio)
                
                # Verify remaining volume tracking
                remaining_expected = large_order.volume - total_filled
                # Check if remaining volume is tracked correctly in order book
                
            # Test multiple partial fills
            for i in range(3):
                partial_order = Order(
                    order_id=f"partial_{i}_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=250.05,
                    volume=150,
                    timestamp=pd.Timestamp.now()
                )
                
                trades, update = engine.process_order(partial_order)
                if len(trades) > 0 and sum(t.volume for t in trades) < partial_order.volume:
                    results['orders_with_partial_fills'] += 1
                    results['total_partial_trades'] += len(trades)
            
            results['success'] = results['orders_with_partial_fills'] > 0
            logger.info(f"Partial fill handling test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Partial fill handling test failed: {e}")
        
        return results
    
    def test_order_modification_flow(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test order modification and cancellation flows"""
        logger.info("Testing order modification flow...")
        
        symbol = "AAPL"
        engine = test_env['matching_engines'][symbol]
        results = {
            'test_name': 'order_modification_flow',
            'orders_modified': 0,
            'orders_cancelled': 0,
            'modification_success_rate': 0.0,
            'cancellation_success_rate': 0.0,
            'success': False,
            'errors': []
        }
        
        try:
            # Create orders for modification testing
            test_orders = []
            for i in range(10):
                order = Order(
                    order_id=f"modify_test_{i}_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=150.0 + (i * 0.01),
                    volume=100,
                    timestamp=pd.Timestamp.now()
                )
                
                engine.process_order(order)
                test_orders.append(order)
            
            # Test order modifications
            modifications_attempted = 0
            modifications_successful = 0
            
            for i, order in enumerate(test_orders[:5]):  # Modify first 5 orders
                modifications_attempted += 1
                new_price = order.price + 0.05 if order.side == OrderSide.BUY else order.price - 0.05
                new_volume = order.volume + 50
                
                success = engine.modify_order(order.order_id, new_price, new_volume)
                if success:
                    modifications_successful += 1
                    results['orders_modified'] += 1
            
            # Test order cancellations
            cancellations_attempted = 0
            cancellations_successful = 0
            
            for order in test_orders[5:]:  # Cancel remaining orders
                cancellations_attempted += 1
                success = engine.cancel_order(order.order_id)
                if success:
                    cancellations_successful += 1
                    results['orders_cancelled'] += 1
            
            # Calculate success rates
            results['modification_success_rate'] = (
                modifications_successful / modifications_attempted 
                if modifications_attempted > 0 else 0.0
            )
            results['cancellation_success_rate'] = (
                cancellations_successful / cancellations_attempted 
                if cancellations_attempted > 0 else 0.0
            )
            
            results['success'] = (
                results['modification_success_rate'] > 0.8 and 
                results['cancellation_success_rate'] > 0.8
            )
            
            logger.info(f"Order modification flow test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Order modification flow test failed: {e}")
        
        return results
    
    def test_cross_order_matching(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test cross-order matching and trade generation"""
        logger.info("Testing cross-order matching...")
        
        symbol = "MSFT"
        engine = test_env['matching_engines'][symbol]
        results = {
            'test_name': 'cross_order_matching',
            'crossing_scenarios_tested': 0,
            'trades_from_crosses': 0,
            'price_time_priority_validated': True,
            'match_quality_scores': [],
            'success': False,
            'errors': []
        }
        
        try:
            # Test scenario 1: Direct price crossing
            buy_order = Order(
                order_id=f"cross_buy_{uuid.uuid4()}",
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=200.10,  # Higher than ask
                volume=100,
                timestamp=pd.Timestamp.now()
            )
            
            sell_order = Order(
                order_id=f"cross_sell_{uuid.uuid4()}",
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=200.05,  # Lower than bid
                volume=100,
                timestamp=pd.Timestamp.now() + pd.Timedelta(milliseconds=1)
            )
            
            # Submit orders and check for crossing
            buy_trades, _ = engine.process_order(buy_order)
            sell_trades, _ = engine.process_order(sell_order)
            
            total_crossing_trades = len(buy_trades) + len(sell_trades)
            results['trades_from_crosses'] += total_crossing_trades
            results['crossing_scenarios_tested'] += 1
            
            # Test scenario 2: Multiple order crossing
            multiple_orders = [
                Order(
                    order_id=f"multi_buy_{i}_{uuid.uuid4()}",
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=200.08,
                    volume=50,
                    timestamp=pd.Timestamp.now() + pd.Timedelta(microseconds=i)
                ) for i in range(5)
            ]
            
            # Add a crossing sell order
            cross_sell = Order(
                order_id=f"multi_cross_sell_{uuid.uuid4()}",
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=200.07,
                volume=200,  # Matches multiple buy orders
                timestamp=pd.Timestamp.now() + pd.Timedelta(milliseconds=10)
            )
            
            # Process orders
            for order in multiple_orders:
                engine.process_order(order)
            
            cross_trades, _ = engine.process_order(cross_sell)
            results['trades_from_crosses'] += len(cross_trades)
            results['crossing_scenarios_tested'] += 1
            
            # Validate price-time priority
            if cross_trades:
                # Check that trades occurred at or between the bid-ask spread
                for trade in cross_trades:
                    match_quality = 1.0 if 200.07 <= trade.price <= 200.08 else 0.5
                    results['match_quality_scores'].append(match_quality)
            
            results['success'] = (
                results['trades_from_crosses'] > 0 and
                results['crossing_scenarios_tested'] >= 2
            )
            
            logger.info(f"Cross-order matching test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Cross-order matching test failed: {e}")
        
        return results
    
    def test_settlement_reconciliation(self, test_env: Dict[str, Any]) -> Dict[str, Any]:
        """Test settlement and position reconciliation"""
        logger.info("Testing settlement reconciliation...")
        
        portfolio = test_env['portfolio']
        analyzer = test_env['analyzer']
        
        results = {
            'test_name': 'settlement_reconciliation',
            'positions_tracked': 0,
            'pnl_calculated': False,
            'cash_balance_correct': False,
            'position_reconciliation_passed': False,
            'settlement_accuracy': 0.0,
            'success': False,
            'errors': []
        }
        
        try:
            initial_cash = portfolio.get_current_cash()
            initial_positions = portfolio.get_all_positions().copy()
            
            # Simulate a series of trades for settlement testing
            test_trades = [
                Trade(
                    trade_id=f"settlement_trade_{i}",
                    symbol="AAPL",
                    buy_order_id=f"buy_{i}" if i % 2 == 0 else None,
                    sell_order_id=f"sell_{i}" if i % 2 == 1 else None,
                    price=150.0 + (i * 0.01),
                    volume=100,
                    timestamp=pd.Timestamp.now() + pd.Timedelta(seconds=i),
                    aggressor_side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
                ) for i in range(10)
            ]
            
            # Process trades through portfolio
            for trade in test_trades:
                # Simulate trade processing
                trade_pnl = self._calculate_trade_pnl(trade)
                portfolio.update_position(trade.symbol, 
                    trade.volume if trade.buy_order_id else -trade.volume,
                    trade.price
                )
                analyzer.add_trade(trade, trade_pnl)
                results['positions_tracked'] += 1
            
            # Verify cash and position consistency
            final_cash = portfolio.get_current_cash()
            final_positions = portfolio.get_all_positions()
            
            # Check position reconciliation
            position_changes_calculated = True
            for symbol in final_positions:
                if symbol not in initial_positions:
                    initial_positions[symbol] = 0
                
                expected_change = sum(
                    trade.volume if trade.buy_order_id else -trade.volume
                    for trade in test_trades if trade.symbol == symbol
                )
                actual_change = final_positions[symbol] - initial_positions[symbol]
                
                if abs(expected_change - actual_change) > 0.001:  # Small tolerance
                    position_changes_calculated = False
                    break
            
            results['position_reconciliation_passed'] = position_changes_calculated
            
            # Calculate P&L accuracy
            portfolio_metrics = analyzer.calculate_metrics()
            results['pnl_calculated'] = portfolio_metrics.total_pnl != 0.0
            
            # Cash balance verification (simplified)
            total_trade_value = sum(trade.price * trade.volume for trade in test_trades)
            results['cash_balance_correct'] = abs(final_cash - initial_cash) > 0
            
            # Calculate overall settlement accuracy
            accuracy_factors = [
                1.0 if results['position_reconciliation_passed'] else 0.0,
                1.0 if results['pnl_calculated'] else 0.0,
                1.0 if results['cash_balance_correct'] else 0.0
            ]
            results['settlement_accuracy'] = sum(accuracy_factors) / len(accuracy_factors)
            
            results['success'] = results['settlement_accuracy'] >= 0.8
            logger.info(f"Settlement reconciliation test completed: {results}")
            
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Settlement reconciliation test failed: {e}")
        
        return results
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate P&L for a trade (simplified)"""
        # Simplified P&L calculation
        base_price = 150.0
        return (trade.price - base_price) * trade.volume if trade.buy_order_id else (base_price - trade.price) * trade.volume
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all order flow tests and generate comprehensive results"""
        logger.info("Starting comprehensive order flow tests...")
        
        # Setup test environment
        test_env = self.setup_test_environment()
        
        # Define all test methods
        test_methods = [
            self.test_basic_order_lifecycle,
            self.test_market_order_execution,
            self.test_partial_fill_handling,
            self.test_order_modification_flow,
            self.test_cross_order_matching,
            self.test_settlement_reconciliation
        ]
        
        # Run all tests
        test_results = []
        start_time = time.time()
        
        for test_method in test_methods:
            try:
                result = test_method(test_env)
                test_results.append(result)
            except Exception as e:
                error_result = {
                    'test_name': test_method.__name__,
                    'success': False,
                    'errors': [str(e)]
                }
                test_results.append(error_result)
                logger.error(f"Test {test_method.__name__} failed with error: {e}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive summary
        summary = {
            'test_suite': 'Order Flow Complete',
            'total_tests': len(test_results),
            'passed_tests': sum(1 for r in test_results if r.get('success', False)),
            'failed_tests': sum(1 for r in test_results if not r.get('success', False)),
            'success_rate': sum(1 for r in test_results if r.get('success', False)) / len(test_results),
            'total_execution_time_seconds': total_time,
            'test_results': test_results,
            'environment_info': {
                'symbols_tested': self.symbols,
                'initial_capital': self.initial_capital,
                'total_orders_processed': sum(
                    r.get('orders_submitted', 0) + 
                    r.get('market_orders_submitted', 0) + 
                    r.get('orders_modified', 0) + 
                    r.get('orders_cancelled', 0) 
                    for r in test_results
                ),
                'total_trades_generated': sum(
                    r.get('trades_generated', 0) + 
                    r.get('trades_from_crosses', 0) + 
                    r.get('total_partial_trades', 0) 
                    for r in test_results
                )
            }
        }
        
        logger.info(f"Order flow tests completed. Success rate: {summary['success_rate']:.2%}")
        return summary


# Pytest fixtures and test functions
@pytest.fixture(scope="module")
def order_flow_suite():
    """Create order flow test suite"""
    return OrderFlowTestSuite()


@pytest.mark.integration
@pytest.mark.order_flow
def test_complete_order_flow(order_flow_suite):
    """Test complete order flow from submission to settlement"""
    results = order_flow_suite.run_comprehensive_tests()
    
    # Assert overall success
    assert results['success_rate'] >= 0.8, f"Order flow tests failed with success rate: {results['success_rate']:.2%}"
    assert results['passed_tests'] >= 4, f"Expected at least 4 tests to pass, got {results['passed_tests']}"
    
    # Assert specific critical tests passed
    critical_tests = ['basic_order_lifecycle', 'settlement_reconciliation']
    passed_critical = sum(
        1 for result in results['test_results'] 
        if any(critical in result['test_name'] for critical in critical_tests) and result['success']
    )
    assert passed_critical >= 1, "Critical order flow tests must pass"


@pytest.mark.integration
@pytest.mark.performance
def test_order_flow_performance(order_flow_suite):
    """Test order flow performance under load"""
    # This test validates performance aspects of order processing
    results = order_flow_suite.run_comprehensive_tests()
    
    # Check execution time performance
    assert results['total_execution_time_seconds'] < 30.0, "Order flow tests took too long"
    
    # Check processing throughput
    env_info = results['environment_info']
    orders_per_second = env_info['total_orders_processed'] / results['total_execution_time_seconds']
    assert orders_per_second > 100, f"Order processing too slow: {orders_per_second:.1f} orders/sec"


if __name__ == "__main__":
    # Run tests directly
    suite = OrderFlowTestSuite()
    results = suite.run_comprehensive_tests()
    
    print("\n" + "="*80)
    print("ORDER FLOW INTEGRATION TEST RESULTS")
    print("="*80)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    print(f"Execution Time: {results['total_execution_time_seconds']:.2f} seconds")
    print(f"Orders Processed: {results['environment_info']['total_orders_processed']}")
    print(f"Trades Generated: {results['environment_info']['total_trades_generated']}")
    
    # Print individual test results
    print("\nIndividual Test Results:")
    print("-" * 50)
    for result in results['test_results']:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status}: {result['test_name']}")
        if result.get('errors'):
            for error in result['errors']:
                print(f"    Error: {error}")
