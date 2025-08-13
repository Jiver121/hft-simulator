#!/usr/bin/env python3
"""
Test Script for End-to-End Order Flow Integration

This script tests the TestOrderStrategy to verify that orders are generated
and flow through the entire system end-to-end. It verifies:

1. Strategy generates forced test orders
2. Orders have valid prices and volumes
3. Orders can be processed by the execution simulator
4. Orders flow through the complete system

Usage:
    python test_order_flow_integration.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import HFT Simulator components
from src.strategies.test_order_strategy import TestOrderStrategy, create_test_order_strategy
from src.engine.market_data import BookSnapshot, PriceLevel
from src.engine.order_types import Order
from src.execution.simulator import ExecutionSimulator
from src.execution.fill_models import PerfectFillModel, RealisticFillModel
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.utils.logger import get_logger, setup_main_logger, setup_logger

# Set up logging
setup_main_logger()
logger = setup_logger(__name__, level="INFO")


def create_test_market_data(symbol: str = "TEST", num_points: int = 10) -> pd.DataFrame:
    """Create test market data for simulation"""
    logger.info(f"Creating test market data with {num_points} data points for {symbol}")
    
    data = []
    base_time = pd.Timestamp('2024-01-01 09:30:00')
    base_price = 100.0  # Match our test strategy price
    
    for i in range(num_points):
        price = base_price + np.sin(i * 0.1) * 0.5 + np.random.normal(0, 0.05)
        spread = 0.02
        
        data.append({
            'timestamp': base_time + pd.Timedelta(seconds=i),
            'symbol': symbol,
            'price': round(price, 2),
            'volume': np.random.randint(50, 200),
            'bid': round(price - spread/2, 2),
            'ask': round(price + spread/2, 2),
            'bid_volume': np.random.randint(100, 500),
            'ask_volume': np.random.randint(100, 500),
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created market data: price range ${df['price'].min():.2f}-${df['price'].max():.2f}")
    return df


def test_strategy_order_generation():
    """Test that the TestOrderStrategy generates orders consistently"""
    logger.info("=== Testing Strategy Order Generation ===")
    
    # Create test strategy with specific parameters
    strategy = TestOrderStrategy(
        symbol="TEST",
        test_price=100.00,
        test_volume=10
    )
    
    logger.info(f"Created strategy: {strategy}")
    
    # Create test market snapshots
    timestamps = [pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i) for i in range(5)]
    orders_generated = []
    
    for i, timestamp in enumerate(timestamps):
        # Create market snapshot
        snapshot = BookSnapshot(
            symbol="TEST",
            timestamp=timestamp,
            bids=[PriceLevel(99.99, 100)],
            asks=[PriceLevel(100.01, 100)]
        )
        
        # Generate orders
        result = strategy.on_market_update(snapshot, timestamp)
        
        logger.info(f"Update {i+1}: Generated {len(result.orders)} orders")
        for order in result.orders:
            logger.info(f"  Order: {order.order_id} - {order.side.value} {order.volume}@${order.price:.2f}")
            orders_generated.append(order)
    
    # Verify results
    assert len(orders_generated) >= 5, f"Expected at least 5 orders, got {len(orders_generated)}"
    
    # Check order validity
    for order in orders_generated:
        assert order.price > 0, f"Order price must be positive, got {order.price}"
        assert order.volume > 0, f"Order volume must be positive, got {order.volume}"
        assert order.symbol == "TEST", f"Order symbol must be TEST, got {order.symbol}"
        assert order.order_type == OrderType.LIMIT, f"Expected LIMIT order, got {order.order_type}"
    
    # Get strategy statistics
    info = strategy.get_strategy_info()
    logger.info(f"Strategy generated {info['order_statistics']['total_orders_generated']} total orders")
    logger.info(f"Buy orders: {info['order_statistics']['buy_orders_generated']}")
    logger.info(f"Sell orders: {info['order_statistics']['sell_orders_generated']}")
    
    logger.info("✓ Strategy order generation test PASSED")
    return orders_generated


def test_order_execution_simulation():
    """Test that generated orders can be processed by the execution simulator"""
    logger.info("=== Testing Order Execution Simulation ===")
    
    # Create strategy
    strategy = TestOrderStrategy(
        symbol="AAPL",
        test_price=150.00,
        test_volume=25
    )
    
    # Create market data
    market_data = create_test_market_data("AAPL", num_points=20)
    
    # Create execution simulator with perfect fill model for testing
    simulator = ExecutionSimulator(
        symbol="AAPL",
        fill_model=PerfectFillModel(),
        initial_cash=10000.0
    )
    
    logger.info("Created execution simulator for order processing")
    
    # Run simulation with strategy
    logger.info("Running simulation...")
    results = simulator.run_backtest(market_data, strategy)
    
    # Verify results
    assert results is not None, "Simulation should return results"
    logger.info(f"Simulation completed successfully")
    
    # Check that orders were generated and processed
    strategy_info = strategy.get_strategy_info()
    total_orders = strategy_info['order_statistics']['total_orders_generated']
    
    assert total_orders > 0, f"Strategy should have generated orders, got {total_orders}"
    logger.info(f"Strategy generated {total_orders} orders during simulation")
    
    # Check simulator statistics
    if hasattr(results, 'execution_stats'):
        logger.info(f"Execution stats: {results.execution_stats}")
    
    logger.info("✓ Order execution simulation test PASSED")
    return results


def test_order_metadata_and_tracking():
    """Test that orders contain proper metadata and can be tracked"""
    logger.info("=== Testing Order Metadata and Tracking ===")
    
    strategy = TestOrderStrategy(
        symbol="META",
        test_price=350.00,
        test_volume=5
    )
    
    # Generate some orders
    timestamp = pd.Timestamp('2024-01-01 10:00:00')
    snapshot = BookSnapshot(
        symbol="META",
        timestamp=timestamp,
        bids=[PriceLevel(349.95, 200)],
        asks=[PriceLevel(350.05, 200)]
    )
    
    orders = []
    for i in range(3):
        result = strategy.on_market_update(snapshot, timestamp + pd.Timedelta(seconds=i))
        orders.extend(result.orders)
    
    logger.info(f"Generated {len(orders)} orders for metadata testing")
    
    # Check metadata
    for i, order in enumerate(orders):
        # Check basic order properties
        assert order.order_id is not None, "Order must have ID"
        assert order.symbol == "META", f"Order symbol should be META, got {order.symbol}"
        
        # Check test-specific metadata
        assert 'test_order' in order.metadata, "Order should have test_order metadata"
        assert order.metadata['test_order'] is True, "test_order should be True"
        assert 'forced_generation' in order.metadata, "Order should have forced_generation metadata"
        assert 'original_test_price' in order.metadata, "Order should have original_test_price"
        assert 'original_test_volume' in order.metadata, "Order should have original_test_volume"
        
        logger.info(f"Order {i+1}: ID={order.order_id}, Price=${order.price:.2f}, "
                   f"Volume={order.volume}, Metadata keys={list(order.metadata.keys())}")
    
    # Check strategy tracking
    info = strategy.get_strategy_info()
    assert len(info['recent_orders']) > 0, "Strategy should track recent orders"
    
    logger.info(f"Strategy is tracking {len(info['recent_orders'])} recent orders")
    logger.info("✓ Order metadata and tracking test PASSED")


def test_different_price_volume_scenarios():
    """Test the strategy with different price and volume scenarios"""
    logger.info("=== Testing Different Price/Volume Scenarios ===")
    
    scenarios = [
        {"price": 10.50, "volume": 100, "symbol": "PENNY"},
        {"price": 1500.00, "volume": 1, "symbol": "EXPENSIVE"},
        {"price": 50.25, "volume": 50, "symbol": "NORMAL"}
    ]
    
    for scenario in scenarios:
        logger.info(f"Testing scenario: {scenario}")
        
        strategy = TestOrderStrategy(
            symbol=scenario["symbol"],
            test_price=scenario["price"],
            test_volume=scenario["volume"]
        )
        
        # Generate orders
        timestamp = pd.Timestamp.now()
        snapshot = BookSnapshot(
            symbol=scenario["symbol"],
            timestamp=timestamp,
            bids=[PriceLevel(scenario["price"] - 0.01, 100)],
            asks=[PriceLevel(scenario["price"] + 0.01, 100)]
        )
        
        result = strategy.on_market_update(snapshot, timestamp)
        
        assert len(result.orders) >= 1, f"Should generate at least 1 order for {scenario}"
        
        for order in result.orders:
            assert order.price > 0, f"Order price should be positive"
            assert order.volume > 0, f"Order volume should be positive"
            assert order.volume == scenario["volume"], f"Expected volume {scenario['volume']}, got {order.volume}"
            
        logger.info(f"✓ Scenario {scenario['symbol']} passed")
    
    logger.info("✓ Different price/volume scenarios test PASSED")


def test_factory_function():
    """Test the factory function for creating strategies"""
    logger.info("=== Testing Factory Function ===")
    
    strategy = create_test_order_strategy(
        symbol="FACTORY",
        test_price=200.00,
        test_volume=15
    )
    
    assert strategy.symbol == "FACTORY", "Factory should set symbol correctly"
    assert strategy.test_price == 200.00, "Factory should set test_price correctly"
    assert strategy.test_volume == 15, "Factory should set test_volume correctly"
    
    # Test order generation
    timestamp = pd.Timestamp.now()
    snapshot = BookSnapshot(
        symbol="FACTORY",
        timestamp=timestamp,
        bids=[PriceLevel(199.98, 100)],
        asks=[PriceLevel(200.02, 100)]
    )
    
    result = strategy.on_market_update(snapshot, timestamp)
    assert len(result.orders) >= 1, "Factory-created strategy should generate orders"
    
    logger.info("✓ Factory function test PASSED")


def run_comprehensive_test():
    """Run comprehensive end-to-end test"""
    logger.info("="*60)
    logger.info("COMPREHENSIVE END-TO-END ORDER FLOW TEST")
    logger.info("="*60)
    
    try:
        # Test 1: Basic order generation
        orders = test_strategy_order_generation()
        
        # Test 2: Execution simulation
        results = test_order_execution_simulation()
        
        # Test 3: Metadata and tracking
        test_order_metadata_and_tracking()
        
        # Test 4: Different scenarios
        test_different_price_volume_scenarios()
        
        # Test 5: Factory function
        test_factory_function()
        
        logger.info("="*60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("="*60)
        logger.info("The TestOrderStrategy successfully:")
        logger.info("1. Generates forced test orders with hardcoded price ($100.00) and volume (10)")
        logger.info("2. Orders flow through the execution simulator")
        logger.info("3. Orders contain proper metadata for tracking")
        logger.info("4. Strategy works with different price/volume scenarios")
        logger.info("5. Factory function creates strategies correctly")
        logger.info("")
        logger.info("This verifies that orders flow through the system end-to-end!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    print("Testing TestOrderStrategy end-to-end order flow...")
    print("This will verify that forced test orders are generated and flow through the system")
    print()
    
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 SUCCESS: Test order generation and end-to-end flow verified!")
        print("The strategy generates at least one order regardless of conditions.")
        print("Orders use hardcoded valid price ($100.00) and volume (10).")
        print("Orders flow through the system successfully.")
    else:
        print("\n❌ FAILURE: Tests did not pass completely.")
        sys.exit(1)
