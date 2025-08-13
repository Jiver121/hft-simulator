#!/usr/bin/env python3
"""
Test Script for Strategy-Simulator Integration

This script tests the fixes to the strategy-simulator integration issues.
It verifies that strategies properly inherit from BaseStrategy and return
StrategyResult objects as expected.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.execution.simulator import ExecutionSimulator
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.engine.market_data import BookSnapshot
from src.engine.order_types import Order, PriceLevel
from src.execution.fill_models import RealisticFillModel
from src.utils.constants import OrderSide, OrderType
from src.utils.logger import get_logger

# Import the fixed strategies from main.py
from main import SimpleMarketMakingStrategy, SimpleMomentumStrategy


def create_test_snapshot(symbol: str, timestamp: pd.Timestamp, 
                        mid_price: float = 100.0, spread: float = 0.02) -> BookSnapshot:
    """Create a test BookSnapshot for strategy testing"""
    
    bid_price = mid_price - spread/2
    ask_price = mid_price + spread/2
    
    bid_level = PriceLevel(price=bid_price, total_volume=500)
    ask_level = PriceLevel(price=ask_price, total_volume=500)
    
    snapshot = BookSnapshot(
        symbol=symbol,
        timestamp=timestamp,
        bids=[bid_level],
        asks=[ask_level],
        last_trade_price=mid_price,
        last_trade_volume=100,
        last_trade_timestamp=timestamp
    )
    
    return snapshot


def test_strategy_inheritance():
    """Test that strategies properly inherit from BaseStrategy"""
    print("\n=== Testing Strategy Inheritance ===")
    
    # Test market making strategy
    mm_strategy = SimpleMarketMakingStrategy("AAPL", spread_bps=20.0, order_size=100)
    assert isinstance(mm_strategy, BaseStrategy), "SimpleMarketMakingStrategy should inherit from BaseStrategy"
    assert mm_strategy.symbol == "AAPL", "Strategy symbol should be set correctly"
    assert mm_strategy.spread_bps == 20.0, "Strategy parameters should be set correctly"
    print("✓ SimpleMarketMakingStrategy inheritance test passed")
    
    # Test momentum strategy
    momentum_strategy = SimpleMomentumStrategy("MSFT", momentum_threshold=0.002, order_size=200)
    assert isinstance(momentum_strategy, BaseStrategy), "SimpleMomentumStrategy should inherit from BaseStrategy"
    assert momentum_strategy.symbol == "MSFT", "Strategy symbol should be set correctly"
    assert momentum_strategy.momentum_threshold == 0.002, "Strategy parameters should be set correctly"
    print("✓ SimpleMomentumStrategy inheritance test passed")


def test_strategy_method_signatures():
    """Test that strategies have correct method signatures"""
    print("\n=== Testing Method Signatures ===")
    
    symbol = "AAPL"
    strategy = SimpleMarketMakingStrategy(symbol)
    
    # Check that on_market_update exists and has correct signature
    assert hasattr(strategy, 'on_market_update'), "Strategy should have on_market_update method"
    
    # Create test snapshot
    timestamp = pd.Timestamp.now()
    snapshot = create_test_snapshot(symbol, timestamp)
    
    # Test method call
    try:
        result = strategy.on_market_update(snapshot, timestamp)
        assert result is not None, "Strategy should return a result"
        print("✓ Method signature test passed")
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        raise


def test_strategy_return_type():
    """Test that strategies return StrategyResult objects"""
    print("\n=== Testing Strategy Return Types ===")
    
    # Test market making strategy
    mm_strategy = SimpleMarketMakingStrategy("AAPL", spread_bps=10.0, order_size=100)
    timestamp = pd.Timestamp.now()
    snapshot = create_test_snapshot("AAPL", timestamp, mid_price=150.0)
    
    result = mm_strategy.on_market_update(snapshot, timestamp)
    
    assert isinstance(result, StrategyResult), f"Expected StrategyResult, got {type(result)}"
    assert hasattr(result, 'orders'), "StrategyResult should have orders attribute"
    assert hasattr(result, 'decision_reason'), "StrategyResult should have decision_reason attribute"
    assert isinstance(result.orders, list), "orders should be a list"
    print("✓ Market making strategy return type test passed")
    
    # Test momentum strategy
    momentum_strategy = SimpleMomentumStrategy("AAPL", momentum_threshold=0.001, order_size=200)
    
    # Need to build up price history first
    for i in range(10):
        test_price = 150.0 + i * 0.1
        test_snapshot = create_test_snapshot("AAPL", timestamp + pd.Timedelta(seconds=i), mid_price=test_price)
        result = momentum_strategy.on_market_update(test_snapshot, timestamp + pd.Timedelta(seconds=i))
        
    assert isinstance(result, StrategyResult), f"Expected StrategyResult, got {type(result)}"
    assert hasattr(result, 'orders'), "StrategyResult should have orders attribute"
    print("✓ Momentum strategy return type test passed")


def test_error_handling():
    """Test error handling for missing/invalid market data"""
    print("\n=== Testing Error Handling ===")
    
    strategy = SimpleMarketMakingStrategy("AAPL")
    timestamp = pd.Timestamp.now()
    
    # Test with None snapshot
    result = strategy.on_market_update(None, timestamp)
    assert isinstance(result, StrategyResult), "Should return StrategyResult even with None snapshot"
    assert len(result.orders) == 0, "Should not generate orders with None snapshot"
    assert "Invalid market data" in result.decision_reason, "Should indicate invalid market data"
    print("✓ None snapshot handling test passed")
    
    # Test with invalid snapshot (no prices)
    empty_snapshot = BookSnapshot(
        symbol="AAPL",
        timestamp=timestamp,
        bids=[],
        asks=[]
    )
    
    result = strategy.on_market_update(empty_snapshot, timestamp)
    assert isinstance(result, StrategyResult), "Should return StrategyResult even with empty snapshot"
    assert len(result.orders) == 0, "Should not generate orders with empty snapshot"
    print("✓ Empty snapshot handling test passed")


def test_risk_management():
    """Test risk management and order validation"""
    print("\n=== Testing Risk Management ===")
    
    strategy = SimpleMarketMakingStrategy("AAPL", order_size=50)  # Smaller order size
    timestamp = pd.Timestamp.now()
    snapshot = create_test_snapshot("AAPL", timestamp, mid_price=150.0)
    
    # Test normal operation
    result = strategy.on_market_update(snapshot, timestamp)
    assert isinstance(result, StrategyResult), "Should return StrategyResult"
    
    # Check that orders are created with proper validation
    if result.orders:
        for order in result.orders:
            assert isinstance(order, Order), "Should return Order objects"
            assert order.volume > 0, "Order volume should be positive"
            assert order.price and order.price > 0, "Order price should be positive"
    
    print("✓ Risk management test passed")


def test_simulator_integration():
    """Test integration with ExecutionSimulator"""
    print("\n=== Testing Simulator Integration ===")
    
    # Create test data
    data = []
    base_time = pd.Timestamp('2024-01-01 09:30:00')
    base_price = 150.0
    
    for i in range(100):  # 100 data points
        price = base_price + np.sin(i * 0.1) * 2 + np.random.normal(0, 0.1)
        spread = 0.02
        
        data.append({
            'timestamp': base_time + pd.Timedelta(seconds=i),
            'symbol': 'AAPL',
            'price': round(price, 2),
            'volume': np.random.randint(100, 1000),
            'bid': round(price - spread/2, 2),
            'ask': round(price + spread/2, 2),
            'bid_volume': np.random.randint(100, 500),
            'ask_volume': np.random.randint(100, 500),
        })
    
    df = pd.DataFrame(data)
    
    # Test with market making strategy
    try:
        simulator = ExecutionSimulator(
            symbol="AAPL",
            fill_model=RealisticFillModel(),
            initial_cash=100000.0
        )
        
        strategy = SimpleMarketMakingStrategy("AAPL", spread_bps=15.0, order_size=50)
        
        result = simulator.run_backtest(
            data_source=df,
            strategy=strategy
        )
        
        assert result is not None, "Simulator should return a result"
        assert result.symbol == "AAPL", "Result should have correct symbol"
        print(f"✓ Simulator integration test passed - P&L: ${result.total_pnl:.2f}, Trades: {result.total_trades}")
        
    except Exception as e:
        print(f"✗ Simulator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all integration tests"""
    print("Strategy-Simulator Integration Tests")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        test_strategy_inheritance()
        test_strategy_method_signatures()
        test_strategy_return_type()
        test_error_handling()
        test_risk_management()
        test_simulator_integration()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED! Strategy-Simulator integration is working correctly.")
        print("The fixes have successfully resolved the integration issues:")
        print("- ✓ Strategies properly inherit from BaseStrategy")
        print("- ✓ Method signatures match parent class expectations")
        print("- ✓ Strategies return StrategyResult objects")
        print("- ✓ Proper error handling for missing/invalid market data")
        print("- ✓ Risk management and order validation working")
        print("- ✓ Integration with ExecutionSimulator successful")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
