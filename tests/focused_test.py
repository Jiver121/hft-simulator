#!/usr/bin/env python3
"""
Focused Integration Test - Testing Core System Functions

This test focuses on the most basic functionality to identify issues.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.simulator import ExecutionSimulator, BacktestResult
from src.execution.fill_models import PerfectFillModel
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.utils.constants import OrderSide, OrderType


class MinimalTestStrategy(BaseStrategy):
    """Absolutely minimal strategy for testing"""
    
    def __init__(self, symbol: str):
        super().__init__("minimal_test", symbol)
        self.order_count = 0
        
    def on_market_update(self, snapshot, timestamp):
        result = StrategyResult(timestamp=timestamp)
        
        # Only place one buy market order on first tick
        if self.order_count == 0 and snapshot and snapshot.best_ask:
            order = self.create_order(
                side=OrderSide.BUY,
                volume=100,
                price=snapshot.best_ask,  # Use exact ask price
                order_type=OrderType.MARKET,
                reason="Single test order"
            )
            result.add_order(order, "Test")
            self.order_count += 1
            print(f"Strategy created order: {order}")
            
        return result


def test_basic_execution():
    """Test the absolute basics"""
    print("Testing basic execution...")
    
    # Create minimal test data
    test_data = []
    for i in range(5):  # Only 5 ticks
        test_data.append({
            'timestamp': pd.Timestamp('2024-01-01 09:30:00') + pd.Timedelta(seconds=i),
            'symbol': 'AAPL',
            'price': 100.0,
            'volume': 1000,
            'bid': 99.95,
            'ask': 100.05,
            'bid_volume': 500,
            'ask_volume': 500
        })
    
    df = pd.DataFrame(test_data)
    print(f"Created test data with {len(df)} rows")
    print(f"Sample data:\n{df.head()}")
    
    # Create simulator
    try:
        simulator = ExecutionSimulator(
            symbol="AAPL",
            fill_model=PerfectFillModel(),
            initial_cash=100000.0
        )
        print("✓ ExecutionSimulator created successfully")
    except Exception as e:
        print(f"✗ Failed to create ExecutionSimulator: {e}")
        return False
    
    # Create strategy
    try:
        strategy = MinimalTestStrategy("AAPL")
        print("✓ Strategy created successfully")
    except Exception as e:
        print(f"✗ Failed to create strategy: {e}")
        return False
    
    # Run backtest
    try:
        result = simulator.run_backtest(
            data_source=df,
            strategy=strategy
        )
        print("✓ Backtest completed successfully")
        print(f"Orders generated: {len(result.orders)}")
        print(f"Trades executed: {len(result.trades)}")
        
        if len(result.orders) > 0:
            print(f"First order: {result.orders[0]}")
        if len(result.trades) > 0:
            print(f"First trade: {result.trades[0]}")
            
        return len(result.orders) > 0  # Success if at least one order was generated
        
    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_execution()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
