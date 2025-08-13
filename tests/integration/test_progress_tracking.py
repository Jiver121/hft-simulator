#!/usr/bin/env python3
"""
Test script to demonstrate the real-time progress tracking functionality
in the ExecutionSimulator.

This script creates sample data and a simple strategy to test the progress
tracking features that were just implemented.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.execution.simulator import ExecutionSimulator
from src.engine.order_types import Order
from src.utils.constants import OrderSide, OrderType


class SimpleTestStrategy:
    """
    A simple test strategy that generates orders occasionally
    to demonstrate the progress tracking functionality.
    """
    
    def __init__(self):
        self.tick_count = 0
        self.order_count = 0
    
    def on_market_update(self, snapshot, current_time):
        """
        Strategy logic: Generate orders occasionally to test progress tracking
        """
        self.tick_count += 1
        
        # Generate an order every 50 ticks to see progress tracking
        if self.tick_count % 50 == 0:
            self.order_count += 1
            
            # Alternate between buy and sell orders
            if self.order_count % 2 == 0:
                # Create a buy order slightly below mid price
                price = snapshot.mid_price - 0.01
                side = OrderSide.BUY
            else:
                # Create a sell order slightly above mid price
                price = snapshot.mid_price + 0.01
                side = OrderSide.SELL
            
            order = Order(
                symbol=snapshot.symbol,
                side=side,
                order_type=OrderType.LIMIT,
                volume=10,
                price=price,
                timestamp=current_time
            )
            
            return [order]
        
        return []


def create_sample_data(num_ticks=1000):
    """
    Create sample market data for testing
    
    Args:
        num_ticks: Number of data points to generate
        
    Returns:
        DataFrame with sample market data
    """
    print(f"Creating sample market data with {num_ticks:,} ticks...")
    
    # Generate timestamps
    start_time = datetime(2023, 1, 1, 9, 30, 0)
    timestamps = [start_time + timedelta(milliseconds=i*100) for i in range(num_ticks)]
    
    # Generate realistic price data with small random walk
    np.random.seed(42)  # For reproducible results
    initial_price = 100.0
    price_changes = np.random.normal(0, 0.01, num_ticks)  # Small random changes
    prices = initial_price + np.cumsum(price_changes)
    
    # Generate volumes
    volumes = np.random.randint(10, 200, num_ticks)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })
    
    print(f"Sample data created: {len(data):,} records from {data['timestamp'].min()} to {data['timestamp'].max()}")
    return data


def test_progress_tracking():
    """Test the progress tracking functionality with different configurations"""
    
    print("=" * 80)
    print("Testing ExecutionSimulator Real-Time Progress Tracking")
    print("=" * 80)
    
    # Create sample data
    sample_data = create_sample_data(num_ticks=1000)
    
    # Initialize simulator
    print("\nInitializing ExecutionSimulator...")
    simulator = ExecutionSimulator(symbol="TEST")
    
    # Test 1: Standard progress tracking (default settings)
    print("\n" + "="*50)
    print("Test 1: Standard Progress Tracking")
    print("="*50)
    
    simulator.set_config(
        show_progress_bar=True,
        verbose_progress=False,
        progress_update_frequency=100,  # Update every 100 ticks
        progress_time_frequency=1.0     # Update every second
    )
    
    strategy = SimpleTestStrategy()
    result = simulator.run_backtest(sample_data, strategy)
    
    print(f"\nResult Summary:")
    print(f"- Total Orders Generated: {len(result.orders)}")
    print(f"- Total Trades Executed: {len(result.trades)}")
    print(f"- Total P&L: ${result.total_pnl:.2f}")
    print(f"- Fill Rate: {result.fill_rate:.1%}")
    
    # Test 2: Verbose progress tracking
    print("\n" + "="*50)
    print("Test 2: Verbose Progress Tracking")
    print("="*50)
    
    simulator.set_config(
        show_progress_bar=True,
        verbose_progress=True,          # Show detailed information
        progress_update_frequency=50,   # More frequent updates
        progress_time_frequency=0.5     # Update every 0.5 seconds
    )
    
    strategy = SimpleTestStrategy()
    result = simulator.run_backtest(sample_data, strategy)
    
    print(f"\nResult Summary:")
    print(f"- Total Orders Generated: {len(result.orders)}")
    print(f"- Total Trades Executed: {len(result.trades)}")
    print(f"- Total P&L: ${result.total_pnl:.2f}")
    print(f"- Fill Rate: {result.fill_rate:.1%}")
    
    # Test 3: No progress display (for comparison)
    print("\n" + "="*50)
    print("Test 3: No Progress Display")
    print("="*50)
    
    simulator.set_config(
        show_progress_bar=False  # Disable progress display
    )
    
    strategy = SimpleTestStrategy()
    result = simulator.run_backtest(sample_data, strategy)
    
    print(f"\nResult Summary:")
    print(f"- Total Orders Generated: {len(result.orders)}")
    print(f"- Total Trades Executed: {len(result.trades)}")
    print(f"- Total P&L: ${result.total_pnl:.2f}")
    print(f"- Fill Rate: {result.fill_rate:.1%}")
    
    # Test 4: High frequency updates (for fast processing)
    print("\n" + "="*50)
    print("Test 4: High Frequency Updates")
    print("="*50)
    
    simulator.set_config(
        show_progress_bar=True,
        verbose_progress=True,
        progress_update_frequency=25,   # Very frequent tick updates
        progress_time_frequency=0.1     # Very frequent time updates
    )
    
    # Use smaller dataset for this test
    small_data = create_sample_data(num_ticks=200)
    strategy = SimpleTestStrategy()
    result = simulator.run_backtest(small_data, strategy)
    
    print(f"\nResult Summary:")
    print(f"- Total Orders Generated: {len(result.orders)}")
    print(f"- Total Trades Executed: {len(result.trades)}")
    print(f"- Total P&L: ${result.total_pnl:.2f}")
    print(f"- Fill Rate: {result.fill_rate:.1%}")
    
    print("\n" + "="*80)
    print("Progress Tracking Tests Completed Successfully!")
    print("="*80)
    print()
    print("Key Features Demonstrated:")
    print("✓ Real-time progress bar with percentage completion")
    print("✓ Orders Generated and Trades Executed counters")
    print("✓ Configurable update frequency (by ticks and time)")
    print("✓ Verbose and compact display modes")
    print("✓ ETA calculation based on processing rate")
    print("✓ Final summary with total statistics")
    print("✓ Works in both verbose and non-verbose modes")


if __name__ == "__main__":
    try:
        test_progress_tracking()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
