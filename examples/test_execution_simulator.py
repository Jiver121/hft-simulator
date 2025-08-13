#!/usr/bin/env python3
"""
Example script demonstrating ExecutionSimulator usage

This script shows how to:
1. Create synthetic market data
2. Define a simple trading strategy
3. Run a backtest simulation
4. Analyze results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.simulator import ExecutionSimulator, BacktestResult
from src.engine.order_types import Order
from src.engine.market_data import BookSnapshot
from src.utils.constants import OrderSide, OrderType


class SimpleMarketMakingStrategy:
    """
    A simple market making strategy for testing
    
    This strategy:
    - Places buy orders below mid-price
    - Places sell orders above mid-price  
    - Maintains a spread around the current price
    """
    
    def __init__(self, spread_bps: int = 50, order_size: int = 100):
        self.spread_bps = spread_bps  # Spread in basis points
        self.order_size = order_size
        self.order_count = 0
    
    def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> list:
        """
        Called on each market data update
        
        Args:
            snapshot: Current market snapshot
            timestamp: Current time
            
        Returns:
            List of orders to place
        """
        orders = []
        
        # Only trade if we have a valid mid-price
        if not snapshot or not snapshot.mid_price:
            return orders
        
        mid_price = snapshot.mid_price
        spread_amount = mid_price * (self.spread_bps / 10000.0)  # Convert bps to decimal
        
        # Generate orders occasionally (every 10th update to avoid overloading)
        self.order_count += 1
        if self.order_count % 10 == 0:
            # Buy order below mid
            buy_price = mid_price - spread_amount / 2
            buy_order = Order(
                order_id=f"buy_{self.order_count}",
                symbol="TEST",
                side=OrderSide.BID,
                order_type=OrderType.LIMIT,
                price=round(buy_price, 2),
                volume=self.order_size,
                timestamp=timestamp,
                source="strategy"
            )
            orders.append(buy_order)
            
            # Sell order above mid  
            sell_price = mid_price + spread_amount / 2
            sell_order = Order(
                order_id=f"sell_{self.order_count}",
                symbol="TEST", 
                side=OrderSide.ASK,
                order_type=OrderType.LIMIT,
                price=round(sell_price, 2),
                volume=self.order_size,
                timestamp=timestamp,
                source="strategy"
            )
            orders.append(sell_order)
        
        return orders


def create_synthetic_market_data(n_records: int = 1000) -> pd.DataFrame:
    """Create synthetic market data for testing"""
    
    print(f"Creating {n_records:,} synthetic market data records...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps (1 second intervals)
    start_time = pd.Timestamp('2023-01-01 09:30:00')
    timestamps = pd.date_range(start_time, periods=n_records, freq='1S')
    
    # Generate realistic price movements (geometric Brownian motion)
    initial_price = 100.0
    dt = 1.0 / (24 * 60 * 60)  # 1 second in years
    volatility = 0.2  # 20% annual volatility
    drift = 0.05  # 5% annual drift
    
    returns = np.random.normal(
        (drift - 0.5 * volatility**2) * dt,
        volatility * np.sqrt(dt),
        n_records
    )
    
    prices = [initial_price]
    for r in returns[1:]:
        prices.append(prices[-1] * np.exp(r))
    
    # Generate volumes (log-normal distribution)
    volumes = np.random.lognormal(mean=4, sigma=1, size=n_records).astype(int)
    volumes = np.clip(volumes, 10, 1000)  # Clip to reasonable range
    
    # Alternate between bid and ask
    sides = ['bid' if i % 2 == 0 else 'ask' for i in range(n_records)]
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes,
        'side': sides
    })
    
    print(f"Generated data:")
    print(f"  Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
    print(f"  Average volume: {data['volume'].mean():.0f}")
    print(f"  Duration: {data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]}")
    
    return data


def main():
    """Run the ExecutionSimulator example"""
    
    print("=" * 60)
    print("ExecutionSimulator Example")
    print("=" * 60)
    
    # Step 1: Create synthetic market data
    market_data = create_synthetic_market_data(1000)
    
    # Step 2: Create trading strategy
    print("\nCreating simple market making strategy...")
    strategy = SimpleMarketMakingStrategy(spread_bps=50, order_size=10)
    
    # Step 3: Initialize simulator
    print("\nInitializing ExecutionSimulator...")
    simulator = ExecutionSimulator(
        symbol="TEST",
        tick_size=0.01,
        initial_cash=100000.0
    )
    
    # Configure simulation parameters
    simulator.set_config(
        max_position_size=1000,
        max_order_size=100,
        save_snapshots=False  # Disable to save memory
    )
    
    # Step 4: Run backtest
    print("\nRunning backtest simulation...")
    try:
        result = simulator.run_backtest(
            data_source=market_data,
            strategy=strategy
        )
        
        # Step 5: Display results
        print("\n" + "=" * 40)
        print("BACKTEST RESULTS")
        print("=" * 40)
        
        print(f"Symbol: {result.symbol}")
        print(f"Duration: {result.duration}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Total Volume: {result.total_volume:,}")
        print(f"Total P&L: ${result.total_pnl:.2f}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Fill Rate: {result.fill_rate:.1%}")
        print(f"Max Drawdown: ${result.max_drawdown:.2f}")
        
        # Additional analysis
        if result.trades:
            trade_sizes = [trade.volume for trade in result.trades]
            trade_prices = [trade.price for trade in result.trades]
            
            print(f"\nTrade Analysis:")
            print(f"  Average trade size: {np.mean(trade_sizes):.1f}")
            print(f"  Price range: ${min(trade_prices):.2f} - ${max(trade_prices):.2f}")
            print(f"  First trade: {result.trades[0].timestamp}")
            print(f"  Last trade: {result.trades[-1].timestamp}")
        
        # Performance tracking
        performance_df = simulator.get_performance_dataframe()
        if not performance_df.empty:
            print(f"\nPerformance Tracking:")
            print(f"  Final cash: ${performance_df['cash'].iloc[-1]:,.2f}")
            print(f"  Final position: {performance_df['position'].iloc[-1]}")
            print(f"  Final portfolio value: ${performance_df['total_value'].iloc[-1]:,.2f}")
            print(f"  Max unrealized P&L: ${performance_df['unrealized_pnl'].max():,.2f}")
            print(f"  Min unrealized P&L: ${performance_df['unrealized_pnl'].min():,.2f}")
        
        print("\n✓ Backtest completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
