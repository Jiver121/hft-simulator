#!/usr/bin/env python3
"""
Large Dataset Generator for HFT Simulator Scalability Testing

This script generates large-scale market data for testing the performance and
scalability of the HFT simulator. It creates realistic tick data with proper
market microstructure patterns.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LargeDatasetGenerator:
    """Generator for large-scale market data"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.base_price = 50000.0  # Starting price for crypto
        
        # Market microstructure parameters
        self.tick_size = 0.01
        self.volatility = 0.02  # 2% daily volatility
        self.spread_basis_points = 1.5  # 1.5 bps spread
        self.min_spread = 0.01
        
        # Volume parameters
        self.base_volume = 1000
        self.volume_volatility = 0.5
        
        # Order arrival parameters
        self.lambda_trades = 100  # trades per minute on average
        self.lambda_quotes = 500  # quote updates per minute
        
    def generate_price_path(self, n_ticks: int) -> np.ndarray:
        """Generate realistic price path using geometric Brownian motion"""
        logger.info(f"Generating price path for {n_ticks:,} ticks")
        
        # Time increments (assuming 1 second intervals on average)
        dt = 1.0 / (24 * 3600)  # fraction of day
        
        # Generate random walks
        random_increments = np.random.normal(0, np.sqrt(dt) * self.volatility, n_ticks)
        
        # Add some trend and mean reversion
        trend = np.linspace(0, 0.001, n_ticks)  # Slight upward trend
        noise = np.random.normal(0, 0.0001, n_ticks)  # Small noise
        
        # Combine effects
        price_changes = random_increments + trend + noise
        
        # Create cumulative price path
        log_prices = np.log(self.base_price) + np.cumsum(price_changes)
        prices = np.exp(log_prices)
        
        # Ensure prices stay reasonable
        prices = np.clip(prices, self.base_price * 0.5, self.base_price * 2.0)
        
        return prices
    
    def generate_spreads(self, prices: np.ndarray) -> tuple:
        """Generate bid-ask spreads based on price levels"""
        logger.info("Generating bid-ask spreads")
        
        # Base spread as percentage of price
        spread_pct = self.spread_basis_points / 10000.0
        spreads = prices * spread_pct
        
        # Add some randomness to spreads
        spread_noise = np.random.uniform(0.8, 1.2, len(spreads))
        spreads = spreads * spread_noise
        
        # Ensure minimum spread
        spreads = np.maximum(spreads, self.min_spread)
        
        # Calculate bid/ask
        half_spreads = spreads / 2
        bids = prices - half_spreads
        asks = prices + half_spreads
        
        # Round to tick size
        bids = np.round(bids / self.tick_size) * self.tick_size
        asks = np.round(asks / self.tick_size) * self.tick_size
        
        return bids, asks, spreads
    
    def generate_volumes(self, n_ticks: int) -> tuple:
        """Generate realistic volume patterns"""
        logger.info("Generating volume data")
        
        # Base volume with lognormal distribution
        volumes = np.random.lognormal(
            mean=np.log(self.base_volume),
            sigma=self.volume_volatility,
            size=n_ticks
        ).astype(int)
        
        # Ensure minimum volume
        volumes = np.maximum(volumes, 1)
        
        # Generate bid/ask volumes
        bid_volumes = np.random.lognormal(
            mean=np.log(self.base_volume * 0.8),
            sigma=self.volume_volatility,
            size=n_ticks
        ).astype(int)
        
        ask_volumes = np.random.lognormal(
            mean=np.log(self.base_volume * 0.8),
            sigma=self.volume_volatility,
            size=n_ticks
        ).astype(int)
        
        bid_volumes = np.maximum(bid_volumes, 1)
        ask_volumes = np.maximum(ask_volumes, 1)
        
        return volumes, bid_volumes, ask_volumes
    
    def generate_timestamps(self, n_ticks: int, start_time: datetime = None) -> pd.DatetimeIndex:
        """Generate realistic timestamps with variable intervals"""
        logger.info(f"Generating timestamps for {n_ticks:,} ticks")
        
        if start_time is None:
            start_time = datetime(2024, 1, 1, 9, 30, 0)
        
        # Generate inter-arrival times (exponential distribution)
        # Average 1 second between ticks, but with realistic variation
        mean_interval = 1.0  # seconds
        intervals = np.random.exponential(mean_interval, n_ticks)
        
        # Convert to timedeltas and create timestamps
        cumulative_seconds = np.cumsum(intervals)
        timestamps = [
            start_time + timedelta(seconds=float(seconds))
            for seconds in cumulative_seconds
        ]
        
        return pd.DatetimeIndex(timestamps)
    
    def generate_market_data_types(self, n_ticks: int) -> list:
        """Generate mix of market data types (trades, quotes, etc.)"""
        # 70% trades, 30% quote updates
        types = np.random.choice(
            ['trade', 'quote'],
            size=n_ticks,
            p=[0.7, 0.3]
        )
        return types.tolist()
    
    def generate_large_dataset(self, n_rows: int, output_file: str) -> str:
        """Generate large dataset and save to CSV"""
        logger.info(f"Starting generation of {n_rows:,} row dataset")
        logger.info(f"Output file: {output_file}")
        
        # Generate price path
        prices = self.generate_price_path(n_rows)
        
        # Generate spreads and bid/ask
        bids, asks, spreads = self.generate_spreads(prices)
        
        # Generate volumes
        volumes, bid_volumes, ask_volumes = self.generate_volumes(n_rows)
        
        # Generate timestamps
        timestamps = self.generate_timestamps(n_rows)
        
        # Generate data types
        data_types = self.generate_market_data_types(n_rows)
        
        # Create DataFrame in chunks to manage memory
        chunk_size = 100000  # Process 100k rows at a time
        chunks = []
        
        logger.info("Creating DataFrame in chunks...")
        for i in range(0, n_rows, chunk_size):
            end_idx = min(i + chunk_size, n_rows)
            chunk_data = {
                'timestamp': timestamps[i:end_idx],
                'symbol': [self.symbol] * (end_idx - i),
                'price': prices[i:end_idx],
                'volume': volumes[i:end_idx],
                'bid': bids[i:end_idx],
                'ask': asks[i:end_idx],
                'bid_volume': bid_volumes[i:end_idx],
                'ask_volume': ask_volumes[i:end_idx],
                'trade_type': data_types[i:end_idx],
                'spread': spreads[i:end_idx]
            }
            
            chunk_df = pd.DataFrame(chunk_data)
            chunks.append(chunk_df)
            
            # Log progress
            if i % (chunk_size * 10) == 0:  # Every 1M rows
                progress = (end_idx / n_rows) * 100
                logger.info(f"Progress: {progress:.1f}% ({end_idx:,}/{n_rows:,} rows)")
        
        # Combine all chunks
        logger.info("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        logger.info(f"Saving to {output_file}...")
        df.to_csv(output_file, index=False)
        
        # Calculate file size
        file_size = output_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Dataset generation complete!")
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Columns: {len(df.columns)}")
        logger.info(f"File size: {file_size_mb:.1f} MB")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Display sample statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        logger.info(f"Average spread: {df['spread'].mean():.4f}")
        logger.info(f"Volume range: {df['volume'].min():,} - {df['volume'].max():,}")
        logger.info(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return output_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate large-scale market data for HFT simulator testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--rows',
        type=int,
        required=True,
        help='Number of rows to generate'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol for the dataset'
    )
    
    parser.add_argument(
        '--base-price',
        type=float,
        default=50000.0,
        help='Starting price for the asset'
    )
    
    parser.add_argument(
        '--volatility',
        type=float,
        default=0.02,
        help='Daily volatility (as decimal, e.g., 0.02 = 2%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    try:
        # Create generator
        generator = LargeDatasetGenerator(symbol=args.symbol)
        generator.base_price = args.base_price
        generator.volatility = args.volatility
        
        # Generate dataset
        output_file = generator.generate_large_dataset(args.rows, args.output)
        
        print(f"\nSuccess! Large dataset generated: {output_file}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
