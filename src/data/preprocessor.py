"""
Data Preprocessing Module for HFT Simulator

This module handles advanced preprocessing of HFT data including feature engineering,
data transformation, and preparation for order book reconstruction.

Educational Notes:
- Preprocessing is crucial for converting raw tick data into usable format
- We create derived features like mid-price, spread, and order flow imbalance
- Time-based features help capture market dynamics and patterns
- Proper preprocessing can significantly improve strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import warnings

from src.utils.logger import get_logger, log_performance, log_memory_usage
from src.utils.helpers import (
    rolling_statistics, exponential_moving_average, 
    calculate_returns, safe_divide, Timer
)
from src.utils.constants import (
    OrderSide, OrderType, MICROSECONDS_PER_SECOND,
    round_to_tick_size, get_side_multiplier
)


class DataPreprocessor:
    """
    Advanced data preprocessor for HFT datasets
    
    This class transforms raw HFT data into a format suitable for:
    - Order book reconstruction
    - Strategy development
    - Performance analysis
    - Visualization
    
    Key transformations include:
    - Time-based aggregation and resampling
    - Feature engineering (technical indicators, market microstructure features)
    - Data normalization and scaling
    - Order flow analysis
    
    Example Usage:
        >>> preprocessor = DataPreprocessor()
        >>> processed_data = preprocessor.process_tick_data(raw_data)
        >>> features = preprocessor.create_features(processed_data)
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialize the data preprocessor
        
        Args:
            tick_size: Minimum price increment for the instrument
        """
        self.logger = get_logger(__name__)
        self.tick_size = tick_size
        
        # Feature engineering parameters
        self.lookback_windows = [5, 10, 20, 50, 100]  # For rolling statistics
        self.ema_alphas = [0.1, 0.3, 0.5]  # For exponential moving averages
        
        self.logger.info(f"DataPreprocessor initialized with tick_size={tick_size}")
    
    @log_performance
    @log_memory_usage
    def process_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw tick data into standardized format
        
        Args:
            df: Raw tick data DataFrame
            
        Returns:
            Processed DataFrame with additional computed fields
            
        Educational Notes:
            - We compute mid-price as the average of best bid and ask
            - Spread is the difference between best ask and bid
            - Order flow imbalance indicates buying vs selling pressure
        """
        if len(df) == 0:
            self.logger.warning("Empty DataFrame provided for processing")
            return df
        
        df = df.copy()
        self.logger.info(f"Processing {len(df):,} tick records")
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Basic price and volume processing
        df = self._process_prices(df)
        df = self._process_volumes(df)
        
        # Time-based features
        if 'timestamp' in df.columns:
            df = self._create_time_features(df)
        
        # Market microstructure features
        df = self._create_microstructure_features(df)
        
        # Order flow features
        if 'side' in df.columns:
            df = self._create_order_flow_features(df)
        
        self.logger.info(f"Processing completed. Output shape: {df.shape}")
        return df
    
    def _process_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize price data"""
        df = df.copy()
        
        if 'price' not in df.columns:
            return df
        
        # Round prices to tick size
        df['price'] = df['price'].apply(lambda x: round_to_tick_size(x, self.tick_size))
        
        # Calculate price changes
        df['price_change'] = df['price'].diff()
        df['price_change_pct'] = df['price'].pct_change()
        
        # Price levels (useful for order book analysis)
        if len(df) > 0:
            min_price = df['price'].min()
            df['price_level'] = ((df['price'] - min_price) / self.tick_size).astype(int)
        
        return df
    
    def _process_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process volume data and create volume-based features"""
        df = df.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # Cumulative volume
        df['cumulative_volume'] = df['volume'].cumsum()
        
        # Volume-weighted features
        if 'price' in df.columns:
            df['vwap_numerator'] = df['price'] * df['volume']
            df['vwap'] = df['vwap_numerator'].cumsum() / df['cumulative_volume']
        
        # Volume changes
        df['volume_change'] = df['volume'].diff()
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from timestamps"""
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['microsecond'] = df['timestamp'].dt.microsecond
        
        # Market session features (assuming US market hours)
        df['market_open'] = ((df['hour'] == 9) & (df['minute'] >= 30)).astype(bool)
        df['market_close'] = ((df['hour'] == 16) & (df['minute'] == 0)).astype(bool)
        df['lunch_time'] = (df['hour'] == 12).astype(bool)
        
        # Time since market open (in minutes)
        market_open_time = df['timestamp'].dt.normalize() + pd.Timedelta(hours=9, minutes=30)
        df['minutes_since_open'] = (df['timestamp'] - market_open_time).dt.total_seconds() / 60
        
        # Inter-arrival times (time between consecutive events)
        df['inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds() * 1000  # milliseconds
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        df = df.copy()
        
        # If we have bid/ask data, compute spread and mid-price
        if 'side' in df.columns and 'price' in df.columns:
            df = self._compute_bid_ask_features(df)
        
        # Price momentum features
        if 'price' in df.columns:
            df = self._compute_momentum_features(df)
        
        # Volatility features
        if 'price_change_pct' in df.columns:
            df = self._compute_volatility_features(df)
        
        return df
    
    def _compute_bid_ask_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute bid-ask spread and related features"""
        df = df.copy()
        
        # Separate bids and asks
        bids = df[df['side'].isin(['bid', 'buy'])].copy()
        asks = df[df['side'].isin(['ask', 'sell'])].copy()
        
        if len(bids) == 0 or len(asks) == 0:
            self.logger.warning("Missing bid or ask data for spread calculation")
            return df
        
        # For each timestamp, find best bid and ask
        if 'timestamp' in df.columns:
            # Group by timestamp and compute best bid/ask
            bid_groups = bids.groupby('timestamp')['price'].max().rename('best_bid')
            ask_groups = asks.groupby('timestamp')['price'].min().rename('best_ask')
            
            # Merge back to main dataframe
            df = df.merge(bid_groups, left_on='timestamp', right_index=True, how='left')
            df = df.merge(ask_groups, left_on='timestamp', right_index=True, how='left')
            
            # Forward fill missing values
            df['best_bid'] = df['best_bid'].ffill()
            df['best_ask'] = df['best_ask'].ffill()
            
            # Compute derived features
            df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
            df['spread'] = df['best_ask'] - df['best_bid']
            df['spread_pct'] = safe_divide(df['spread'], df['mid_price']) * 100
            
            # Spread in ticks
            df['spread_ticks'] = df['spread'] / self.tick_size
            
            # Price relative to mid
            df['price_vs_mid'] = df['price'] - df['mid_price']
            df['price_vs_mid_pct'] = safe_divide(df['price_vs_mid'], df['mid_price']) * 100
        
        return df
    
    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price momentum and trend features"""
        df = df.copy()
        
        if 'price' not in df.columns:
            return df
        
        # Simple moving averages
        for window in self.lookback_windows:
            if len(df) >= window:
                df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
                df[f'price_vs_sma_{window}'] = df['price'] - df[f'sma_{window}']
        
        # Exponential moving averages
        for alpha in self.ema_alphas:
            alpha_str = str(alpha).replace('.', '')
            df[f'ema_{alpha_str}'] = exponential_moving_average(df['price'], alpha)
            df[f'price_vs_ema_{alpha_str}'] = df['price'] - df[f'ema_{alpha_str}']
        
        # Price momentum (rate of change)
        for window in [5, 10, 20]:
            if len(df) >= window:
                df[f'momentum_{window}'] = df['price'].pct_change(periods=window)
        
        # Trend direction
        df['trend_5'] = np.where(df['price'] > df['price'].shift(5), 1, 
                                np.where(df['price'] < df['price'].shift(5), -1, 0))
        
        return df
    
    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility and risk features"""
        df = df.copy()
        
        if 'price_change_pct' not in df.columns:
            return df
        
        # Rolling volatility (standard deviation of returns)
        for window in self.lookback_windows:
            if len(df) >= window:
                df[f'volatility_{window}'] = df['price_change_pct'].rolling(window=window).std()
        
        # Realized volatility (sum of squared returns)
        for window in [10, 20, 50]:
            if len(df) >= window:
                df[f'realized_vol_{window}'] = np.sqrt(
                    (df['price_change_pct'] ** 2).rolling(window=window).sum()
                )
        
        # High-low volatility proxy
        if len(df) >= 20:
            df['high_20'] = df['price'].rolling(window=20).max()
            df['low_20'] = df['price'].rolling(window=20).min()
            df['hl_volatility'] = safe_divide(df['high_20'] - df['low_20'], df['price'])
        
        return df
    
    def _create_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create order flow and liquidity features"""
        df = df.copy()
        
        if 'side' not in df.columns or 'volume' not in df.columns:
            return df
        
        # Order flow imbalance
        df['side_numeric'] = df['side'].map({
            'bid': 1, 'buy': 1, 'ask': -1, 'sell': -1
        }).fillna(0)
        
        df['signed_volume'] = df['volume'] * df['side_numeric']
        
        # Rolling order flow imbalance
        for window in [10, 20, 50]:
            if len(df) >= window:
                df[f'order_flow_{window}'] = df['signed_volume'].rolling(window=window).sum()
                total_volume = df['volume'].rolling(window=window).sum()
                df[f'order_flow_ratio_{window}'] = safe_divide(df[f'order_flow_{window}'], total_volume)
        
        # Buy/sell pressure
        df['buy_volume'] = np.where(df['side_numeric'] > 0, df['volume'], 0)
        df['sell_volume'] = np.where(df['side_numeric'] < 0, df['volume'], 0)
        
        for window in [10, 20]:
            if len(df) >= window:
                buy_vol = df['buy_volume'].rolling(window=window).sum()
                sell_vol = df['sell_volume'].rolling(window=window).sum()
                df[f'buy_sell_ratio_{window}'] = safe_divide(buy_vol, sell_vol)
        
        return df
    
    @log_performance
    def create_order_book_snapshots(self, df: pd.DataFrame, 
                                   depth: int = 10) -> pd.DataFrame:
        """
        Create order book snapshots from tick data
        
        Args:
            df: Tick data with order book updates
            depth: Number of price levels to include on each side
            
        Returns:
            DataFrame with order book snapshots
            
        Educational Notes:
            - Order book snapshots show the state of the book at specific times
            - Each snapshot includes best bids/asks up to specified depth
            - This is essential for market making and liquidity analysis
        """
        if len(df) == 0 or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        self.logger.info(f"Creating order book snapshots with depth={depth}")
        
        snapshots = []
        
        # Group by timestamp to create snapshots
        for timestamp, group in df.groupby('timestamp'):
            snapshot = self._create_single_snapshot(group, timestamp, depth)
            if snapshot:
                snapshots.append(snapshot)
        
        if not snapshots:
            return pd.DataFrame()
        
        result = pd.DataFrame(snapshots)
        self.logger.info(f"Created {len(result)} order book snapshots")
        
        return result
    
    def _create_single_snapshot(self, group: pd.DataFrame, 
                               timestamp: pd.Timestamp, 
                               depth: int) -> Optional[Dict]:
        """Create a single order book snapshot"""
        
        if 'side' not in group.columns or 'price' not in group.columns:
            return None
        
        # Separate bids and asks
        bids = group[group['side'].isin(['bid', 'buy'])].copy()
        asks = group[group['side'].isin(['ask', 'sell'])].copy()
        
        # Sort bids (highest first) and asks (lowest first)
        bids = bids.sort_values('price', ascending=False).head(depth)
        asks = asks.sort_values('price', ascending=True).head(depth)
        
        snapshot = {
            'timestamp': timestamp,
            'mid_price': None,
            'spread': None,
            'total_bid_volume': 0,
            'total_ask_volume': 0,
        }
        
        # Add bid levels
        for i, (_, bid) in enumerate(bids.iterrows()):
            snapshot[f'bid_price_{i+1}'] = bid['price']
            snapshot[f'bid_volume_{i+1}'] = bid.get('volume', 0)
            snapshot['total_bid_volume'] += bid.get('volume', 0)
        
        # Add ask levels
        for i, (_, ask) in enumerate(asks.iterrows()):
            snapshot[f'ask_price_{i+1}'] = ask['price']
            snapshot[f'ask_volume_{i+1}'] = ask.get('volume', 0)
            snapshot['total_ask_volume'] += ask.get('volume', 0)
        
        # Calculate mid-price and spread
        if len(bids) > 0 and len(asks) > 0:
            best_bid = bids.iloc[0]['price']
            best_ask = asks.iloc[0]['price']
            snapshot['mid_price'] = (best_bid + best_ask) / 2
            snapshot['spread'] = best_ask - best_bid
        
        return snapshot
    
    @log_performance
    def resample_data(self, df: pd.DataFrame, 
                     frequency: str = '1S',
                     aggregation_method: str = 'last') -> pd.DataFrame:
        """
        Resample tick data to regular time intervals
        
        Args:
            df: Input DataFrame with timestamp column
            frequency: Resampling frequency (e.g., '1S', '100ms', '1min')
            aggregation_method: How to aggregate ('last', 'mean', 'ohlc')
            
        Returns:
            Resampled DataFrame
            
        Educational Notes:
            - Resampling converts irregular tick data to regular intervals
            - This is useful for time series analysis and visualization
            - Different aggregation methods serve different purposes
        """
        if len(df) == 0 or 'timestamp' not in df.columns:
            return df
        
        self.logger.info(f"Resampling data to {frequency} using {aggregation_method} method")
        
        df = df.copy()
        df = df.set_index('timestamp')
        
        if aggregation_method == 'last':
            resampled = df.resample(frequency).last()
        elif aggregation_method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            resampled = df[numeric_cols].resample(frequency).mean()
        elif aggregation_method == 'ohlc':
            if 'price' in df.columns:
                ohlc = df['price'].resample(frequency).ohlc()
                volume = df['volume'].resample(frequency).sum() if 'volume' in df.columns else None
                
                resampled = ohlc
                if volume is not None:
                    resampled['volume'] = volume
            else:
                resampled = df.resample(frequency).last()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()
        
        # Remove rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        self.logger.info(f"Resampled from {len(df)} to {len(resampled)} records")
        
        return resampled
    
    def create_features_for_ml(self, df: pd.DataFrame, 
                              target_column: str = 'price',
                              prediction_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features suitable for machine learning models
        
        Args:
            df: Input DataFrame
            target_column: Column to predict
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            Tuple of (features_df, target_series)
            
        Educational Notes:
            - ML features should be stationary and not look into the future
            - We create lagged features to avoid data leakage
            - Target variable is shifted to represent future values
        """
        if len(df) == 0 or target_column not in df.columns:
            return pd.DataFrame(), pd.Series()
        
        self.logger.info(f"Creating ML features for {target_column} prediction")
        
        df = df.copy()
        features = pd.DataFrame(index=df.index)
        
        # Lagged features (avoid lookahead bias)
        for lag in [1, 2, 3, 5, 10]:
            features[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics features
        for window in [5, 10, 20]:
            if len(df) >= window:
                features[f'{target_column}_mean_{window}'] = df[target_column].rolling(window).mean().shift(1)
                features[f'{target_column}_std_{window}'] = df[target_column].rolling(window).std().shift(1)
                features[f'{target_column}_min_{window}'] = df[target_column].rolling(window).min().shift(1)
                features[f'{target_column}_max_{window}'] = df[target_column].rolling(window).max().shift(1)
        
        # Technical indicators
        if 'volume' in df.columns:
            features['volume_lag_1'] = df['volume'].shift(1)
            features['volume_mean_10'] = df['volume'].rolling(10).mean().shift(1)
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['minute'] = df['timestamp'].dt.minute
            features['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Market microstructure features
        if 'spread' in df.columns:
            features['spread_lag_1'] = df['spread'].shift(1)
        
        if 'order_flow_10' in df.columns:
            features['order_flow_lag_1'] = df['order_flow_10'].shift(1)
        
        # Create target variable (future values)
        target = df[target_column].shift(-prediction_horizon)
        
        # Remove rows with NaN values
        feature_valid = features.notna().all(axis=1)
        target_valid = target.notna()
        valid_indices = feature_valid & target_valid
        features = features[valid_indices]
        target = target[valid_indices]
        
        self.logger.info(f"Created {len(features.columns)} features for {len(features)} samples")
        
        return features, target
    
    def normalize_features(self, df: pd.DataFrame, 
                          method: str = 'zscore',
                          exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize features for better model performance
        
        Args:
            df: DataFrame with features to normalize
            method: Normalization method ('zscore', 'minmax', 'robust')
            exclude_columns: Columns to exclude from normalization
            
        Returns:
            Normalized DataFrame
        """
        if len(df) == 0:
            return df
        
        exclude_columns = exclude_columns or []
        df = df.copy()
        
        # Select numeric columns for normalization
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]
        
        self.logger.info(f"Normalizing {len(columns_to_normalize)} columns using {method} method")
        
        if method == 'zscore':
            df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
        elif method == 'minmax':
            df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())
        elif method == 'robust':
            median = df[columns_to_normalize].median()
            mad = (df[columns_to_normalize] - median).abs().median()
            df[columns_to_normalize] = (df[columns_to_normalize] - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return df


# Convenience functions
def preprocess_hft_data(df: pd.DataFrame, tick_size: float = 0.01) -> pd.DataFrame:
    """
    Convenience function to preprocess HFT data with default settings
    
    Args:
        df: Raw HFT data
        tick_size: Minimum price increment
        
    Returns:
        Processed DataFrame
    """
    preprocessor = DataPreprocessor(tick_size=tick_size)
    return preprocessor.process_tick_data(df)


def create_order_book_snapshots(df: pd.DataFrame, depth: int = 10) -> pd.DataFrame:
    """
    Convenience function to create order book snapshots
    
    Args:
        df: Tick data
        depth: Order book depth
        
    Returns:
        DataFrame with order book snapshots
    """
    preprocessor = DataPreprocessor()
    return preprocessor.create_order_book_snapshots(df, depth)