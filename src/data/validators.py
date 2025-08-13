"""
Data Validation Module for HFT Simulator

This module provides comprehensive validation for HFT data to ensure data quality
and consistency before processing by the order book engine.

Educational Notes:
- Data validation is crucial in HFT systems to prevent erroneous trades
- Invalid data can cause significant financial losses in real trading
- We check for logical consistency, data ranges, and temporal ordering
- Validation rules are based on market microstructure principles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings

from src.utils.logger import get_logger, log_performance
from src.utils.helpers import detect_outliers, clean_price_data
from src.utils.constants import (
    DATA_LIMITS, OrderSide, OrderType, 
    validate_price, validate_volume, EPSILON
)


class ValidationResult:
    """
    Container for validation results with detailed error reporting
    
    This class stores the results of data validation operations,
    including error counts, warnings, and detailed error messages.
    """
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_rows': 0,
            'valid_rows': 0,
            'error_rows': 0,
            'warning_rows': 0,
        }
        self.error_details = {}
    
    def add_error(self, error_type: str, message: str, row_indices: List[int] = None):
        """Add an error to the validation result"""
        self.is_valid = False
        self.errors.append({
            'type': error_type,
            'message': message,
            'count': len(row_indices) if row_indices else 1,
            'rows': row_indices[:10] if row_indices else []  # Store first 10 for debugging
        })
        
        if error_type not in self.error_details:
            self.error_details[error_type] = 0
        self.error_details[error_type] += len(row_indices) if row_indices else 1
    
    def add_warning(self, warning_type: str, message: str, row_indices: List[int] = None):
        """Add a warning to the validation result"""
        self.warnings.append({
            'type': warning_type,
            'message': message,
            'count': len(row_indices) if row_indices else 1,
            'rows': row_indices[:10] if row_indices else []
        })
    
    def update_stats(self, total_rows: int, valid_rows: int):
        """Update validation statistics"""
        self.stats['total_rows'] = total_rows
        self.stats['valid_rows'] = valid_rows
        self.stats['error_rows'] = total_rows - valid_rows
        self.stats['warning_rows'] = sum(w['count'] for w in self.warnings)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results"""
        return {
            'is_valid': self.is_valid,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'stats': self.stats,
            'error_types': list(self.error_details.keys()),
            'error_counts': self.error_details,
        }
    
    def print_summary(self):
        """Print a human-readable summary of validation results"""
        print(f"\n=== Data Validation Summary ===")
        print(f"Total Rows: {self.stats['total_rows']:,}")
        print(f"Valid Rows: {self.stats['valid_rows']:,}")
        print(f"Error Rows: {self.stats['error_rows']:,}")
        print(f"Warning Rows: {self.stats['warning_rows']:,}")
        print(f"Overall Status: {'PASS' if self.is_valid else 'FAIL'}")
        
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error['type']}: {error['message']} ({error['count']} rows)")
        
        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning['type']}: {warning['message']} ({warning['count']} rows)")


class DataValidator:
    """
    Comprehensive data validator for HFT datasets
    
    This class performs various validation checks on HFT data including:
    - Schema validation (required columns, data types)
    - Range validation (prices, volumes within reasonable bounds)
    - Logical validation (bid <= ask, positive volumes, etc.)
    - Temporal validation (chronological order, reasonable timestamps)
    - Market microstructure validation (spread checks, tick size compliance)
    
    Example Usage:
        >>> validator = DataValidator()
        >>> result = validator.validate_dataframe(df)
        >>> if result.is_valid:
        ...     print("Data is valid!")
        ... else:
        ...     result.print_summary()
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the data validator
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.logger = get_logger(__name__)
        self.strict_mode = strict_mode
        
        # Validation thresholds
        self.price_outlier_threshold = 5.0  # Standard deviations
        self.volume_outlier_threshold = 5.0
        self.spread_threshold = 0.1  # 10% of mid-price
        self.timestamp_gap_threshold = timedelta(hours=1)  # Max gap between timestamps
        
        self.logger.info(f"DataValidator initialized (strict_mode={strict_mode})")
    
    @log_performance
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive validation on a DataFrame
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with detailed validation information
            
        Educational Notes:
            - We perform multiple types of validation checks
            - Each check can produce errors (data is invalid) or warnings (suspicious but usable)
            - The order of checks matters - basic checks first, then complex ones
        """
        result = ValidationResult()
        
        if df.empty:
            result.add_error('empty_data', 'DataFrame is empty')
            return result
        
        self.logger.info(f"Validating DataFrame with {len(df):,} rows")
        
        # 1. Schema validation
        self._validate_schema(df, result)
        
        # 2. Data type validation
        self._validate_data_types(df, result)
        
        # 3. Range validation
        self._validate_ranges(df, result)
        
        # 4. Missing data validation
        self._validate_missing_data(df, result)
        
        # 5. Logical consistency validation
        self._validate_logical_consistency(df, result)
        
        # 6. Temporal validation
        self._validate_temporal_consistency(df, result)
        
        # 7. Market microstructure validation
        self._validate_market_microstructure(df, result)
        
        # 8. Statistical validation (outlier detection)
        self._validate_statistical_properties(df, result)
        
        # Update final statistics
        valid_rows = len(df) - sum(result.error_details.values())
        result.update_stats(len(df), max(0, valid_rows))
        
        self.logger.info(f"Validation completed: {result.stats['valid_rows']:,}/{result.stats['total_rows']:,} valid rows")
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame, result: ValidationResult):
        """Validate DataFrame schema (required columns)"""
        required_columns = ['timestamp', 'price', 'volume']
        optional_columns = ['side', 'order_type', 'order_id', 'symbol']
        
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            result.add_error(
                'missing_columns',
                f'Missing required columns: {missing_required}'
            )
        
        missing_optional = [col for col in optional_columns if col not in df.columns]
        if missing_optional:
            result.add_warning(
                'missing_optional_columns',
                f'Missing optional columns: {missing_optional}'
            )
        
        # Check for unexpected columns
        expected_columns = set(required_columns + optional_columns)
        unexpected_columns = set(df.columns) - expected_columns
        if unexpected_columns:
            result.add_warning(
                'unexpected_columns',
                f'Unexpected columns found: {list(unexpected_columns)}'
            )
    
    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data types of columns"""
        
        # Price should be numeric
        if 'price' in df.columns:
            non_numeric_prices = df[~pd.to_numeric(df['price'], errors='coerce').notna()]
            if not non_numeric_prices.empty:
                result.add_error(
                    'invalid_price_type',
                    f'Non-numeric prices found',
                    non_numeric_prices.index.tolist()
                )
        
        # Volume should be numeric and integer-like
        if 'volume' in df.columns:
            non_numeric_volumes = df[~pd.to_numeric(df['volume'], errors='coerce').notna()]
            if not non_numeric_volumes.empty:
                result.add_error(
                    'invalid_volume_type',
                    f'Non-numeric volumes found',
                    non_numeric_volumes.index.tolist()
                )
        
        # Timestamp should be datetime-convertible
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'], errors='coerce')
            except Exception:
                result.add_error(
                    'invalid_timestamp_type',
                    'Timestamps cannot be converted to datetime'
                )
    
    def _validate_ranges(self, df: pd.DataFrame, result: ValidationResult):
        """Validate that values are within reasonable ranges"""
        
        # Price range validation
        if 'price' in df.columns:
            numeric_prices = pd.to_numeric(df['price'], errors='coerce')
            
            # Check for negative prices
            negative_prices = df[numeric_prices < 0]
            if not negative_prices.empty:
                result.add_error(
                    'negative_prices',
                    f'Negative prices found',
                    negative_prices.index.tolist()
                )
            
            # Check for zero prices
            zero_prices = df[numeric_prices == 0]
            if not zero_prices.empty:
                result.add_error(
                    'zero_prices',
                    f'Zero prices found',
                    zero_prices.index.tolist()
                )
            
            # Check for extremely high prices
            high_prices = df[numeric_prices > DATA_LIMITS['max_price']]
            if not high_prices.empty:
                result.add_warning(
                    'high_prices',
                    f'Extremely high prices found (>{DATA_LIMITS["max_price"]})',
                    high_prices.index.tolist()
                )
            
            # Check for extremely low prices
            low_prices = df[(numeric_prices > 0) & (numeric_prices < DATA_LIMITS['min_price'])]
            if not low_prices.empty:
                result.add_warning(
                    'low_prices',
                    f'Extremely low prices found (<{DATA_LIMITS["min_price"]})',
                    low_prices.index.tolist()
                )
        
        # Volume range validation
        if 'volume' in df.columns:
            numeric_volumes = pd.to_numeric(df['volume'], errors='coerce')
            
            # Check for negative volumes
            negative_volumes = df[numeric_volumes < 0]
            if not negative_volumes.empty:
                result.add_error(
                    'negative_volumes',
                    f'Negative volumes found',
                    negative_volumes.index.tolist()
                )
            
            # Check for zero volumes
            zero_volumes = df[numeric_volumes == 0]
            if not zero_volumes.empty:
                result.add_error(
                    'zero_volumes',
                    f'Zero volumes found',
                    zero_volumes.index.tolist()
                )
            
            # Check for extremely high volumes
            high_volumes = df[numeric_volumes > DATA_LIMITS['max_volume']]
            if not high_volumes.empty:
                result.add_warning(
                    'high_volumes',
                    f'Extremely high volumes found (>{DATA_LIMITS["max_volume"]})',
                    high_volumes.index.tolist()
                )
    
    def _validate_missing_data(self, df: pd.DataFrame, result: ValidationResult):
        """Validate missing data patterns"""
        
        critical_columns = ['timestamp', 'price', 'volume']
        
        for col in critical_columns:
            if col in df.columns:
                missing_data = df[df[col].isna()]
                if not missing_data.empty:
                    result.add_error(
                        f'missing_{col}',
                        f'Missing values in critical column {col}',
                        missing_data.index.tolist()
                    )
        
        # Check for rows with all missing data
        all_missing = df[df.isna().all(axis=1)]
        if not all_missing.empty:
            result.add_error(
                'empty_rows',
                f'Rows with all missing data found',
                all_missing.index.tolist()
            )
    
    def _validate_logical_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """Validate logical consistency of the data"""
        
        # If we have both bid and ask data, validate spread consistency
        if 'side' in df.columns and 'price' in df.columns:
            # Group by timestamp to check bid-ask spreads
            if 'timestamp' in df.columns:
                grouped = df.groupby('timestamp')
                
                for timestamp, group in grouped:
                    bids = group[group['side'] == 'bid']
                    asks = group[group['side'] == 'ask']
                    
                    if not bids.empty and not asks.empty:
                        max_bid = bids['price'].max()
                        min_ask = asks['price'].min()
                        
                        # Bid should not be higher than ask (crossed market)
                        if max_bid > min_ask:
                            crossed_rows = group.index.tolist()
                            result.add_error(
                                'crossed_market',
                                f'Crossed market detected at {timestamp}: bid={max_bid}, ask={min_ask}',
                                crossed_rows
                            )
        
        # Validate order types and sides consistency
        if 'order_type' in df.columns:
            valid_order_types = ['market', 'limit', 'cancel', 'modify']
            invalid_types = df[~df['order_type'].isin(valid_order_types)]
            if not invalid_types.empty:
                result.add_warning(
                    'invalid_order_types',
                    f'Invalid order types found: {invalid_types["order_type"].unique()}',
                    invalid_types.index.tolist()
                )
        
        if 'side' in df.columns:
            valid_sides = ['bid', 'ask', 'buy', 'sell']
            invalid_sides = df[~df['side'].isin(valid_sides)]
            if not invalid_sides.empty:
                result.add_warning(
                    'invalid_sides',
                    f'Invalid sides found: {invalid_sides["side"].unique()}',
                    invalid_sides.index.tolist()
                )
    
    def _validate_temporal_consistency(self, df: pd.DataFrame, result: ValidationResult):
        """Validate temporal consistency and ordering"""
        
        if 'timestamp' not in df.columns:
            return
        
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Check for invalid timestamps
        invalid_timestamps = df[timestamps.isna()]
        if not invalid_timestamps.empty:
            result.add_error(
                'invalid_timestamps',
                f'Invalid timestamps found',
                invalid_timestamps.index.tolist()
            )
            return
        
        # Check if data is sorted by timestamp
        if not timestamps.is_monotonic_increasing:
            result.add_warning(
                'unsorted_timestamps',
                'Data is not sorted by timestamp'
            )
        
        # Check for duplicate timestamps
        duplicate_timestamps = df[timestamps.duplicated()]
        if not duplicate_timestamps.empty:
            result.add_warning(
                'duplicate_timestamps',
                f'Duplicate timestamps found',
                duplicate_timestamps.index.tolist()
            )
        
        # Check for unreasonable timestamp gaps
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            large_gaps = time_diffs[time_diffs > self.timestamp_gap_threshold]
            
            if not large_gaps.empty:
                result.add_warning(
                    'large_timestamp_gaps',
                    f'Large timestamp gaps found (>{self.timestamp_gap_threshold})',
                    large_gaps.index.tolist()
                )
        
        # Check for timestamps in the future (if strict mode)
        if self.strict_mode:
            future_timestamps = df[timestamps > pd.Timestamp.now()]
            if not future_timestamps.empty:
                result.add_warning(
                    'future_timestamps',
                    f'Future timestamps found',
                    future_timestamps.index.tolist()
                )
    
    def _validate_market_microstructure(self, df: pd.DataFrame, result: ValidationResult):
        """Validate market microstructure properties"""
        
        if 'price' not in df.columns:
            return
        
        prices = pd.to_numeric(df['price'], errors='coerce').dropna()
        
        if len(prices) < 2:
            return
        
        # Check for reasonable tick sizes
        price_diffs = prices.diff().dropna()
        non_zero_diffs = price_diffs[price_diffs != 0]
        
        if not non_zero_diffs.empty:
            min_tick = non_zero_diffs.abs().min()
            
            # Warn if tick size is very small (sub-penny)
            if min_tick < 0.001:
                result.add_warning(
                    'small_tick_size',
                    f'Very small tick size detected: {min_tick}'
                )
        
        # Check for reasonable price volatility
        if len(prices) > 100:  # Need sufficient data
            returns = prices.pct_change().dropna()
            volatility = returns.std()
            
            # Warn if volatility is extremely high (>50% per tick)
            if volatility > 0.5:
                result.add_warning(
                    'high_volatility',
                    f'Extremely high price volatility: {volatility:.4f}'
                )
        
        # Validate spreads if we have bid/ask data
        if 'side' in df.columns and 'timestamp' in df.columns:
            self._validate_spreads(df, result)
    
    def _validate_spreads(self, df: pd.DataFrame, result: ValidationResult):
        """Validate bid-ask spreads"""
        
        # Group by timestamp to calculate spreads
        grouped = df.groupby('timestamp')
        wide_spreads = []
        
        for timestamp, group in grouped:
            bids = group[group['side'] == 'bid']
            asks = group[group['side'] == 'ask']
            
            if not bids.empty and not asks.empty:
                best_bid = bids['price'].max()
                best_ask = asks['price'].min()
                
                if best_ask > best_bid:  # Normal market
                    spread = best_ask - best_bid
                    mid_price = (best_bid + best_ask) / 2
                    spread_pct = spread / mid_price
                    
                    # Check for unreasonably wide spreads
                    if spread_pct > self.spread_threshold:
                        wide_spreads.extend(group.index.tolist())
        
        if wide_spreads:
            result.add_warning(
                'wide_spreads',
                f'Wide spreads detected (>{self.spread_threshold*100:.1f}% of mid-price)',
                wide_spreads
            )
    
    def _validate_statistical_properties(self, df: pd.DataFrame, result: ValidationResult):
        """Validate statistical properties and detect outliers"""
        
        # Price outlier detection
        if 'price' in df.columns:
            prices = pd.to_numeric(df['price'], errors='coerce').dropna()
            
            if len(prices) > 10:  # Need sufficient data for outlier detection
                try:
                    outliers = detect_outliers(prices, method='modified_zscore', 
                                             threshold=self.price_outlier_threshold)
                    
                    if outliers.any():
                        outlier_indices = df.index[outliers].tolist()
                        result.add_warning(
                            'price_outliers',
                            f'Price outliers detected',
                            outlier_indices
                        )
                except Exception as e:
                    self.logger.warning(f"Price outlier detection failed: {str(e)}")
        
        # Volume outlier detection
        if 'volume' in df.columns:
            volumes = pd.to_numeric(df['volume'], errors='coerce').dropna()
            
            if len(volumes) > 10:
                try:
                    outliers = detect_outliers(volumes, method='iqr', 
                                             threshold=self.volume_outlier_threshold)
                    
                    if outliers.any():
                        outlier_indices = df.index[outliers].tolist()
                        result.add_warning(
                            'volume_outliers',
                            f'Volume outliers detected',
                            outlier_indices
                        )
                except Exception as e:
                    self.logger.warning(f"Volume outlier detection failed: {str(e)}")
    
    def validate_order_book_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Specialized validation for order book data
        
        Args:
            df: DataFrame with order book data
            
        Returns:
            ValidationResult with order book specific validation
        """
        result = self.validate_dataframe(df)
        
        # Additional order book specific validations
        if 'side' in df.columns and 'price' in df.columns and 'volume' in df.columns:
            # Check for proper order book structure
            self._validate_order_book_structure(df, result)
        
        return result
    
    def _validate_order_book_structure(self, df: pd.DataFrame, result: ValidationResult):
        """Validate order book structure and properties"""
        
        # Check that we have both bids and asks
        if 'side' in df.columns:
            sides = df['side'].unique()
            
            if 'bid' not in sides and 'buy' not in sides:
                result.add_warning(
                    'missing_bids',
                    'No bid/buy orders found in data'
                )
            
            if 'ask' not in sides and 'sell' not in sides:
                result.add_warning(
                    'missing_asks',
                    'No ask/sell orders found in data'
                )
        
        # Validate order book depth consistency
        if 'timestamp' in df.columns:
            # Sample a few timestamps to check book structure
            sample_timestamps = df['timestamp'].drop_duplicates().sample(
                min(10, len(df['timestamp'].unique()))
            )
            
            for timestamp in sample_timestamps:
                book_snapshot = df[df['timestamp'] == timestamp]
                
                # Check for reasonable number of price levels
                if len(book_snapshot) > 1000:  # Very deep book
                    result.add_warning(
                        'very_deep_book',
                        f'Very deep order book at {timestamp}: {len(book_snapshot)} levels'
                    )


# Convenience functions
def validate_hft_data(df: pd.DataFrame, strict_mode: bool = False) -> ValidationResult:
    """
    Convenience function to validate HFT data
    
    Args:
        df: DataFrame to validate
        strict_mode: Whether to apply strict validation rules
        
    Returns:
        ValidationResult
    """
    validator = DataValidator(strict_mode=strict_mode)
    return validator.validate_dataframe(df)


def quick_data_check(df: pd.DataFrame) -> bool:
    """
    Quick data quality check - returns True if data passes basic validation
    
    Args:
        df: DataFrame to check
        
    Returns:
        True if data passes basic validation
    """
    result = validate_hft_data(df, strict_mode=False)
    return result.is_valid and result.stats['valid_rows'] > 0