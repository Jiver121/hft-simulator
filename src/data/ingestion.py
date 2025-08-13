"""
Data Ingestion Module for HFT Simulator

This module handles loading and initial processing of high-frequency trading data
from various sources, with special focus on Kaggle HFT datasets.

Educational Notes:
- HFT data typically contains tick-by-tick order book updates
- Each row represents an order book event (new order, cancellation, trade)
- Timestamps are usually in microseconds for high precision
- Data can be very large (millions of rows per trading day)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
import warnings
from datetime import datetime, timedelta
import gc

from config.settings import get_config
from src.utils.logger import get_logger, log_performance, log_memory_usage
from src.utils.helpers import (
    Timer, get_memory_usage, check_memory_limit, 
    estimate_dataframe_memory, ensure_directory
)
from src.utils.constants import (
    STANDARD_COLUMNS, DATA_LIMITS, DEFAULT_CHUNK_SIZE,
    OrderSide, OrderType, validate_price, validate_volume
)


class DataIngestion:
    """
    Main class for ingesting HFT data from various sources
    
    This class provides methods to:
    1. Load data from CSV files (most common for Kaggle datasets)
    2. Validate and clean the data
    3. Convert to standardized format
    4. Handle memory-efficient processing for large datasets
    
    Example Usage:
        >>> ingestion = DataIngestion()
        >>> data = ingestion.load_csv('hft_data.csv')
        >>> print(f"Loaded {len(data)} rows")
        
        # For large files, use chunked processing
        >>> for chunk in ingestion.load_csv_chunks('large_hft_data.csv'):
        ...     process_chunk(chunk)
    """
    
    def __init__(self, config=None):
        """
        Initialize the data ingestion system
        
        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        
        # Data processing parameters
        self.chunk_size = self.config.data.chunk_size
        self.max_memory_usage = self.config.data.max_memory_usage
        self.validate_data = self.config.data.validate_data
        self.drop_invalid_rows = self.config.data.drop_invalid_rows
        
        # Column mapping for different data formats
        self.column_mapping = self.config.data.column_mapping.copy()
        
        # Statistics tracking
        self.stats = {
            'total_rows_loaded': 0,
            'total_rows_dropped': 0,
            'files_processed': 0,
            'processing_time': 0.0,
            'memory_peak': 0.0
        }
        
        self.logger.info("DataIngestion initialized")
    
    @log_performance
    @log_memory_usage
    def load_csv(self, 
                 filepath: Union[str, Path], 
                 **kwargs) -> pd.DataFrame:
        """
        Load HFT data from a CSV file
        
        This method loads a complete CSV file into memory. For large files,
        consider using load_csv_chunks() instead.
        
        Args:
            filepath: Path to the CSV file
            **kwargs: Additional arguments passed to pandas.read_csv()
        
        Returns:
            DataFrame with standardized column names and data types
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            MemoryError: If the file is too large for available memory
            
        Educational Notes:
            - CSV is the most common format for HFT data sharing
            - Files can be very large (GB+), so memory management is crucial
            - Data often needs cleaning and standardization
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.logger.info(f"Loading data from {filepath}")
        
        # Check file size and available memory
        file_size_mb = filepath.stat().st_size / 1024 / 1024
        memory_info = get_memory_usage()
        available_mb = memory_info.get('available', 1000)
        
        if file_size_mb > available_mb * 0.5:  # File > 50% of available memory
            self.logger.warning(
                f"Large file detected ({file_size_mb:.1f}MB). "
                f"Consider using load_csv_chunks() for better memory management."
            )
        
        # Default CSV reading parameters optimized for HFT data
        csv_params = {
            'parse_dates': False,  # We'll handle timestamps separately
            'low_memory': False,   # Ensure consistent data types
            'engine': 'c',         # Faster C engine
            'na_values': ['', 'NULL', 'null', 'NaN', 'nan'],
            'keep_default_na': True,
        }
        csv_params.update(kwargs)
        
        try:
            with Timer() as timer:
                # Load the data
                df = pd.read_csv(filepath, **csv_params)
                
                # Standardize column names and data types
                df = self._standardize_dataframe(df)
                
                # Validate and clean if requested
                if self.validate_data:
                    df = self._validate_and_clean(df)
                
                # Update statistics
                self.stats['total_rows_loaded'] += len(df)
                self.stats['files_processed'] += 1
                self.stats['processing_time'] += timer.elapsed()
                
                memory_usage = estimate_dataframe_memory(df)
                self.stats['memory_peak'] = max(self.stats['memory_peak'], memory_usage)
                
                self.logger.info(
                    f"Successfully loaded {len(df):,} rows from {filepath.name} "
                    f"({memory_usage:.1f}MB, {timer.elapsed():.1f}ms)"
                )
                
                return df
                
        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {str(e)}")
            raise
    
    def load_csv_chunks(self, 
                       filepath: Union[str, Path], 
                       chunk_size: Optional[int] = None,
                       **kwargs) -> Iterator[pd.DataFrame]:
        """
        Load HFT data from CSV file in chunks for memory-efficient processing
        
        This is the recommended method for large HFT datasets that don't fit in memory.
        Each chunk is processed independently and can be handled by downstream components.
        
        Args:
            filepath: Path to the CSV file
            chunk_size: Number of rows per chunk (uses config default if None)
            **kwargs: Additional arguments passed to pandas.read_csv()
            
        Yields:
            DataFrame chunks with standardized format
            
        Example:
            >>> ingestion = DataIngestion()
            >>> total_trades = 0
            >>> for chunk in ingestion.load_csv_chunks('large_file.csv'):
            ...     trades = chunk[chunk['order_type'] == 'trade']
            ...     total_trades += len(trades)
            ...     # Process chunk here
            >>> print(f"Total trades: {total_trades}")
        """
        filepath = Path(filepath)
        chunk_size = chunk_size or self.chunk_size
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        self.logger.info(f"Loading data in chunks from {filepath} (chunk_size={chunk_size:,})")
        
        # CSV parameters for chunked reading
        csv_params = {
            'chunksize': chunk_size,
            'parse_dates': False,
            'low_memory': True,    # Important for chunked reading
            'engine': 'c',
            'na_values': ['', 'NULL', 'null', 'NaN', 'nan'],
            'keep_default_na': True,
        }
        csv_params.update(kwargs)
        
        try:
            chunk_count = 0
            total_rows = 0
            
            with Timer() as total_timer:
                # Create chunked reader
                chunk_reader = pd.read_csv(filepath, **csv_params)
                
                for chunk in chunk_reader:
                    with Timer() as chunk_timer:
                        # Check memory usage before processing
                        if not check_memory_limit(self.max_memory_usage * 100):
                            self.logger.warning("Memory usage high, forcing garbage collection")
                            gc.collect()
                        
                        # Standardize the chunk
                        chunk = self._standardize_dataframe(chunk)
                        
                        # Validate and clean if requested
                        if self.validate_data:
                            chunk = self._validate_and_clean(chunk)
                        
                        chunk_count += 1
                        total_rows += len(chunk)
                        
                        self.logger.debug(
                            f"Processed chunk {chunk_count}: {len(chunk):,} rows "
                            f"({chunk_timer.elapsed():.1f}ms)"
                        )
                        
                        yield chunk
                
                # Update statistics
                self.stats['total_rows_loaded'] += total_rows
                self.stats['files_processed'] += 1
                self.stats['processing_time'] += total_timer.elapsed()
                
                self.logger.info(
                    f"Successfully processed {chunk_count} chunks, {total_rows:,} total rows "
                    f"from {filepath.name} ({total_timer.elapsed():.1f}ms)"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load chunks from {filepath}: {str(e)}")
            raise
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame column names and data types
        
        This method converts various HFT data formats to a standard format
        that can be used throughout the simulator.
        
        Args:
            df: Raw DataFrame from CSV
            
        Returns:
            Standardized DataFrame
            
        Educational Notes:
            - Different data sources use different column names
            - We standardize to a common format for consistency
            - Data types are optimized for memory and performance
        """
        df = df.copy()
        
        # Enhanced column mapping with more comprehensive fallback logic
        original_columns = df.columns.tolist()
        column_renames = {}
        
        # Define comprehensive column mappings
        column_mappings = {
            'timestamp': [
                'timestamp', 'time', 'datetime', 'date_time', 'ts', 'date', 'Time',
                'Timestamp', 'TIMESTAMP', 'DateTime', 'DATE_TIME'
            ],
            'price': [
                'price', 'px', 'close', 'last', 'mid_price', 'midprice', 'Price',
                'PRICE', 'Close', 'Last', 'last_price', 'LastPrice', 'bid_price_1',
                'ask_price_1', 'mid'
            ],
            'volume': [
                'volume', 'vol', 'size', 'qty', 'quantity', 'Volume', 'VOLUME',
                'Size', 'Qty', 'Quantity', 'last_volume', 'LastVolume', 'bid_volume_1',
                'ask_volume_1'
            ],
            'side': [
                'side', 'Side', 'SIDE', 'direction', 'Direction', 'type'
            ],
            'order_type': [
                'order_type', 'ordertype', 'type', 'order_kind', 'OrderType',
                'ORDER_TYPE', 'Type'
            ],
            'order_id': [
                'order_id', 'orderid', 'id', 'ID', 'OrderId', 'ORDER_ID', 'ref_id'
            ],
            'symbol': [
                'symbol', 'Symbol', 'SYMBOL', 'instrument', 'Instrument', 'ticker',
                'Ticker', 'security', 'Security'
            ]
        }
        
        # First pass: exact matches from config
        for standard_name, current_name in self.column_mapping.items():
            if current_name in df.columns:
                column_renames[current_name] = standard_name
        
        # Second pass: comprehensive fallback matching
        for standard_col, possible_names in column_mappings.items():
            if standard_col not in [v for v in column_renames.values()]:  # Not already mapped
                for possible_name in possible_names:
                    if possible_name in df.columns and possible_name not in column_renames:
                        column_renames[possible_name] = standard_col
                        break
        
        # Third pass: intelligent inference for missing critical columns
        if 'price' not in [v for v in column_renames.values()]:
            # Look for numeric columns that might be prices
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in column_renames and any(keyword in col.lower() for keyword in ['px', 'price', 'close', 'last', 'mid']):
                    column_renames[col] = 'price'
                    break
            # If still no price found, use the first numeric column that looks reasonable
            if 'price' not in [v for v in column_renames.values()]:
                for col in numeric_cols:
                    if col not in column_renames:
                        sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 0
                        # Heuristic: reasonable price range
                        if 0.001 <= sample_val <= 100000:
                            column_renames[col] = 'price'
                            self.logger.info(f"Inferred price column: {col} (sample value: {sample_val})")
                            break
        
        if 'volume' not in [v for v in column_renames.values()]:
            # Look for integer columns that might be volumes
            for col in df.columns:
                if col not in column_renames and df[col].dtype in ['int64', 'int32', 'float64']:
                    if any(keyword in col.lower() for keyword in ['vol', 'size', 'qty', 'quantity']):
                        column_renames[col] = 'volume'
                        break
            # If still no volume found, create a default volume column
            if 'volume' not in [v for v in column_renames.values()]:
                df['volume'] = 100  # Default volume
                self.logger.info("Created default volume column with value 100")
        
        # Rename columns
        if column_renames:
            df = df.rename(columns=column_renames)
            self.logger.info(f"Column mapping applied: {column_renames}")
        
        # Handle missing timestamp
        if 'timestamp' not in df.columns:
            if len(df) > 0:
                # Create synthetic timestamps
                start_time = pd.Timestamp('2024-01-01 09:30:00')
                df['timestamp'] = pd.date_range(start_time, periods=len(df), freq='1S')
                self.logger.info("Created synthetic timestamp column")
            else:
                df['timestamp'] = pd.Timestamp.now()
        
        # Ensure minimum required columns exist
        required_columns = ['timestamp', 'price', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"Still missing required columns after mapping: {missing_columns}")
            # Create minimal defaults for completely missing columns
            for col in missing_columns:
                if col == 'price' and len(df) > 0:
                    df[col] = 100.0  # Default price
                elif col == 'volume' and len(df) > 0:
                    df[col] = 100  # Default volume
                elif col == 'timestamp' and len(df) > 0:
                    df[col] = pd.Timestamp.now()
            self.logger.info(f"Created default values for missing columns: {missing_columns}")
        
        # Standardize data types
        df = self._optimize_data_types(df)
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            df = self._process_timestamps(df)
        
        # Standardize categorical columns
        if 'side' in df.columns:
            df['side'] = df['side'].astype(str).str.lower()
            # Map various side representations
            side_mapping = {
                'b': 'bid', 'bid': 'bid', 'buy': 'bid',
                's': 'ask', 'ask': 'ask', 'sell': 'ask',
                '1': 'bid', '2': 'ask',  # Some datasets use numeric codes
            }
            df['side'] = df['side'].map(side_mapping).fillna(df['side'])
        
        if 'order_type' in df.columns:
            df['order_type'] = df['order_type'].astype(str).str.lower()
            # Map various order type representations
            type_mapping = {
                'm': 'market', 'market': 'market', 'mkt': 'market',
                'l': 'limit', 'limit': 'limit', 'lmt': 'limit',
                'c': 'cancel', 'cancel': 'cancel', 'cnl': 'cancel',
            }
            df['order_type'] = df['order_type'].map(type_mapping).fillna(df['order_type'])
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types for memory efficiency and performance
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            DataFrame with optimized data types
        """
        df = df.copy()
        
        # Optimize numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                continue
                
            if col in ['price']:
                # Prices need high precision
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            elif col in ['volume', 'order_id']:
                # Volumes and IDs can be integers
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].notna().all() and (df[col] >= 0).all():
                    max_val = df[col].max()
                    if max_val < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype('uint32')
                    else:
                        df[col] = df[col].astype('uint64')
            elif df[col].dtype in ['int64', 'float64']:
                # Try to downcast other numeric columns
                if df[col].dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert string columns to categories if they have few unique values
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['timestamp']:  # Don't categorize timestamps
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    df[col] = df[col].astype('category')
        
        return df
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and standardize timestamp columns
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with processed timestamps
        """
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        try:
            # Multiple strategies for timestamp parsing
            original_dtype = df['timestamp'].dtype
            
            if df['timestamp'].dtype == 'object':
                # String timestamps - try multiple parsing strategies
                # First, try pandas default parsing
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                # If many failed, try common formats
                if df['timestamp'].isna().sum() > len(df) * 0.1:  # More than 10% failed
                    common_formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f',
                        '%m/%d/%Y %H:%M:%S',
                        '%d/%m/%Y %H:%M:%S',
                        '%Y%m%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S.%f',
                        '%Y-%m-%dT%H:%M:%S.%fZ',
                    ]
                    
                    for fmt in common_formats:
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                            if df['timestamp'].isna().sum() < len(df) * 0.1:
                                self.logger.info(f"Successfully parsed timestamps with format: {fmt}")
                                break
                        except:
                            continue
            
            elif df['timestamp'].dtype in ['int64', 'float64']:
                # Numeric timestamps - enhanced unit detection
                sample_values = df['timestamp'].dropna().head(10).values
                if len(sample_values) > 0:
                    avg_ts = np.mean(sample_values)
                    
                    # More sophisticated unit detection
                    current_epoch = pd.Timestamp.now().timestamp()
                    
                    # Test different units and see which gives reasonable dates
                    units_to_test = [
                        ('ns', 1e9),  # nanoseconds
                        ('us', 1e6),  # microseconds  
                        ('ms', 1e3),  # milliseconds
                        ('s', 1),     # seconds
                    ]
                    
                    best_unit = 's'  # default
                    for unit, divisor in units_to_test:
                        test_timestamp = avg_ts / divisor
                        # Check if timestamp is within reasonable range (1990-2050)
                        if 631152000 <= test_timestamp <= 2524608000:  # 1990-2050 in seconds
                            best_unit = unit
                            break
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit=best_unit, errors='coerce')
                    self.logger.info(f"Parsed numeric timestamps using unit: {best_unit}")
            
            # Handle any remaining NaT values
            nat_count = df['timestamp'].isna().sum()
            if nat_count > 0:
                self.logger.warning(f"{nat_count} timestamps could not be parsed, creating synthetic timestamps")
                
                # Fill NaT values with synthetic timestamps
                if nat_count < len(df):  # Some valid timestamps exist
                    # Interpolate missing timestamps
                    df['timestamp'] = df['timestamp'].interpolate(method='time')
                else:
                    # All timestamps are invalid, create synthetic ones
                    start_time = pd.Timestamp('2024-01-01 09:30:00')
                    df['timestamp'] = pd.date_range(start_time, periods=len(df), freq='1S')
                    self.logger.info("Created synthetic timestamps for entire dataset")
            
            # Ensure timestamps are sorted
            if not df['timestamp'].is_monotonic_increasing:
                df = df.sort_values('timestamp').reset_index(drop=True)
                self.logger.info("Sorted data by timestamp")
            
            # Log timestamp range
            if len(df) > 0:
                ts_min, ts_max = df['timestamp'].min(), df['timestamp'].max()
                duration = ts_max - ts_min
                self.logger.info(f"Timestamp range: {ts_min} to {ts_max} (duration: {duration})")
            
        except Exception as e:
            self.logger.error(f"Failed to process timestamps: {str(e)}")
            # Fallback: create synthetic timestamps
            try:
                start_time = pd.Timestamp('2024-01-01 09:30:00')
                df['timestamp'] = pd.date_range(start_time, periods=len(df), freq='1S')
                self.logger.info("Created fallback synthetic timestamps")
            except Exception as fallback_error:
                self.logger.error(f"Even fallback timestamp creation failed: {fallback_error}")
        
        return df
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the data according to business rules
        
        Args:
            df: DataFrame to validate and clean
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Remove rows with missing critical data
        critical_columns = ['timestamp', 'price', 'volume']
        existing_critical = [col for col in critical_columns if col in df.columns]
        
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Validate price ranges
        if 'price' in df.columns:
            valid_prices = (df['price'] >= DATA_LIMITS['min_price']) & \
                          (df['price'] <= DATA_LIMITS['max_price'])
            df = df[valid_prices]
        
        # Validate volume ranges
        if 'volume' in df.columns:
            valid_volumes = (df['volume'] >= DATA_LIMITS['min_volume']) & \
                           (df['volume'] <= DATA_LIMITS['max_volume'])
            df = df[valid_volumes]
        
        # Remove duplicate timestamps (keep last)
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Log cleaning results
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            self.stats['total_rows_dropped'] += rows_dropped
            drop_pct = (rows_dropped / initial_rows) * 100
            self.logger.info(f"Dropped {rows_dropped:,} invalid rows ({drop_pct:.1f}%)")
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data statistics and information
        """
        info = {
            'shape': df.shape,
            'memory_usage_mb': estimate_dataframe_memory(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': None,
            'price_range': None,
            'volume_stats': None,
        }
        
        # Timestamp information
        if 'timestamp' in df.columns and not df['timestamp'].empty:
            info['date_range'] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration': df['timestamp'].max() - df['timestamp'].min(),
            }
        
        # Price information
        if 'price' in df.columns and not df['price'].empty:
            info['price_range'] = {
                'min': df['price'].min(),
                'max': df['price'].max(),
                'mean': df['price'].mean(),
                'std': df['price'].std(),
            }
        
        # Volume information
        if 'volume' in df.columns and not df['volume'].empty:
            info['volume_stats'] = {
                'total': df['volume'].sum(),
                'mean': df['volume'].mean(),
                'median': df['volume'].median(),
                'std': df['volume'].std(),
            }
        
        return info
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the data processing operations
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_rows_loaded': 0,
            'total_rows_dropped': 0,
            'files_processed': 0,
            'processing_time': 0.0,
            'memory_peak': 0.0
        }
        self.logger.info("Processing statistics reset")


# Convenience functions for quick data loading
def load_hft_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Convenience function to quickly load HFT data
    
    Args:
        filepath: Path to the data file
        **kwargs: Additional arguments for DataIngestion
        
    Returns:
        Loaded and processed DataFrame
    """
    ingestion = DataIngestion()
    return ingestion.load_csv(filepath, **kwargs)


def load_hft_data_chunks(filepath: Union[str, Path], 
                        chunk_size: int = DEFAULT_CHUNK_SIZE,
                        **kwargs) -> Iterator[pd.DataFrame]:
    """
    Convenience function to load HFT data in chunks
    
    Args:
        filepath: Path to the data file
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for DataIngestion
        
    Yields:
        DataFrame chunks
    """
    ingestion = DataIngestion()
    yield from ingestion.load_csv_chunks(filepath, chunk_size=chunk_size, **kwargs)