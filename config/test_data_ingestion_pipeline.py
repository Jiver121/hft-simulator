#!/usr/bin/env python3
"""
Test script for the fixed Data Ingestion Pipeline

This script tests the enhanced DataIngestion.load_csv() method with various
CSV formats and validates that the pipeline works end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(num_rows=1000, symbol='AAPL', format_type='standard'):
    """
    Generate sample HFT data for testing
    
    Args:
        num_rows: Number of rows to generate
        symbol: Symbol for the data
        format_type: Type of format to generate ('standard', 'kaggle', 'messy')
    
    Returns:
        DataFrame with sample data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_time = pd.Timestamp('2024-01-01 09:30:00')
    timestamps = pd.date_range(start_time, periods=num_rows, freq='1S')
    
    # Generate price data (random walk)
    initial_price = 150.0
    price_changes = np.random.normal(0, 0.001, num_rows)  # 0.1% volatility
    prices = initial_price * np.exp(np.cumsum(price_changes))
    
    # Generate volumes
    volumes = np.random.randint(1, 1000, num_rows)
    
    # Generate sides and order types
    sides = np.random.choice(['bid', 'ask'], num_rows)
    order_types = np.random.choice(['limit', 'market', 'cancel'], num_rows)
    order_ids = np.arange(1, num_rows + 1)
    
    if format_type == 'standard':
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'price': prices,
            'volume': volumes,
            'side': sides,
            'order_type': order_types,
            'order_id': order_ids
        })
    
    elif format_type == 'kaggle':
        # Simulate Kaggle-style column names
        df = pd.DataFrame({
            'Time': timestamps.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'Symbol': symbol,
            'Price': prices,
            'Size': volumes,
            'Side': np.where(np.random.random(num_rows) > 0.5, 'B', 'S'),
            'Type': np.random.choice(['L', 'M', 'C'], num_rows),
            'ID': order_ids
        })
    
    elif format_type == 'messy':
        # Simulate messy data with inconsistent column names and missing values
        df = pd.DataFrame({
            'ts': timestamps.astype(np.int64) // 10**9,  # Unix timestamps
            'instrument': symbol,
            'px': prices,
            'qty': volumes,
            'direction': np.random.choice([1, 2], num_rows),  # 1=buy, 2=sell
            'ordertype': np.random.choice(['LMT', 'MKT'], num_rows),
            'ref_id': order_ids
        })
        
        # Introduce some missing values
        mask = np.random.random(num_rows) < 0.05  # 5% missing
        df.loc[mask, 'px'] = np.nan
        # Apply missing values to qty for a subset
        mask_indices = np.where(mask)[0][:50]  # Get first 50 indices where mask is True
        if len(mask_indices) > 0:
            df.loc[mask_indices, 'qty'] = np.nan
    
    elif format_type == 'minimal':
        # Only basic columns available
        df = pd.DataFrame({
            'date_time': timestamps,
            'close': prices,
            'vol': volumes
        })
    
    elif format_type == 'corrupted':
        # Corrupted data for testing error handling
        df = pd.DataFrame({
            'timestamp': ['invalid_date'] * (num_rows // 2) + timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()[:(num_rows - num_rows // 2)],
            'price': ['not_a_number'] * 10 + prices.tolist()[:-10],
            'volume': volumes
        })
    
    return df

def test_data_ingestion_formats():
    """Test data ingestion with various formats"""
    from src.data.ingestion import DataIngestion
    
    logger.info("Testing DataIngestion with various formats...")
    
    formats_to_test = ['standard', 'kaggle', 'messy', 'minimal', 'corrupted']
    
    for format_type in formats_to_test:
        logger.info(f"\n=== Testing {format_type} format ===")
        
        # Generate sample data
        sample_data = generate_sample_data(1000, format_type=format_type)
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            # Test data ingestion
            ingestion = DataIngestion()
            loaded_data = ingestion.load_csv(tmp_file_path)
            
            logger.info(f"Successfully loaded {len(loaded_data)} rows")
            logger.info(f"Columns: {list(loaded_data.columns)}")
            logger.info(f"Data types: {loaded_data.dtypes.to_dict()}")
            
            # Validate required columns exist
            required_cols = ['timestamp', 'price', 'volume']
            missing_cols = [col for col in required_cols if col not in loaded_data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
            else:
                logger.info("✓ All required columns present")
            
            # Validate data quality
            if 'timestamp' in loaded_data.columns:
                ts_range = loaded_data['timestamp'].max() - loaded_data['timestamp'].min()
                logger.info(f"Timestamp range: {ts_range}")
            
            if 'price' in loaded_data.columns:
                price_stats = loaded_data['price'].describe()
                logger.info(f"Price range: ${price_stats['min']:.2f} - ${price_stats['max']:.2f}")
            
            # Test data info functionality
            data_info = ingestion.get_data_info(loaded_data)
            logger.info(f"Data info: {data_info}")
            
            logger.info(f"✅ {format_type} format test PASSED")
            
        except Exception as e:
            logger.error(f"❌ {format_type} format test FAILED: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def test_chunked_processing():
    """Test chunked data processing"""
    from src.data.ingestion import DataIngestion
    
    logger.info("\n=== Testing Chunked Processing ===")
    
    # Generate larger dataset
    large_data = generate_sample_data(5000, format_type='standard')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        large_data.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    try:
        ingestion = DataIngestion()
        
        total_rows = 0
        chunk_count = 0
        
        for chunk in ingestion.load_csv_chunks(tmp_file_path, chunk_size=500):
            chunk_count += 1
            total_rows += len(chunk)
            logger.info(f"Processed chunk {chunk_count}: {len(chunk)} rows")
        
        logger.info(f"✓ Processed {chunk_count} chunks, {total_rows} total rows")
        logger.info("✅ Chunked processing test PASSED")
        
    except Exception as e:
        logger.error(f"❌ Chunked processing test FAILED: {str(e)}")
    
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass

def test_convenience_functions():
    """Test convenience functions"""
    from src.data.ingestion import load_hft_data, load_hft_data_chunks
    
    logger.info("\n=== Testing Convenience Functions ===")
    
    # Generate test data
    test_data = generate_sample_data(500, format_type='standard')
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
        test_data.to_csv(tmp_file.name, index=False)
        tmp_file_path = tmp_file.name
    
    try:
        # Test load_hft_data
        data = load_hft_data(tmp_file_path)
        logger.info(f"✓ load_hft_data: loaded {len(data)} rows")
        
        # Test load_hft_data_chunks
        total_rows = 0
        for chunk in load_hft_data_chunks(tmp_file_path, chunk_size=100):
            total_rows += len(chunk)
        logger.info(f"✓ load_hft_data_chunks: processed {total_rows} rows")
        
        logger.info("✅ Convenience functions test PASSED")
        
    except Exception as e:
        logger.error(f"❌ Convenience functions test FAILED: {str(e)}")
    
    finally:
        try:
            os.unlink(tmp_file_path)
        except:
            pass

def main():
    """Run all tests"""
    logger.info("🧪 Starting Data Ingestion Pipeline Tests")
    logger.info("=" * 60)
    
    # Test different data formats
    test_data_ingestion_formats()
    
    # Test chunked processing
    test_chunked_processing()
    
    # Test convenience functions
    test_convenience_functions()
    
    logger.info("=" * 60)
    logger.info("🎉 All tests completed!")

if __name__ == "__main__":
    main()
