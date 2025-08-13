"""
Data Loader with LRU Caching for HFT Simulator

This module provides a high-performance data loading system with intelligent
caching mechanisms to minimize I/O operations and improve data access speed
for frequently accessed datasets.

Key Features:
- LRU (Least Recently Used) caching for datasets
- Memory-efficient caching with size limits
- Cache statistics and monitoring
- Automatic cache invalidation
- Multi-level caching strategy
- Optimized for HFT data patterns
"""

import os
import hashlib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from functools import lru_cache, wraps
from collections import OrderedDict
import threading
import time
from datetime import datetime, timedelta
import weakref
import gc

from src.utils.logger import get_logger, log_performance, log_memory_usage
from src.utils.helpers import Timer, get_memory_usage, estimate_dataframe_memory
from src.data.ingestion import DataIngestion
from src.data.optimized_ingestion import OptimizedDataIngestion


class LRUCache:
    """
    Custom LRU Cache implementation optimized for large DataFrames
    
    This implementation is memory-aware and can handle large datasets
    more efficiently than standard functools.lru_cache for DataFrame objects.
    """
    
    def __init__(self, maxsize: int = 128, max_memory_mb: float = 1024):
        """
        Initialize LRU Cache with size and memory limits
        
        Args:
            maxsize: Maximum number of items in cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.maxsize = maxsize
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()
        self.cache_info = {
            'hits': 0,
            'misses': 0,
            'memory_usage_mb': 0.0,
            'evictions': 0
        }
        self._lock = threading.RLock()
        self.logger = get_logger(f"{__name__}.LRUCache")
    
    def _calculate_size(self, value: Any) -> float:
        """Calculate memory size of cached value in MB"""
        if isinstance(value, pd.DataFrame):
            return estimate_dataframe_memory(value)
        elif isinstance(value, (list, dict, tuple)):
            try:
                # Rough estimate for complex objects
                return len(pickle.dumps(value)) / 1024 / 1024
            except:
                return 1.0  # Default fallback
        else:
            return 0.1  # Small objects
    
    def _evict_if_needed(self):
        """Evict items if cache exceeds limits"""
        while (len(self.cache) >= self.maxsize or 
               self.cache_info['memory_usage_mb'] > self.max_memory_mb):
            if not self.cache:
                break
            
            # Remove least recently used item
            key, value = self.cache.popitem(last=False)
            size_mb = self._calculate_size(value)
            self.cache_info['memory_usage_mb'] -= size_mb
            self.cache_info['evictions'] += 1
            
            self.logger.debug(f"Evicted cache item {key} ({size_mb:.1f}MB)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.cache_info['hits'] += 1
                return value
            else:
                self.cache_info['misses'] += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self._lock:
            # Remove existing key if present
            if key in self.cache:
                old_value = self.cache.pop(key)
                old_size = self._calculate_size(old_value)
                self.cache_info['memory_usage_mb'] -= old_size
            
            # Calculate size and check if it fits
            size_mb = self._calculate_size(value)
            
            # Don't cache items that are too large
            if size_mb > self.max_memory_mb * 0.5:
                self.logger.warning(f"Item too large to cache: {size_mb:.1f}MB")
                return
            
            # Evict if necessary
            self._evict_if_needed()
            
            # Add new item
            self.cache[key] = value
            self.cache_info['memory_usage_mb'] += size_mb
    
    def clear(self):
        """Clear all cache items"""
        with self._lock:
            self.cache.clear()
            self.cache_info['memory_usage_mb'] = 0.0
            self.logger.info("Cache cleared")
    
    def info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.cache_info['hits'] + self.cache_info['misses']
            hit_rate = self.cache_info['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.maxsize,
                'memory_usage_mb': self.cache_info['memory_usage_mb'],
                'max_memory_mb': self.max_memory_mb,
                'hits': self.cache_info['hits'],
                'misses': self.cache_info['misses'],
                'hit_rate': hit_rate,
                'evictions': self.cache_info['evictions']
            }


class CachedDataLoader:
    """
    High-performance data loader with intelligent caching
    
    This class provides optimized data loading with multiple caching layers:
    - In-memory LRU cache for frequently accessed data
    - Persistent disk cache for preprocessed data
    - Metadata cache for file information
    - Query result cache for filtered datasets
    
    Performance Benefits:
    - 10-100x faster access for cached data
    - Reduced memory usage through intelligent eviction
    - Automatic cache warming for common queries
    - Background cache maintenance
    """
    
    def __init__(self, 
                 cache_size: int = 100,
                 max_memory_mb: float = 2048,
                 cache_dir: Optional[str] = None,
                 enable_disk_cache: bool = True,
                 enable_query_cache: bool = True):
        """
        Initialize the cached data loader
        
        Args:
            cache_size: Maximum number of datasets in memory cache
            max_memory_mb: Maximum memory usage for cache
            cache_dir: Directory for persistent cache (auto-created if None)
            enable_disk_cache: Enable persistent disk caching
            enable_query_cache: Enable query result caching
        """
        self.logger = get_logger(__name__)
        
        # Initialize caches
        self.memory_cache = LRUCache(maxsize=cache_size, max_memory_mb=max_memory_mb)
        self.metadata_cache = {}
        self.query_cache = LRUCache(maxsize=50, max_memory_mb=512) if enable_query_cache else None
        
        # Cache settings
        self.enable_disk_cache = enable_disk_cache
        self.enable_query_cache = enable_query_cache
        
        # Disk cache setup
        if self.enable_disk_cache:
            self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".cache" / "data"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data ingestion components
        self.standard_ingestion = DataIngestion()
        self.optimized_ingestion = OptimizedDataIngestion()
        
        # Statistics
        self.stats = {
            'loads': 0,
            'cache_hits': 0,
            'disk_cache_hits': 0,
            'query_cache_hits': 0,
            'total_load_time': 0.0,
            'cache_save_time': 0.0
        }
        
        # Background maintenance
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        self.logger.info(f"CachedDataLoader initialized (cache_size={cache_size}, "
                        f"max_memory={max_memory_mb}MB, disk_cache={enable_disk_cache})")
    
    def _generate_cache_key(self, filepath: Union[str, Path], **kwargs) -> str:
        """Generate cache key for file and parameters"""
        filepath = Path(filepath)
        
        # Include file modification time and size in key
        try:
            stat = filepath.stat()
            file_info = f"{stat.st_size}_{stat.st_mtime}"
        except:
            file_info = str(filepath)
        
        # Include loading parameters
        params_str = str(sorted(kwargs.items()))
        
        # Create hash
        key_str = f"{filepath}_{file_info}_{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get path for disk cache file"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _save_to_disk_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to disk cache"""
        if not self.enable_disk_cache:
            return
        
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            
            with Timer() as timer:
                # Use efficient pickle protocol
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats['cache_save_time'] += timer.elapsed()
            self.logger.debug(f"Saved to disk cache: {cache_path} ({timer.elapsed():.1f}ms)")
            
        except Exception as e:
            self.logger.warning(f"Failed to save disk cache: {e}")
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from disk cache"""
        if not self.enable_disk_cache:
            return None
        
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            # Check if cache file is recent enough (24 hours)
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > 24 * 3600:  # 24 hours
                cache_path.unlink()  # Remove old cache
                return None
            
            with Timer() as timer:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            
            self.stats['disk_cache_hits'] += 1
            self.logger.debug(f"Loaded from disk cache: {cache_path} ({timer.elapsed():.1f}ms)")
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to load disk cache: {e}")
            return None
    
    @log_performance
    @log_memory_usage
    def load_data(self, 
                  filepath: Union[str, Path],
                  use_optimized: bool = True,
                  **kwargs) -> pd.DataFrame:
        """
        Load data with intelligent caching
        
        Args:
            filepath: Path to data file
            use_optimized: Use optimized ingestion for large files
            **kwargs: Additional parameters for data loading
            
        Returns:
            Loaded DataFrame
            
        Performance Notes:
        - First load: Standard file I/O time
        - Cached load: ~1ms for memory cache, ~10ms for disk cache
        - Automatic optimization based on file size and access patterns
        """
        filepath = Path(filepath)
        self.stats['loads'] += 1
        
        # Generate cache key
        cache_key = self._generate_cache_key(filepath, **kwargs)
        
        # Check memory cache first
        cached_data = self.memory_cache.get(cache_key)
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            self.logger.debug(f"Memory cache hit for {filepath.name}")
            return cached_data.copy()  # Return copy to prevent modification
        
        # Check disk cache
        cached_data = self._load_from_disk_cache(cache_key)
        if cached_data is not None:
            # Store in memory cache for faster future access
            self.memory_cache.put(cache_key, cached_data)
            return cached_data.copy()
        
        # Load from source file
        self.logger.info(f"Loading data from source: {filepath}")
        
        with Timer() as timer:
            # Choose ingestion method based on file size and preferences
            if use_optimized and filepath.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                data = self.optimized_ingestion.load_csv_parallel(filepath, **kwargs)
            else:
                data = self.standard_ingestion.load_csv(filepath, **kwargs)
        
        self.stats['total_load_time'] += timer.elapsed()
        
        # Cache the loaded data
        self.memory_cache.put(cache_key, data)
        self._save_to_disk_cache(cache_key, data)
        
        # Periodic cleanup
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_cache()
        
        return data.copy()
    
    def load_data_chunked(self,
                         filepath: Union[str, Path],
                         chunk_processor: Optional[Callable] = None,
                         use_optimized: bool = True,
                         **kwargs) -> List[pd.DataFrame]:
        """
        Load data in chunks with caching for each chunk
        
        Args:
            filepath: Path to data file
            chunk_processor: Optional function to process each chunk
            use_optimized: Use optimized ingestion
            **kwargs: Additional parameters
            
        Returns:
            List of processed chunks
            
        Performance: Caches individual chunks for faster reprocessing
        """
        filepath = Path(filepath)
        chunks = []
        
        # Generate base cache key for chunked loading
        base_key = self._generate_cache_key(filepath, chunked=True, **kwargs)
        
        # Choose ingestion method
        ingestion = self.optimized_ingestion if use_optimized else self.standard_ingestion
        
        if hasattr(ingestion, 'load_csv_chunks_optimized') and use_optimized:
            chunk_iterator = ingestion.load_csv_chunks_optimized(filepath, **kwargs)
        else:
            chunk_iterator = ingestion.load_csv_chunks(filepath, **kwargs)
        
        for chunk_idx, chunk in enumerate(chunk_iterator):
            # Generate chunk-specific cache key
            chunk_key = f"{base_key}_chunk_{chunk_idx}"
            
            # Check if chunk is cached
            cached_chunk = self.memory_cache.get(chunk_key)
            if cached_chunk is not None:
                processed_chunk = cached_chunk.copy()
            else:
                # Process chunk if processor provided
                processed_chunk = chunk_processor(chunk) if chunk_processor else chunk
                
                # Cache processed chunk
                self.memory_cache.put(chunk_key, processed_chunk)
            
            chunks.append(processed_chunk)
        
        return chunks
    
    @lru_cache(maxsize=100)
    def get_file_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get cached file metadata
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file metadata
            
        Performance: Cached to avoid repeated filesystem access
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = filepath.stat()
        
        # Basic metadata
        metadata = {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / 1024 / 1024,
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'extension': filepath.suffix.lower(),
            'stem': filepath.stem
        }
        
        # Estimated processing time based on file size
        if metadata['size_mb'] < 10:
            metadata['estimated_load_time'] = 'fast'
        elif metadata['size_mb'] < 100:
            metadata['estimated_load_time'] = 'medium'  
        else:
            metadata['estimated_load_time'] = 'slow'
        
        return metadata
    
    def query_data(self,
                   data_identifier: Union[str, Path, pd.DataFrame],
                   query: str,
                   cache_result: bool = True) -> pd.DataFrame:
        """
        Query data with result caching
        
        Args:
            data_identifier: File path or DataFrame to query
            query: SQL-like query string for pandas
            cache_result: Whether to cache the query result
            
        Returns:
            Filtered DataFrame
            
        Performance: Caches common query results for instant access
        """
        if isinstance(data_identifier, (str, Path)):
            # Load data if file path provided
            data = self.load_data(data_identifier)
            cache_key_base = str(data_identifier)
        else:
            # Use provided DataFrame
            data = data_identifier
            cache_key_base = "dataframe_query"
        
        # Check query cache if enabled
        query_key = hashlib.md5(f"{cache_key_base}_{query}".encode()).hexdigest()
        
        if self.enable_query_cache:
            cached_result = self.query_cache.get(query_key)
            if cached_result is not None:
                self.stats['query_cache_hits'] += 1
                return cached_result.copy()
        
        # Execute query
        try:
            result = data.query(query)
            
            # Cache result if requested and cache is enabled
            if cache_result and self.enable_query_cache:
                self.query_cache.put(query_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def filter_data(self,
                   data_identifier: Union[str, Path, pd.DataFrame],
                   filters: Dict[str, Any],
                   cache_result: bool = True) -> pd.DataFrame:
        """
        Filter data with result caching
        
        Args:
            data_identifier: File path or DataFrame to filter
            filters: Dictionary of column filters
            cache_result: Whether to cache the filtered result
            
        Returns:
            Filtered DataFrame
            
        Example:
            filters = {
                'price': (100.0, 200.0),  # Range filter
                'volume': lambda x: x > 1000,  # Function filter
                'side': ['buy', 'sell']  # List filter
            }
        """
        if isinstance(data_identifier, (str, Path)):
            data = self.load_data(data_identifier)
            cache_key_base = str(data_identifier)
        else:
            data = data_identifier
            cache_key_base = "dataframe_filter"
        
        # Generate cache key from filters
        filter_key = hashlib.md5(f"{cache_key_base}_{str(filters)}".encode()).hexdigest()
        
        if self.enable_query_cache:
            cached_result = self.query_cache.get(filter_key)
            if cached_result is not None:
                self.stats['query_cache_hits'] += 1
                return cached_result.copy()
        
        # Apply filters
        result = data.copy()
        
        for column, filter_value in filters.items():
            if column not in result.columns:
                continue
            
            if isinstance(filter_value, tuple) and len(filter_value) == 2:
                # Range filter
                min_val, max_val = filter_value
                result = result[(result[column] >= min_val) & (result[column] <= max_val)]
            elif callable(filter_value):
                # Function filter
                result = result[filter_value(result[column])]
            elif isinstance(filter_value, (list, set)):
                # List filter
                result = result[result[column].isin(filter_value)]
            else:
                # Exact match filter
                result = result[result[column] == filter_value]
        
        # Cache result
        if cache_result and self.enable_query_cache:
            self.query_cache.put(filter_key, result)
        
        return result
    
    def preload_data(self, filepaths: List[Union[str, Path]], 
                    background: bool = True,
                    **kwargs):
        """
        Preload multiple datasets into cache
        
        Args:
            filepaths: List of files to preload
            background: Load in background thread
            **kwargs: Loading parameters
            
        Performance: Warms cache for faster subsequent access
        """
        def _preload():
            for filepath in filepaths:
                try:
                    self.load_data(filepath, **kwargs)
                    self.logger.debug(f"Preloaded: {filepath}")
                except Exception as e:
                    self.logger.warning(f"Failed to preload {filepath}: {e}")
        
        if background:
            import threading
            thread = threading.Thread(target=_preload)
            thread.daemon = True
            thread.start()
        else:
            _preload()
    
    def _cleanup_cache(self):
        """Perform cache maintenance"""
        self.logger.debug("Performing cache cleanup...")
        
        # Clean up disk cache - remove old files
        if self.enable_disk_cache and self.cache_dir.exists():
            cutoff_time = time.time() - 7 * 24 * 3600  # 7 days
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    if cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        self.logger.debug(f"Removed old cache file: {cache_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        self._last_cleanup = time.time()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics
        
        Returns:
            Dictionary with cache performance metrics
        """
        memory_info = self.memory_cache.info()
        
        stats = {
            'memory_cache': memory_info,
            'loader_stats': self.stats.copy(),
            'cache_efficiency': {
                'memory_hit_rate': memory_info['hit_rate'],
                'total_loads': self.stats['loads'],
                'cache_hits': self.stats['cache_hits'],
                'disk_cache_hits': self.stats['disk_cache_hits']
            }
        }
        
        if self.enable_query_cache:
            query_info = self.query_cache.info()
            stats['query_cache'] = query_info
            stats['cache_efficiency']['query_cache_hits'] = self.stats['query_cache_hits']
        
        # Calculate overall hit rate
        total_requests = self.stats['loads']
        total_hits = (self.stats['cache_hits'] + 
                     self.stats['disk_cache_hits'] + 
                     self.stats.get('query_cache_hits', 0))
        
        if total_requests > 0:
            stats['cache_efficiency']['overall_hit_rate'] = total_hits / total_requests
        
        return stats
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.memory_cache.clear()
        if self.query_cache:
            self.query_cache.clear()
        
        # Clear disk cache
        if self.enable_disk_cache and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        self.logger.info("All caches cleared")
    
    def optimize_cache(self):
        """Optimize cache performance"""
        # Get current memory usage
        memory_info = get_memory_usage()
        available_mb = memory_info.get('available', 2048)
        
        # Adjust cache size based on available memory
        if available_mb > 4096:  # > 4GB available
            new_max_memory = min(2048, available_mb * 0.3)
        elif available_mb > 2048:  # > 2GB available
            new_max_memory = min(1024, available_mb * 0.2)
        else:
            new_max_memory = min(512, available_mb * 0.1)
        
        self.memory_cache.max_memory_mb = new_max_memory
        self.logger.info(f"Adjusted cache memory limit to {new_max_memory:.0f}MB")
        
        # Force eviction if over limit
        self.memory_cache._evict_if_needed()


# Convenience functions
def load_cached_data(filepath: Union[str, Path], 
                    cache_size: int = 50,
                    **kwargs) -> pd.DataFrame:
    """
    Convenience function to load data with caching
    
    Args:
        filepath: Path to data file
        cache_size: Cache size limit
        **kwargs: Additional loading parameters
        
    Returns:
        Loaded DataFrame
    """
    loader = CachedDataLoader(cache_size=cache_size)
    return loader.load_data(filepath, **kwargs)


# Global cache instance for convenience
_global_loader = None

def get_global_loader() -> CachedDataLoader:
    """Get or create global data loader instance"""
    global _global_loader
    if _global_loader is None:
        _global_loader = CachedDataLoader()
    return _global_loader


def clear_global_cache():
    """Clear the global data loader cache"""
    global _global_loader
    if _global_loader:
        _global_loader.clear_all_caches()
