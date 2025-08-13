"""
Real-time Data Streaming Infrastructure
=====================================

Institutional-grade data processing pipeline with:
- Apache Flink/Spark Streaming for real-time processing
- Time-series database integration (InfluxDB/TimescaleDB)
- Data lake architecture (Apache Iceberg/Delta Lake)
- Change Data Capture (CDC) for order book updates
- Real-time data quality monitoring and validation
- Multi-provider market data ingestion (20+ providers)
- Tick-by-tick compression and storage optimization
- Historical data replay system for backtesting
- Market data normalization and enrichment
- Real-time index calculation engine
"""

from .flink.stream_processor import FlinkStreamProcessor
from .spark.spark_processor import SparkStreamProcessor
from .databases.time_series import TimeSeriesManager
from .datalake.lake_manager import DataLakeManager
from .cdc.cdc_processor import CDCProcessor
from .quality.data_quality import DataQualityMonitor
from .providers.provider_manager import MarketDataProviderManager
from .compression.compressor import TickDataCompressor
from .replay.replay_engine import HistoricalReplayEngine
from .normalization.normalizer import DataNormalizer
from .indexing.index_calculator import IndexCalculator

__all__ = [
    'FlinkStreamProcessor',
    'SparkStreamProcessor', 
    'TimeSeriesManager',
    'DataLakeManager',
    'CDCProcessor',
    'DataQualityMonitor',
    'MarketDataProviderManager',
    'TickDataCompressor',
    'HistoricalReplayEngine',
    'DataNormalizer',
    'IndexCalculator'
]
