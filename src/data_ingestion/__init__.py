"""
Data Ingestion Module

Comprehensive real-time market data ingestion pipeline supporting:
- WebSocket connections to multiple exchanges
- Data normalization and validation
- Order book depth tracking
- Trade stream processing  
- Reconnection logic and error handling
- Data quality monitoring and latency metrics
"""

from .websocket_client import BinanceWebSocketClient, WebSocketDataProcessor
from .data_pipeline import DataIngestionPipeline
from .quality_monitor import DataQualityMonitor

__all__ = [
    'BinanceWebSocketClient',
    'WebSocketDataProcessor', 
    'DataIngestionPipeline',
    'DataQualityMonitor'
]
