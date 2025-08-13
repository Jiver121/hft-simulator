"""
Real-time Trading Infrastructure for HFT Simulator

This package provides the core infrastructure for real-time trading capabilities,
including market data feeds, broker integrations, and live order management.

Components:
- data_feeds: Real-time market data ingestion
- brokers: Broker API integrations
- order_management: Live order execution and management
- risk_management: Real-time risk controls
- stream_processing: High-performance data processing
- monitoring: System health and performance monitoring
"""

from .data_feeds import RealTimeDataFeed, WebSocketDataFeed
from .order_management import RealTimeOrderManager
from .risk_management import RealTimeRiskManager
from .stream_processing import StreamProcessor

__version__ = "1.0.0"
__all__ = [
    "RealTimeDataFeed",
    "WebSocketDataFeed", 
    "RealTimeOrderManager",
    "RealTimeRiskManager",
    "StreamProcessor"
]