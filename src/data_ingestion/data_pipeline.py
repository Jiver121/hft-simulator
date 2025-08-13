"""
Data Ingestion Pipeline

Comprehensive pipeline for processing, normalizing, and routing real-time market data
from multiple sources with quality validation and monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
from src.utils.logger import get_logger
from .websocket_client import MarketDataPoint, WebSocketConfig, BinanceWebSocketClient


@dataclass
class PipelineConfig:
    """Configuration for data ingestion pipeline"""
    # Data sources
    websocket_configs: List[WebSocketConfig] = field(default_factory=list)
    
    # Processing
    enable_normalization: bool = True
    enable_validation: bool = True
    enable_deduplication: bool = True
    
    # Buffering
    buffer_size: int = 10000
    batch_size: int = 100
    flush_interval: float = 1.0
    
    # Quality control
    max_latency_ms: float = 500.0
    min_data_rate: float = 1.0  # messages per second
    quality_check_interval: float = 30.0


class DataIngestionPipeline:
    """
    Main data ingestion pipeline that coordinates multiple data sources
    and provides normalized, validated market data to consumers
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger("DataIngestionPipeline")
        
        # Data sources
        self.websocket_clients = []
        self.data_queue = asyncio.Queue(maxsize=config.buffer_size)
        
        # Processing components
        self.normalizer = DataNormalizer()
        self.validator = DataValidator()
        self.deduplicator = DataDeduplicator()
        
        # State tracking
        self.running = False
        self.start_time = time.time()
        self.message_count = 0
        self.error_count = 0
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_validated': 0,
            'messages_dropped': 0,
            'sources_active': 0,
            'latency_samples': deque(maxlen=1000),
            'last_update': time.time()
        }
        
        # Subscribers
        self.subscribers = []
    
    async def initialize(self):
        """Initialize the pipeline and all data sources"""
        self.logger.info("Initializing data ingestion pipeline...")
        
        # Create WebSocket clients
        for ws_config in self.config.websocket_configs:
            client = BinanceWebSocketClient(ws_config)
            client.add_subscriber(self._handle_websocket_data)
            self.websocket_clients.append(client)
        
        self.logger.info(f"Initialized {len(self.websocket_clients)} WebSocket clients")
    
    async def start(self):
        """Start the data ingestion pipeline"""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.logger.info("Starting data ingestion pipeline...")
        
        # Start WebSocket connections
        connection_tasks = []
        for client in self.websocket_clients:
            task = asyncio.create_task(self._start_websocket_client(client))
            connection_tasks.append(task)
        
        # Start processing pipeline
        processing_task = asyncio.create_task(self._process_data_stream())
        
        # Start quality monitoring
        monitoring_task = asyncio.create_task(self._quality_monitor())
        
        # Wait for all tasks
        await asyncio.gather(
            *connection_tasks,
            processing_task,
            monitoring_task,
            return_exceptions=True
        )
    
    async def stop(self):
        """Stop the data ingestion pipeline"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("Stopping data ingestion pipeline...")
        
        # Disconnect all WebSocket clients
        for client in self.websocket_clients:
            await client.disconnect()
        
        self.logger.info("Data ingestion pipeline stopped")
    
    async def _start_websocket_client(self, client: BinanceWebSocketClient):
        """Start a WebSocket client and handle its data stream"""
        try:
            # Connect
            connected = await client.connect()
            if not connected:
                self.logger.error(f"Failed to connect WebSocket client")
                return
            
            self.stats['sources_active'] += 1
            
            # Start streaming
            async for data_point in client.start_streaming():
                if not self.running:
                    break
                
                # Add to processing queue
                try:
                    await self.data_queue.put(data_point)
                except asyncio.QueueFull:
                    self.logger.warning("Data queue full, dropping message")
                    self.stats['messages_dropped'] += 1
            
        except Exception as e:
            self.logger.error(f"WebSocket client error: {e}")
        finally:
            self.stats['sources_active'] -= 1
    
    async def _handle_websocket_data(self, data_point: MarketDataPoint):
        """Handle incoming WebSocket data"""
        try:
            await self.data_queue.put(data_point)
        except asyncio.QueueFull:
            self.logger.warning("Data queue full, dropping WebSocket message")
            self.stats['messages_dropped'] += 1
    
    async def _process_data_stream(self):
        """Main data processing loop"""
        batch = []
        last_flush = time.time()
        
        while self.running:
            try:
                # Get data with timeout
                try:
                    data_point = await asyncio.wait_for(
                        self.data_queue.get(),
                        timeout=self.config.flush_interval
                    )
                    batch.append(data_point)
                except asyncio.TimeoutError:
                    # Timeout reached, process batch if not empty
                    if batch:
                        await self._process_batch(batch)
                        batch.clear()
                        last_flush = time.time()
                    continue
                
                # Process batch when full or time interval reached
                if (len(batch) >= self.config.batch_size or 
                    time.time() - last_flush >= self.config.flush_interval):
                    await self._process_batch(batch)
                    batch.clear()
                    last_flush = time.time()
                
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                self.error_count += 1
    
    async def _process_batch(self, batch: List[MarketDataPoint]):
        """Process a batch of data points"""
        processed_data = []
        
        for data_point in batch:
            try:
                # Normalize data
                if self.config.enable_normalization:
                    data_point = await self.normalizer.normalize(data_point)
                
                # Validate data
                if self.config.enable_validation:
                    is_valid = await self.validator.validate(data_point)
                    if not is_valid:
                        self.stats['messages_dropped'] += 1
                        continue
                
                # Check for duplicates
                if self.config.enable_deduplication:
                    is_duplicate = await self.deduplicator.is_duplicate(data_point)
                    if is_duplicate:
                        continue
                
                processed_data.append(data_point)
                self.stats['messages_processed'] += 1
                
                # Track latency
                if data_point.latency_ms:
                    self.stats['latency_samples'].append(data_point.latency_ms)
                
            except Exception as e:
                self.logger.error(f"Error processing data point: {e}")
                self.error_count += 1
        
        # Notify subscribers
        if processed_data:
            await self._notify_subscribers(processed_data)
    
    async def _notify_subscribers(self, data_points: List[MarketDataPoint]):
        """Notify all subscribers of new data"""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(data_points)
                else:
                    subscriber(data_points)
            except Exception as e:
                self.logger.error(f"Subscriber error: {e}")
    
    async def _quality_monitor(self):
        """Monitor data quality and pipeline health"""
        while self.running:
            try:
                await asyncio.sleep(self.config.quality_check_interval)
                
                # Calculate metrics
                uptime = time.time() - self.start_time
                message_rate = self.stats['messages_processed'] / max(uptime, 1)
                
                # Check data rate
                if message_rate < self.config.min_data_rate:
                    self.logger.warning(f"Low data rate: {message_rate:.1f} msg/s")
                
                # Check latency
                if self.stats['latency_samples']:
                    avg_latency = sum(self.stats['latency_samples']) / len(self.stats['latency_samples'])
                    if avg_latency > self.config.max_latency_ms:
                        self.logger.warning(f"High latency: {avg_latency:.1f}ms")
                
                # Log status
                self.logger.info(f"Pipeline status: {message_rate:.1f} msg/s, "
                               f"{self.stats['sources_active']} sources, "
                               f"{self.data_queue.qsize()} queued")
                
            except Exception as e:
                self.logger.error(f"Quality monitor error: {e}")
    
    def add_subscriber(self, callback: Callable[[List[MarketDataPoint]], None]):
        """Add subscriber for processed data"""
        self.subscribers.append(callback)
        self.logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    def remove_subscriber(self, callback: Callable[[List[MarketDataPoint]], None]):
        """Remove subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            self.logger.info(f"Removed subscriber, total: {len(self.subscribers)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        uptime = time.time() - self.start_time
        
        return {
            'pipeline': {
                'running': self.running,
                'uptime_seconds': uptime,
                'sources_active': self.stats['sources_active'],
                'queue_size': self.data_queue.qsize(),
                'subscribers': len(self.subscribers)
            },
            'processing': {
                'messages_processed': self.stats['messages_processed'],
                'messages_dropped': self.stats['messages_dropped'],
                'error_count': self.error_count,
                'rate_per_second': self.stats['messages_processed'] / max(uptime, 1)
            },
            'quality': {
                'avg_latency_ms': sum(self.stats['latency_samples']) / len(self.stats['latency_samples']) 
                                if self.stats['latency_samples'] else 0,
                'latency_samples': len(self.stats['latency_samples'])
            }
        }


class DataNormalizer:
    """Normalize data from different sources into consistent format"""
    
    def __init__(self):
        self.logger = get_logger("DataNormalizer")
    
    async def normalize(self, data_point: MarketDataPoint) -> MarketDataPoint:
        """Normalize a data point"""
        # Ensure timestamp is timezone-aware
        if data_point.timestamp.tz is None:
            data_point.timestamp = data_point.timestamp.tz_localize('UTC')
        
        # Normalize symbol format
        data_point.symbol = data_point.symbol.upper().replace('/', '')
        
        # Ensure numeric values are properly typed
        if data_point.price and isinstance(data_point.price, str):
            data_point.price = float(data_point.price)
        
        if data_point.quantity and isinstance(data_point.quantity, str):
            data_point.quantity = float(data_point.quantity)
        
        return data_point


class DataValidator:
    """Validate data quality and consistency"""
    
    def __init__(self):
        self.logger = get_logger("DataValidator")
    
    async def validate(self, data_point: MarketDataPoint) -> bool:
        """Validate a data point"""
        try:
            # Check required fields
            if not data_point.symbol or not data_point.timestamp:
                return False
            
            # Validate prices
            if data_point.price is not None:
                if data_point.price <= 0 or data_point.price > 1000000:
                    return False
            
            # Validate quantities
            if data_point.quantity is not None:
                if data_point.quantity < 0:
                    return False
            
            # Validate bid/ask spread
            if data_point.best_bid and data_point.best_ask:
                if data_point.best_bid >= data_point.best_ask:
                    return False
            
            # Validate timestamp (not too old or in future)
            now = pd.Timestamp.now(tz='UTC')
            if abs((now - data_point.timestamp).total_seconds()) > 300:  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False


class DataDeduplicator:
    """Remove duplicate data points"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.recent_data = deque(maxlen=window_size)
        self.logger = get_logger("DataDeduplicator")
    
    async def is_duplicate(self, data_point: MarketDataPoint) -> bool:
        """Check if data point is a duplicate"""
        # Create signature for comparison
        signature = (
            data_point.symbol,
            data_point.event_type,
            data_point.price,
            data_point.quantity,
            data_point.trade_id
        )
        
        # Check if signature exists in recent data
        if signature in self.recent_data:
            return True
        
        # Add to recent data
        self.recent_data.append(signature)
        return False
