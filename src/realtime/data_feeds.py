"""
Real-Time Market Data Feeds for HFT Simulator

This module provides real-time market data ingestion capabilities supporting
multiple data sources including WebSocket feeds, REST APIs, and FIX protocol.

Key Features:
- WebSocket connections for low-latency streaming
- Multiple data source support (exchanges, vendors)
- Automatic failover and reconnection
- Data normalization and validation
- Rate limiting and backpressure handling
- Comprehensive error handling and logging

Supported Data Sources:
- Exchange direct feeds (simulated)
- Financial data providers (IEX, Alpha Vantage, etc.)
- Cryptocurrency exchanges (Binance, Coinbase, etc.)
- Custom WebSocket endpoints
"""

import asyncio
import json
import websockets
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import logging
from urllib.parse import urljoin
import ssl
import time

from src.utils.logger import get_logger
from src.utils.constants import OrderSide
from src.engine.order_types import MarketDataPoint
from src.engine.market_data import BookSnapshot


@dataclass
class DataFeedConfig:
    """Configuration for real-time data feeds"""
    
    # Connection settings
    url: str
    symbols: List[str]
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    
    # Connection parameters
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 30.0
    connection_timeout: float = 10.0
    
    # Data processing
    buffer_size: int = 10000
    enable_compression: bool = True
    validate_data: bool = True
    
    # Rate limiting
    max_messages_per_second: int = 1000
    backpressure_threshold: int = 5000
    
    # Failover
    backup_urls: List[str] = field(default_factory=list)
    enable_failover: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_raw_messages: bool = False


@dataclass
class MarketDataMessage:
    """Standardized market data message"""
    
    symbol: str
    timestamp: pd.Timestamp
    message_type: str  # 'quote', 'trade', 'book_update', 'heartbeat'
    
    # Price data
    price: Optional[float] = None
    volume: Optional[int] = None
    
    # Order book data
    bid_price: Optional[float] = None
    bid_volume: Optional[int] = None
    ask_price: Optional[float] = None
    ask_volume: Optional[int] = None
    
    # Trade data
    trade_id: Optional[str] = None
    side: Optional[OrderSide] = None
    
    # Book depth (for full book updates)
    bids: List[tuple] = field(default_factory=list)  # [(price, volume), ...]
    asks: List[tuple] = field(default_factory=list)  # [(price, volume), ...]
    
    # Metadata
    sequence_number: Optional[int] = None
    source: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_market_data_point(self) -> MarketDataPoint:
        """Convert to MarketDataPoint for compatibility"""
        return MarketDataPoint(
            timestamp=self.timestamp,
            price=self.price or 0.0,
            volume=self.volume or 0,
            best_bid=self.bid_price,
            best_ask=self.ask_price,
            bid_volume=self.bid_volume,
            ask_volume=self.ask_volume,
            metadata={
                'symbol': self.symbol,
                'message_type': self.message_type,
                'source': self.source,
                'sequence_number': self.sequence_number
            }
        )


class RealTimeDataFeed(ABC):
    """
    Abstract base class for real-time market data feeds
    
    This class defines the interface that all data feed implementations
    must follow, ensuring consistency across different data sources.
    """
    
    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Connection state
        self.connected = False
        self.reconnect_count = 0
        self.last_heartbeat = None
        
        # Data processing
        self.message_buffer = deque(maxlen=config.buffer_size)
        self.subscribers: List[Callable[[MarketDataMessage], None]] = []
        self.message_count = 0
        self.last_message_time = time.time()
        
        # Rate limiting
        self.message_timestamps = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_errors': 0,
            'data_errors': 0,
            'reconnections': 0,
            'uptime_start': time.time()
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to market data for symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data for symbols"""
        pass
    
    @abstractmethod
    async def start_streaming(self) -> AsyncIterator[MarketDataMessage]:
        """Start streaming market data"""
        pass
    
    def add_subscriber(self, callback: Callable[[MarketDataMessage], None]) -> None:
        """Add callback for market data updates"""
        self.subscribers.append(callback)
        self.logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    def remove_subscriber(self, callback: Callable[[MarketDataMessage], None]) -> None:
        """Remove callback"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            self.logger.info(f"Removed subscriber, total: {len(self.subscribers)}")
    
    async def _notify_subscribers(self, message: MarketDataMessage) -> None:
        """Notify all subscribers of new market data"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback: {e}")
    
    def _validate_message(self, message: MarketDataMessage) -> bool:
        """Validate market data message"""
        if not self.config.validate_data:
            return True
        
        # Basic validation
        if not message.symbol or not message.timestamp:
            return False
        
        # Price validation
        if message.price is not None and (message.price <= 0 or not np.isfinite(message.price)):
            return False
        
        # Volume validation
        if message.volume is not None and message.volume < 0:
            return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        self.message_timestamps.append(now)
        
        # Count messages in last second
        recent_messages = sum(1 for ts in self.message_timestamps if now - ts <= 1.0)
        
        return recent_messages <= self.config.max_messages_per_second
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feed statistics"""
        now = time.time()
        uptime = now - self.stats['uptime_start']
        
        return {
            **self.stats,
            'connected': self.connected,
            'uptime_seconds': uptime,
            'messages_per_second': self.stats['messages_received'] / max(uptime, 1),
            'buffer_size': len(self.message_buffer),
            'subscriber_count': len(self.subscribers),
            'reconnect_count': self.reconnect_count
        }


class WebSocketDataFeed(RealTimeDataFeed):
    """
    WebSocket-based real-time data feed
    
    Supports various WebSocket-based market data sources with automatic
    reconnection, heartbeat monitoring, and comprehensive error handling.
    """
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.streaming_task: Optional[asyncio.Task] = None
        
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            self.logger.info(f"Connecting to WebSocket: {self.config.url}")
            
            # Setup SSL context if needed
            ssl_context = None
            if self.config.url.startswith('wss://'):
                ssl_context = ssl.create_default_context()
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    ssl=ssl_context,
                    compression='deflate' if self.config.enable_compression else None,
                    ping_interval=self.config.heartbeat_interval,
                    ping_timeout=self.config.connection_timeout
                ),
                timeout=self.config.connection_timeout
            )
            
            self.connected = True
            self.last_heartbeat = time.time()
            self.logger.info("WebSocket connection established")
            
            # Start heartbeat monitoring
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.stats['connection_errors'] += 1
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection"""
        self.connected = False
        
        # Cancel tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.streaming_task:
            self.streaming_task.cancel()
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.logger.info("WebSocket connection closed")
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to market data for symbols"""
        if not self.connected or not self.websocket:
            self.logger.error("Not connected - cannot subscribe")
            return False
        
        try:
            # Generic subscription message format
            # This would be customized for specific exchanges
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            self.logger.info(f"Subscribed to symbols: {symbols}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from market data for symbols"""
        if not self.connected or not self.websocket:
            return False
        
        try:
            unsubscribe_message = {
                "method": "UNSUBSCRIBE", 
                "params": [f"{symbol.lower()}@ticker" for symbol in symbols],
                "id": int(time.time())
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            self.logger.info(f"Unsubscribed from symbols: {symbols}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe: {e}")
            return False
    
    async def start_streaming(self) -> AsyncIterator[MarketDataMessage]:
        """Start streaming market data"""
        if not self.connected:
            raise RuntimeError("Not connected - call connect() first")
        
        self.logger.info("Starting market data stream")
        
        try:
            async for raw_message in self.websocket:
                try:
                    # Parse JSON message
                    data = json.loads(raw_message)
                    
                    # Log raw message if configured
                    if self.config.log_raw_messages:
                        self.logger.debug(f"Raw message: {raw_message}")
                    
                    # Convert to standardized format
                    message = self._parse_message(data)
                    
                    if message and self._validate_message(message):
                        # Check rate limits
                        if not self._check_rate_limit():
                            self.logger.warning("Rate limit exceeded, dropping message")
                            continue
                        
                        # Update statistics
                        self.stats['messages_received'] += 1
                        self.stats['messages_processed'] += 1
                        self.last_message_time = time.time()
                        
                        # Add to buffer
                        self.message_buffer.append(message)
                        
                        # Notify subscribers
                        await self._notify_subscribers(message)
                        
                        yield message
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON: {e}")
                    self.stats['data_errors'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    self.stats['data_errors'] += 1
        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.connected = False
            
            # Attempt reconnection if configured
            if self.config.enable_failover and self.reconnect_count < self.config.max_reconnect_attempts:
                await self._attempt_reconnection()
        
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            self.connected = False
    
    def _parse_message(self, data: Dict[str, Any]) -> Optional[MarketDataMessage]:
        """
        Parse raw WebSocket message into standardized format
        
        This is a generic parser - would be customized for specific exchanges
        """
        try:
            # Handle different message types
            if 'stream' in data and 'data' in data:
                # Binance combined stream message (/stream?streams=...)
                stream_data = data['data']
                symbol = stream_data.get('s', '').upper()
                
                if '@ticker' in data['stream']:
                    return MarketDataMessage(
                        symbol=symbol,
                        timestamp=pd.Timestamp.now(),
                        message_type='quote',
                        price=float(stream_data.get('c', 0)),  # Close price
                        volume=int(float(stream_data.get('v', 0))),  # Volume
                        bid_price=float(stream_data.get('b', 0)),  # Best bid
                        ask_price=float(stream_data.get('a', 0)),  # Best ask
                        source='websocket'
                    )
            # Binance single stream message (/ws) for ticker
            elif isinstance(data, dict) and all(k in data for k in ['e', 'E', 's', 'c']):
                return MarketDataMessage(
                    symbol=str(data.get('s', '')).upper(),
                    timestamp=pd.Timestamp.now(),
                    message_type='quote',
                    price=float(data.get('c', 0)),
                    volume=int(float(data.get('v', 0) or 0)),
                    bid_price=float(data.get('b', 0) or 0),
                    ask_price=float(data.get('a', 0) or 0),
                    source='websocket'
                )
            
            elif 'symbol' in data:
                # Generic format
                return MarketDataMessage(
                    symbol=data['symbol'].upper(),
                    timestamp=pd.Timestamp.now(),
                    message_type=data.get('type', 'unknown'),
                    price=data.get('price'),
                    volume=data.get('volume'),
                    bid_price=data.get('bid'),
                    ask_price=data.get('ask'),
                    source='websocket'
                )
            
            # Handle heartbeat/ping messages
            elif 'ping' in data:
                return MarketDataMessage(
                    symbol='HEARTBEAT',
                    timestamp=pd.Timestamp.now(),
                    message_type='heartbeat',
                    source='websocket'
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse message: {e}")
            return None
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health via heartbeat"""
        while self.connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.websocket:
                    # Check if websocket is still open
                    try:
                        # Try to access state - if connection is closed this will fail
                        websocket_open = getattr(self.websocket, 'state', None) is not None
                        
                        if websocket_open:
                            # Send ping if supported
                            try:
                                await self.websocket.ping()
                                self.last_heartbeat = time.time()
                            except Exception as e:
                                self.logger.warning(f"Heartbeat failed: {e}")
                                break
                        else:
                            break
                    except Exception as e:
                        # Websocket connection may be closed
                        self.logger.debug(f"Websocket state check failed: {e}")
                        break
                else:
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                break
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect with exponential backoff"""
        self.reconnect_count += 1
        self.stats['reconnections'] += 1
        
        delay = min(self.config.reconnect_delay * (2 ** self.reconnect_count), 60)
        self.logger.info(f"Attempting reconnection {self.reconnect_count}/{self.config.max_reconnect_attempts} in {delay}s")
        
        await asyncio.sleep(delay)
        
        if await self.connect():
            self.reconnect_count = 0
            # Re-subscribe to symbols
            await self.subscribe(self.config.symbols)


class MockDataFeed(RealTimeDataFeed):
    """
    Mock data feed for testing and development
    
    Generates realistic market data for testing purposes without
    requiring external connections.
    """
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.running = False
        self.price_state = {symbol: 100.0 for symbol in config.symbols}
        
    async def connect(self) -> bool:
        """Mock connection - always succeeds"""
        self.connected = True
        self.logger.info("Mock data feed connected")
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnection"""
        self.connected = False
        self.running = False
        self.logger.info("Mock data feed disconnected")
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Mock subscription"""
        self.logger.info(f"Mock subscribed to: {symbols}")
        return True
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Mock unsubscription"""
        self.logger.info(f"Mock unsubscribed from: {symbols}")
        return True
    
    async def start_streaming(self) -> AsyncIterator[MarketDataMessage]:
        """Generate mock market data stream"""
        self.running = True
        self.logger.info("Starting mock data stream")
        
        while self.running and self.connected:
            for symbol in self.config.symbols:
                # Generate realistic price movement
                current_price = self.price_state[symbol]
                price_change = np.random.normal(0, 0.001) * current_price
                new_price = max(current_price + price_change, 0.01)
                self.price_state[symbol] = new_price
                
                # Generate bid/ask spread
                spread = np.random.uniform(0.01, 0.05)
                bid_price = new_price - spread/2
                ask_price = new_price + spread/2
                
                message = MarketDataMessage(
                    symbol=symbol,
                    timestamp=pd.Timestamp.now(),
                    message_type='quote',
                    price=new_price,
                    volume=np.random.randint(100, 1000),
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_volume=np.random.randint(500, 2000),
                    ask_volume=np.random.randint(500, 2000),
                    source='mock'
                )
                
                self.stats['messages_received'] += 1
                self.stats['messages_processed'] += 1
                
                await self._notify_subscribers(message)
                yield message
            
            # Control update frequency
            await asyncio.sleep(0.1)  # 10 updates per second


# Factory function for creating data feeds
def create_data_feed(feed_type: str, config: DataFeedConfig) -> RealTimeDataFeed:
    """
    Factory function to create appropriate data feed instance
    
    Args:
        feed_type: Type of feed ('websocket', 'mock', etc.)
        config: Feed configuration
        
    Returns:
        Configured data feed instance
    """
    if feed_type.lower() == 'websocket':
        return WebSocketDataFeed(config)
    elif feed_type.lower() == 'mock':
        return MockDataFeed(config)
    else:
        raise ValueError(f"Unknown feed type: {feed_type}")