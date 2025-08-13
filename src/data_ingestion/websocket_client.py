#!/usr/bin/env python3
"""
Binance WebSocket Client for Real-Time Market Data Ingestion

This module provides a production-ready WebSocket client for ingesting real-time
market data from Binance with comprehensive features:

- Multi-stream WebSocket connections
- Order book depth updates
- Trade stream processing
- Automatic reconnection with exponential backoff
- Data normalization and validation
- Latency monitoring and quality metrics
- Error handling and circuit breaker patterns
"""

import asyncio
import json
import websockets
import time
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import pandas as pd
import numpy as np
from enum import Enum

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.websocket_reconnect import WebSocketReconnectHandler


class StreamType(Enum):
    """WebSocket stream types"""
    TICKER = "ticker"
    DEPTH = "depth"
    TRADE = "trade"
    KLINE = "kline"
    BOOK_TICKER = "bookTicker"


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket client"""
    # Connection settings
    base_url: str = "wss://stream.binance.com:9443"
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    stream_types: List[StreamType] = field(default_factory=lambda: [StreamType.TICKER])
    
    # Connection parameters
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 20.0
    connection_timeout: float = 10.0
    
    # Data processing
    buffer_size: int = 10000
    enable_data_validation: bool = True
    enable_latency_monitoring: bool = True
    
    # Quality monitoring
    max_latency_ms: float = 100.0
    quality_check_interval: float = 30.0
    
    # Rate limiting
    max_messages_per_second: int = 1000
    backpressure_threshold: int = 5000


@dataclass
class MarketDataPoint:
    """Normalized market data point"""
    symbol: str
    timestamp: pd.Timestamp
    event_type: str
    
    # Price data
    price: Optional[float] = None
    quantity: Optional[float] = None
    
    # Order book data
    bids: List[tuple] = field(default_factory=list)  # [(price, quantity), ...]
    asks: List[tuple] = field(default_factory=list)  # [(price, quantity), ...]
    
    # Best bid/ask
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    best_bid_qty: Optional[float] = None
    best_ask_qty: Optional[float] = None
    
    # Trade data
    trade_id: Optional[int] = None
    is_buyer_maker: Optional[bool] = None
    
    # Statistics
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_percent_24h: Optional[float] = None
    
    # Quality metrics
    latency_ms: Optional[float] = None
    source_timestamp: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'price': self.price,
            'quantity': self.quantity,
            'bids': self.bids[:10],  # Top 10 levels
            'asks': self.asks[:10],  # Top 10 levels
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'best_bid_qty': self.best_bid_qty,
            'best_ask_qty': self.best_ask_qty,
            'trade_id': self.trade_id,
            'is_buyer_maker': self.is_buyer_maker,
            'volume_24h': self.volume_24h,
            'price_change_24h': self.price_change_24h,
            'price_change_percent_24h': self.price_change_percent_24h,
            'latency_ms': self.latency_ms
        }


class WebSocketDataProcessor:
    """Process and normalize incoming WebSocket data"""
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.logger = get_logger("WebSocketDataProcessor")
        
        # Order book tracking
        self.order_books = {}  # symbol -> {bids: [], asks: []}
        self.last_trade_id = {}  # symbol -> trade_id
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'messages_invalid': 0,
            'latency_samples': deque(maxlen=1000),
            'symbols_active': set(),
            'last_update': time.time()
        }
    
    def process_message(self, raw_data: str) -> Optional[MarketDataPoint]:
        """Process raw WebSocket message into normalized data point"""
        try:
            data = json.loads(raw_data)
            receive_time = pd.Timestamp.now()
            
            # Handle different message formats
            if 'stream' in data and 'data' in data:
                # Combined stream format: /stream?streams=...
                return self._process_stream_message(data, receive_time)
            elif 'e' in data:
                # Single stream format: /ws/btcusdt@ticker
                return self._process_single_message(data, receive_time)
            else:
                self.logger.debug(f"Unknown message format: {data}")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.stats['messages_invalid'] += 1
            return None
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            self.stats['messages_invalid'] += 1
            return None
    
    def _process_stream_message(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> Optional[MarketDataPoint]:
        """Process combined stream message"""
        stream_name = data.get('stream', '')
        stream_data = data.get('data', {})
        
        if '@ticker' in stream_name:
            return self._process_ticker_data(stream_data, receive_time)
        elif '@depth' in stream_name:
            return self._process_depth_data(stream_data, receive_time)
        elif '@trade' in stream_name:
            return self._process_trade_data(stream_data, receive_time)
        elif '@bookTicker' in stream_name:
            return self._process_book_ticker_data(stream_data, receive_time)
        else:
            self.logger.debug(f"Unknown stream type: {stream_name}")
            return None
    
    def _process_single_message(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> Optional[MarketDataPoint]:
        """Process single stream message"""
        event_type = data.get('e', '')
        
        if event_type == '24hrTicker':
            return self._process_ticker_data(data, receive_time)
        elif event_type == 'depthUpdate':
            return self._process_depth_data(data, receive_time)
        elif event_type == 'trade':
            return self._process_trade_data(data, receive_time)
        elif event_type == 'bookTicker':
            return self._process_book_ticker_data(data, receive_time)
        else:
            self.logger.debug(f"Unknown event type: {event_type}")
            return None
    
    def _process_ticker_data(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> MarketDataPoint:
        """Process ticker data (24hr statistics)"""
        symbol = data.get('s', '').upper()
        
        # Calculate latency if event time is available
        latency_ms = None
        source_timestamp = None
        if 'E' in data:
            source_timestamp = pd.Timestamp(data['E'], unit='ms')
            latency_ms = (receive_time - source_timestamp).total_seconds() * 1000
            self.stats['latency_samples'].append(latency_ms)
        
        point = MarketDataPoint(
            symbol=symbol,
            timestamp=receive_time,
            event_type='ticker',
            price=float(data.get('c', 0)) if data.get('c') else None,  # Close price
            quantity=float(data.get('v', 0)) if data.get('v') else None,  # Volume
            best_bid=float(data.get('b', 0)) if data.get('b') else None,  # Best bid
            best_ask=float(data.get('a', 0)) if data.get('a') else None,  # Best ask
            best_bid_qty=float(data.get('B', 0)) if data.get('B') else None,  # Best bid qty
            best_ask_qty=float(data.get('A', 0)) if data.get('A') else None,  # Best ask qty
            volume_24h=float(data.get('v', 0)) if data.get('v') else None,
            price_change_24h=float(data.get('p', 0)) if data.get('p') else None,
            price_change_percent_24h=float(data.get('P', 0)) if data.get('P') else None,
            latency_ms=latency_ms,
            source_timestamp=source_timestamp
        )
        
        self.stats['messages_processed'] += 1
        self.stats['symbols_active'].add(symbol)
        return point
    
    def _process_depth_data(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> MarketDataPoint:
        """Process order book depth update"""
        symbol = data.get('s', '').upper()
        
        # Parse bids and asks
        bids = [(float(price), float(qty)) for price, qty in data.get('b', [])]
        asks = [(float(price), float(qty)) for price, qty in data.get('a', [])]
        
        # Update local order book tracking
        if symbol not in self.order_books:
            self.order_books[symbol] = {'bids': {}, 'asks': {}}
        
        # Apply updates to local book
        for price, qty in bids:
            if qty == 0:
                self.order_books[symbol]['bids'].pop(price, None)
            else:
                self.order_books[symbol]['bids'][price] = qty
        
        for price, qty in asks:
            if qty == 0:
                self.order_books[symbol]['asks'].pop(price, None)
            else:
                self.order_books[symbol]['asks'][price] = qty
        
        # Get best bid/ask
        book = self.order_books[symbol]
        best_bid = max(book['bids'].keys()) if book['bids'] else None
        best_ask = min(book['asks'].keys()) if book['asks'] else None
        
        # Calculate latency
        latency_ms = None
        source_timestamp = None
        if 'E' in data:
            source_timestamp = pd.Timestamp(data['E'], unit='ms')
            latency_ms = (receive_time - source_timestamp).total_seconds() * 1000
            self.stats['latency_samples'].append(latency_ms)
        
        point = MarketDataPoint(
            symbol=symbol,
            timestamp=receive_time,
            event_type='depth_update',
            bids=sorted([(p, q) for p, q in book['bids'].items()], reverse=True)[:20],
            asks=sorted([(p, q) for p, q in book['asks'].items()])[:20],
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_qty=book['bids'].get(best_bid) if best_bid else None,
            best_ask_qty=book['asks'].get(best_ask) if best_ask else None,
            latency_ms=latency_ms,
            source_timestamp=source_timestamp
        )
        
        self.stats['messages_processed'] += 1
        self.stats['symbols_active'].add(symbol)
        return point
    
    def _process_trade_data(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> MarketDataPoint:
        """Process individual trade data"""
        symbol = data.get('s', '').upper()
        trade_id = int(data.get('t', 0)) if data.get('t') else None
        
        # Calculate latency
        latency_ms = None
        source_timestamp = None
        if 'T' in data:
            source_timestamp = pd.Timestamp(data['T'], unit='ms')
            latency_ms = (receive_time - source_timestamp).total_seconds() * 1000
            self.stats['latency_samples'].append(latency_ms)
        
        point = MarketDataPoint(
            symbol=symbol,
            timestamp=receive_time,
            event_type='trade',
            price=float(data.get('p', 0)) if data.get('p') else None,
            quantity=float(data.get('q', 0)) if data.get('q') else None,
            trade_id=trade_id,
            is_buyer_maker=data.get('m', False),
            latency_ms=latency_ms,
            source_timestamp=source_timestamp
        )
        
        # Track trade sequence
        if trade_id:
            self.last_trade_id[symbol] = trade_id
        
        self.stats['messages_processed'] += 1
        self.stats['symbols_active'].add(symbol)
        return point
    
    def _process_book_ticker_data(self, data: Dict[str, Any], receive_time: pd.Timestamp) -> MarketDataPoint:
        """Process book ticker (best bid/ask) data"""
        symbol = data.get('s', '').upper()
        
        point = MarketDataPoint(
            symbol=symbol,
            timestamp=receive_time,
            event_type='book_ticker',
            best_bid=float(data.get('b', 0)) if data.get('b') else None,
            best_ask=float(data.get('a', 0)) if data.get('a') else None,
            best_bid_qty=float(data.get('B', 0)) if data.get('B') else None,
            best_ask_qty=float(data.get('A', 0)) if data.get('A') else None,
        )
        
        self.stats['messages_processed'] += 1
        self.stats['symbols_active'].add(symbol)
        return point
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        latency_samples = list(self.stats['latency_samples'])
        
        return {
            'messages_processed': self.stats['messages_processed'],
            'messages_invalid': self.stats['messages_invalid'],
            'symbols_active': len(self.stats['symbols_active']),
            'symbols_list': list(self.stats['symbols_active']),
            'latency_avg_ms': np.mean(latency_samples) if latency_samples else 0,
            'latency_p50_ms': np.percentile(latency_samples, 50) if latency_samples else 0,
            'latency_p95_ms': np.percentile(latency_samples, 95) if latency_samples else 0,
            'latency_p99_ms': np.percentile(latency_samples, 99) if latency_samples else 0,
            'order_books_tracked': len(self.order_books),
            'uptime_seconds': time.time() - self.stats['last_update']
        }


class BinanceWebSocketClient:
    """
    Production-ready Binance WebSocket client for real-time market data ingestion
    """
    
    def __init__(self, config: Optional[WebSocketConfig] = None):
        self.config = config or WebSocketConfig()
        self.logger = get_logger("BinanceWebSocketClient")
        self.processor = WebSocketDataProcessor(self.config)
        
        # Connection management
        self.websocket = None
        self.reconnect_handler = WebSocketReconnectHandler(
            max_attempts=self.config.max_reconnect_attempts,
            initial_delay=self.config.reconnect_delay
        )
        
        # State tracking
        self.connected = False
        self.streaming = False
        self.start_time = time.time()
        
        # Message handling
        self.message_queue = asyncio.Queue(maxsize=self.config.buffer_size)
        self.subscribers = []
        self.message_count = 0
        
        # Quality monitoring
        self.quality_monitor_task = None
        self.last_heartbeat = time.time()
        
        # Setup signal handling for graceful shutdown
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def add_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Add subscriber for market data updates"""
        self.subscribers.append(callback)
        self.logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    def remove_subscriber(self, callback: Callable[[MarketDataPoint], None]):
        """Remove subscriber"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            self.logger.info(f"Removed subscriber, total: {len(self.subscribers)}")
    
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            # Build WebSocket URL for combined streams
            streams = []
            for symbol in self.config.symbols:
                symbol_lower = symbol.lower()
                for stream_type in self.config.stream_types:
                    if stream_type == StreamType.TICKER:
                        streams.append(f"{symbol_lower}@ticker")
                    elif stream_type == StreamType.DEPTH:
                        streams.append(f"{symbol_lower}@depth20@100ms")
                    elif stream_type == StreamType.TRADE:
                        streams.append(f"{symbol_lower}@trade")
                    elif stream_type == StreamType.BOOK_TICKER:
                        streams.append(f"{symbol_lower}@bookTicker")
            
            # Use combined stream endpoint for multiple streams
            if len(streams) > 1:
                url = f"{self.config.base_url}/stream?streams={'/'.join(streams)}"
            else:
                url = f"{self.config.base_url}/ws/{streams[0]}"
            
            self.logger.info(f"Connecting to: {url}")
            self.logger.info(f"Subscribing to {len(streams)} streams for {len(self.config.symbols)} symbols")
            
            # Connect with timeout
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    url,
                    ping_interval=self.config.heartbeat_interval,
                    ping_timeout=self.config.connection_timeout
                ),
                timeout=self.config.connection_timeout
            )
            
            self.connected = True
            self.last_heartbeat = time.time()
            self.logger.info("WebSocket connection established successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect WebSocket and cleanup"""
        self.running = False
        self.connected = False
        self.streaming = False
        
        # Cancel quality monitoring
        if self.quality_monitor_task:
            self.quality_monitor_task.cancel()
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.logger.info("WebSocket connection closed")
    
    async def start_streaming(self) -> AsyncIterator[MarketDataPoint]:
        """Start streaming market data"""
        if not self.connected:
            raise RuntimeError("Not connected - call connect() first")
        
        self.streaming = True
        self.logger.info("Starting market data streaming")
        
        # Start quality monitoring
        self.quality_monitor_task = asyncio.create_task(self._quality_monitor())
        
        try:
            async for raw_message in self.websocket:
                if not self.running:
                    break
                
                # Process message
                data_point = self.processor.process_message(raw_message)
                if data_point:
                    self.message_count += 1
                    
                    # Notify subscribers
                    for subscriber in self.subscribers:
                        try:
                            if asyncio.iscoroutinefunction(subscriber):
                                await subscriber(data_point)
                            else:
                                subscriber(data_point)
                        except Exception as e:
                            self.logger.error(f"Subscriber error: {e}")
                    
                    # Update heartbeat
                    self.last_heartbeat = time.time()
                    
                    yield data_point
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed by server")
            self.connected = False
            
            # Attempt reconnection if configured
            if self.running and self.reconnect_handler:
                await self._attempt_reconnection()
                
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            self.connected = False
        
        finally:
            self.streaming = False
    
    async def _attempt_reconnection(self):
        """Attempt to reconnect with exponential backoff"""
        self.logger.info("Attempting to reconnect...")
        
        for attempt in range(self.config.max_reconnect_attempts):
            if not self.running:
                break
            
            delay = self.reconnect_handler.get_delay(attempt)
            self.logger.info(f"Reconnection attempt {attempt + 1}/{self.config.max_reconnect_attempts} in {delay}s")
            
            await asyncio.sleep(delay)
            
            if await self.connect():
                self.logger.info("Reconnection successful")
                return True
        
        self.logger.error("All reconnection attempts failed")
        return False
    
    async def _quality_monitor(self):
        """Monitor data quality and connection health"""
        while self.running and self.connected:
            try:
                await asyncio.sleep(self.config.quality_check_interval)
                
                stats = self.get_statistics()
                
                # Check latency
                if stats['latency_p95_ms'] > self.config.max_latency_ms:
                    self.logger.warning(f"High latency detected: {stats['latency_p95_ms']:.1f}ms (P95)")
                
                # Check message flow
                time_since_last_message = time.time() - self.last_heartbeat
                if time_since_last_message > 60:  # No messages for 1 minute
                    self.logger.warning(f"No messages received for {time_since_last_message:.1f}s")
                
                # Log stats periodically
                self.logger.info(f"Quality check: {stats['messages_processed']} messages, "
                               f"{stats['symbols_active']} symbols, "
                               f"{stats['latency_avg_ms']:.1f}ms avg latency")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Quality monitor error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        processor_stats = self.processor.get_statistics()
        
        uptime = time.time() - self.start_time
        message_rate = self.message_count / uptime if uptime > 0 else 0
        
        return {
            'connection': {
                'connected': self.connected,
                'streaming': self.streaming,
                'uptime_seconds': uptime,
                'reconnect_attempts': self.reconnect_handler.attempt_count if self.reconnect_handler else 0,
            },
            'messages': {
                'total_count': self.message_count,
                'rate_per_second': message_rate,
                'queue_size': self.message_queue.qsize(),
                'subscribers': len(self.subscribers)
            },
            'processing': processor_stats,
            'config': {
                'symbols': self.config.symbols,
                'stream_types': [st.value for st in self.config.stream_types],
                'buffer_size': self.config.buffer_size,
                'max_latency_ms': self.config.max_latency_ms
            }
        }
    
    async def run_continuous(self):
        """Run continuous streaming with automatic reconnection"""
        self.logger.info("Starting continuous market data streaming...")
        
        while self.running:
            try:
                # Connect if not connected
                if not self.connected:
                    success = await self.connect()
                    if not success:
                        await asyncio.sleep(5)  # Wait before retry
                        continue
                
                # Start streaming
                message_count = 0
                async for data_point in self.start_streaming():
                    message_count += 1
                    
                    # Log progress periodically
                    if message_count % 1000 == 0:
                        stats = self.get_statistics()
                        self.logger.info(f"Processed {message_count} messages, "
                                       f"{stats['processing']['symbols_active']} symbols active")
                
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                self.connected = False
                await asyncio.sleep(5)  # Wait before reconnection attempt
        
        await self.disconnect()
        self.logger.info("Continuous streaming stopped")


async def main():
    """Main entry point for testing WebSocket client"""
    print("🚀 Binance WebSocket Client Test")
    print("=" * 50)
    
    # Configure client for comprehensive testing
    config = WebSocketConfig(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        stream_types=[StreamType.TICKER, StreamType.DEPTH, StreamType.TRADE],
        enable_latency_monitoring=True,
        quality_check_interval=10.0,
        max_latency_ms=200.0
    )
    
    client = BinanceWebSocketClient(config)
    
    # Add a simple subscriber to print data
    def print_data(data_point: MarketDataPoint):
        if data_point.event_type == 'ticker':
            price = data_point.price or 0
            quantity = data_point.quantity or 0
            spread = (data_point.best_ask or 0) - (data_point.best_bid or 0)
            print(f"📈 {data_point.symbol}: ${price:.4f} "
                  f"(Vol: {quantity:,.0f}, "
                  f"Spread: ${spread:.4f})")
        elif data_point.event_type == 'trade':
            side = "🟢 BUY" if not data_point.is_buyer_maker else "🔴 SELL"
            price = data_point.price or 0
            quantity = data_point.quantity or 0
            print(f"💱 {data_point.symbol}: {side} {quantity:.6f} @ ${price:.4f}")
        elif data_point.event_type == 'depth_update':
            best_bid = data_point.best_bid or 0
            best_ask = data_point.best_ask or 0
            print(f"📊 {data_point.symbol}: Book update - "
                  f"Best: ${best_bid:.4f} / ${best_ask:.4f}")
    
    client.add_subscriber(print_data)
    
    try:
        # Connect and start streaming
        connected = await client.connect()
        if not connected:
            print("❌ Failed to connect")
            return 1
        
        print("✅ Connected successfully")
        print("📊 Starting data streaming for 60 seconds...")
        
        # Stream for a test period
        start_time = time.time()
        message_count = 0
        
        async for data_point in client.start_streaming():
            message_count += 1
            
            # Show progress every 50 messages
            if message_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = message_count / elapsed
                print(f"📈 Progress: {message_count} messages at {rate:.1f} msg/s")
            
            # Test for 60 seconds
            if time.time() - start_time > 60:
                break
        
        # Print final statistics
        stats = client.get_statistics()
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        print(f"⏱️  Runtime: {stats['connection']['uptime_seconds']:.1f}s")
        print(f"📨 Messages: {stats['messages']['total_count']:,}")
        print(f"📈 Rate: {stats['messages']['rate_per_second']:.1f} msg/s")
        print(f"💫 Symbols: {stats['processing']['symbols_active']}")
        print(f"📊 Avg Latency: {stats['processing']['latency_avg_ms']:.1f}ms")
        print(f"🔄 P95 Latency: {stats['processing']['latency_p95_ms']:.1f}ms")
        print(f"📚 Order Books: {stats['processing']['order_books_tracked']}")
        
        if stats['processing']['latency_p95_ms'] < 100:
            print("🎉 LOW LATENCY: Excellent performance!")
        elif stats['processing']['latency_p95_ms'] < 500:
            print("✅ GOOD LATENCY: Acceptable for trading")
        else:
            print("⚠️  HIGH LATENCY: May impact trading performance")
        
        await client.disconnect()
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        await client.disconnect()
        return 1
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        await client.disconnect()
        return 1


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
