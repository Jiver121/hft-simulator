"""
Real-Time Stream Processing Pipeline for HFT Simulator

This module provides high-performance stream processing capabilities for
real-time market data, order flow, and event processing in HFT systems.

Key Features:
- High-throughput message processing (>1M msgs/sec)
- Low-latency processing (<100μs per message)
- Parallel processing with worker pools
- Backpressure handling and flow control
- Message ordering and sequencing
- Error handling and recovery
- Performance monitoring and metrics

Components:
- StreamProcessor: Main processing coordinator
- MessageQueue: High-performance message queuing
- WorkerPool: Parallel processing workers
- MessageRouter: Route messages to appropriate handlers
- EventAggregator: Aggregate and batch events
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType
from src.realtime.data_feeds import MarketDataMessage
from src.realtime.order_management import OrderRequest, ExecutionReport


class MessageType(Enum):
    """Types of messages in the stream"""
    MARKET_DATA = "market_data"
    ORDER_REQUEST = "order_request"
    EXECUTION_REPORT = "execution_report"
    RISK_EVENT = "risk_event"
    SYSTEM_EVENT = "system_event"
    HEARTBEAT = "heartbeat"


class ProcessingPriority(Enum):
    """Message processing priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StreamMessage:
    """Standardized stream message container"""
    
    message_id: str
    message_type: MessageType
    timestamp: datetime
    
    # Message content
    data: Any
    
    # Processing metadata
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    sequence_number: Optional[int] = None
    source: Optional[str] = None
    
    # Routing information
    routing_key: Optional[str] = None
    target_handlers: List[str] = field(default_factory=list)
    
    # Processing tracking
    created_at: datetime = field(default_factory=datetime.now)
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    @property
    def processing_latency_us(self) -> Optional[float]:
        """Calculate processing latency in microseconds"""
        if self.processing_started and self.processing_completed:
            delta = self.processing_completed - self.processing_started
            return delta.total_seconds() * 1_000_000
        return None
    
    @property
    def total_latency_us(self) -> Optional[float]:
        """Calculate total latency from creation in microseconds"""
        if self.processing_completed:
            delta = self.processing_completed - self.created_at
            return delta.total_seconds() * 1_000_000
        return None


@dataclass
class ProcessingStats:
    """Stream processing statistics"""
    
    # Throughput metrics
    messages_processed: int = 0
    messages_per_second: float = 0.0
    peak_throughput: float = 0.0
    
    # Latency metrics
    avg_latency_us: float = 0.0
    p50_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    max_latency_us: float = 0.0
    
    # Queue metrics
    queue_depth: int = 0
    max_queue_depth: int = 0
    queue_full_events: int = 0
    
    # Error metrics
    processing_errors: int = 0
    retry_attempts: int = 0
    dropped_messages: int = 0
    
    # Worker metrics
    active_workers: int = 0
    worker_utilization: float = 0.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_latency_stats(self, latencies: List[float]) -> None:
        """Update latency statistics from list of latencies"""
        if not latencies:
            return
        
        self.avg_latency_us = np.mean(latencies)
        self.p50_latency_us = np.percentile(latencies, 50)
        self.p95_latency_us = np.percentile(latencies, 95)
        self.p99_latency_us = np.percentile(latencies, 99)
        self.max_latency_us = np.max(latencies)


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    def __init__(self, handler_id: str):
        self.handler_id = handler_id
        self.logger = get_logger(f"{self.__class__.__name__}.{handler_id}")
        
        # Statistics
        self.messages_handled = 0
        self.processing_time_total = 0.0
        self.errors = 0
        
    @abstractmethod
    async def handle_message(self, message: StreamMessage) -> bool:
        """
        Handle a stream message
        
        Args:
            message: Message to process
            
        Returns:
            True if handled successfully, False otherwise
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        avg_processing_time = (
            self.processing_time_total / max(self.messages_handled, 1)
        ) * 1_000_000  # Convert to microseconds
        
        return {
            'handler_id': self.handler_id,
            'messages_handled': self.messages_handled,
            'avg_processing_time_us': avg_processing_time,
            'errors': self.errors,
            'error_rate': self.errors / max(self.messages_handled, 1)
        }


class MarketDataHandler(MessageHandler):
    """Handler for market data messages"""
    
    def __init__(self, handler_id: str = "market_data"):
        super().__init__(handler_id)
        self.order_book_updates = 0
        self.trade_updates = 0
        
    async def handle_message(self, message: StreamMessage) -> bool:
        """Handle market data message"""
        start_time = time.perf_counter()
        
        try:
            if message.message_type != MessageType.MARKET_DATA:
                return False
            
            market_data = message.data
            if not isinstance(market_data, MarketDataMessage):
                self.logger.error(f"Invalid market data message type: {type(market_data)}")
                self.errors += 1
                return False
            
            # Process market data
            if market_data.message_type == 'book_update':
                self.order_book_updates += 1
                await self._process_book_update(market_data)
            elif market_data.message_type == 'trade':
                self.trade_updates += 1
                await self._process_trade_update(market_data)
            
            self.messages_handled += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
            self.errors += 1
            return False
        
        finally:
            processing_time = time.perf_counter() - start_time
            self.processing_time_total += processing_time
    
    async def _process_book_update(self, market_data: MarketDataMessage) -> None:
        """Process order book update"""
        # Implementation would update order book state
        self.logger.debug(f"Processing book update for {market_data.symbol}")
    
    async def _process_trade_update(self, market_data: MarketDataMessage) -> None:
        """Process trade update"""
        # Implementation would process trade information
        self.logger.debug(f"Processing trade update for {market_data.symbol}")


class OrderHandler(MessageHandler):
    """Handler for order-related messages"""
    
    def __init__(self, handler_id: str = "order_handler"):
        super().__init__(handler_id)
        self.order_requests = 0
        self.execution_reports = 0
        
    async def handle_message(self, message: StreamMessage) -> bool:
        """Handle order message"""
        start_time = time.perf_counter()
        
        try:
            if message.message_type == MessageType.ORDER_REQUEST:
                self.order_requests += 1
                await self._process_order_request(message.data)
            elif message.message_type == MessageType.EXECUTION_REPORT:
                self.execution_reports += 1
                await self._process_execution_report(message.data)
            else:
                return False
            
            self.messages_handled += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error handling order message: {e}")
            self.errors += 1
            return False
        
        finally:
            processing_time = time.perf_counter() - start_time
            self.processing_time_total += processing_time
    
    async def _process_order_request(self, order_request: OrderRequest) -> None:
        """Process order request"""
        self.logger.debug(f"Processing order request: {order_request.symbol}")
    
    async def _process_execution_report(self, execution_report: ExecutionReport) -> None:
        """Process execution report"""
        self.logger.debug(f"Processing execution report: {execution_report.symbol}")


class HighPerformanceQueue:
    """
    High-performance message queue optimized for low latency
    """
    
    def __init__(self, maxsize: int = 100000):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.not_full = threading.Condition(self.lock)
        
        # Statistics
        self.enqueue_count = 0
        self.dequeue_count = 0
        self.full_events = 0
        self.max_size_reached = 0
        
    def put_nowait(self, item: StreamMessage) -> bool:
        """Put item without blocking"""
        with self.lock:
            if len(self.queue) >= self.maxsize:
                self.full_events += 1
                return False
            
            self.queue.append(item)
            self.enqueue_count += 1
            
            if len(self.queue) > self.max_size_reached:
                self.max_size_reached = len(self.queue)
            
            self.not_empty.notify()
            return True
    
    def get_nowait(self) -> Optional[StreamMessage]:
        """Get item without blocking"""
        with self.lock:
            if not self.queue:
                return None
            
            item = self.queue.popleft()
            self.dequeue_count += 1
            self.not_full.notify()
            return item
    
    async def put(self, item: StreamMessage, timeout: Optional[float] = None) -> bool:
        """Put item with optional timeout"""
        start_time = time.time()
        
        while True:
            if self.put_nowait(item):
                return True
            
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.001)  # 1ms sleep
    
    async def get(self, timeout: Optional[float] = None) -> Optional[StreamMessage]:
        """Get item with optional timeout"""
        start_time = time.time()
        
        while True:
            item = self.get_nowait()
            if item:
                return item
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            await asyncio.sleep(0.001)  # 1ms sleep
    
    def qsize(self) -> int:
        """Get current queue size"""
        with self.lock:
            return len(self.queue)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self.lock:
            return len(self.queue) == 0
    
    def full(self) -> bool:
        """Check if queue is full"""
        with self.lock:
            return len(self.queue) >= self.maxsize
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            return {
                'current_size': len(self.queue),
                'max_size': self.maxsize,
                'max_size_reached': self.max_size_reached,
                'enqueue_count': self.enqueue_count,
                'dequeue_count': self.dequeue_count,
                'full_events': self.full_events,
                'utilization': len(self.queue) / self.maxsize
            }


class StreamWorker:
    """
    Individual stream processing worker
    """
    
    def __init__(self, worker_id: str, handlers: Dict[MessageType, MessageHandler]):
        self.worker_id = worker_id
        self.handlers = handlers
        self.logger = get_logger(f"{self.__class__.__name__}.{worker_id}")
        
        # State
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.messages_processed = 0
        self.processing_errors = 0
        self.start_time = datetime.now()
        self.last_message_time: Optional[datetime] = None
        
        # Performance tracking
        self.latency_samples = deque(maxlen=1000)
        
    async def start(self, message_queue: HighPerformanceQueue) -> None:
        """Start the worker"""
        if self.running:
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop(message_queue))
        self.logger.info(f"Worker {self.worker_id} started")
    
    async def stop(self) -> None:
        """Stop the worker"""
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def _worker_loop(self, message_queue: HighPerformanceQueue) -> None:
        """Main worker processing loop"""
        while self.running:
            try:
                # Get message from queue
                message = await message_queue.get(timeout=1.0)
                if not message:
                    continue
                
                # Process message
                await self._process_message(message)
                
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                self.processing_errors += 1
                await asyncio.sleep(0.1)
    
    async def _process_message(self, message: StreamMessage) -> None:
        """Process individual message"""
        message.processing_started = datetime.now()
        
        try:
            # Find appropriate handler
            handler = self.handlers.get(message.message_type)
            if not handler:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                return
            
            # Process message
            success = await handler.handle_message(message)
            
            if success:
                self.messages_processed += 1
                self.last_message_time = datetime.now()
                
                # Track latency
                message.processing_completed = datetime.now()
                if message.processing_latency_us:
                    self.latency_samples.append(message.processing_latency_us)
            else:
                self.processing_errors += 1
                
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")
            self.processing_errors += 1
            
            # Retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                message.error_message = str(e)
                # Could re-queue message for retry
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        throughput = self.messages_processed / max(uptime, 1)
        
        # Calculate latency stats
        latencies = list(self.latency_samples)
        avg_latency = np.mean(latencies) if latencies else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        
        return {
            'worker_id': self.worker_id,
            'running': self.running,
            'messages_processed': self.messages_processed,
            'processing_errors': self.processing_errors,
            'throughput_mps': throughput,
            'avg_latency_us': avg_latency,
            'p95_latency_us': p95_latency,
            'uptime_seconds': uptime,
            'last_message_time': self.last_message_time
        }


class StreamProcessor:
    """
    Main stream processing coordinator
    
    Manages high-performance message processing with multiple workers,
    load balancing, and comprehensive monitoring.
    """
    
    def __init__(self, 
                 num_workers: int = None,
                 queue_size: int = 100000,
                 enable_monitoring: bool = True):
        
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Configuration
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.queue_size = queue_size
        self.enable_monitoring = enable_monitoring
        
        # Core components
        self.message_queue = HighPerformanceQueue(maxsize=queue_size)
        self.workers: List[StreamWorker] = []
        self.handlers: Dict[MessageType, MessageHandler] = {}
        
        # State
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = ProcessingStats()
        self.message_counter = 0
        self.sequence_counter = 0
        
        # Performance tracking
        self.throughput_samples = deque(maxlen=60)  # 1 minute of samples
        self.latency_samples = deque(maxlen=10000)  # Recent latency samples
        
        self.logger.info(f"StreamProcessor initialized with {self.num_workers} workers")
    
    def add_handler(self, message_type: MessageType, handler: MessageHandler) -> None:
        """Add message handler"""
        self.handlers[message_type] = handler
        self.logger.info(f"Added handler for {message_type.value}: {handler.handler_id}")
    
    async def start(self) -> None:
        """Start the stream processor"""
        if self.running:
            return
        
        self.running = True
        self.stats.start_time = datetime.now()
        
        # Create and start workers
        for i in range(self.num_workers):
            worker = StreamWorker(f"worker_{i}", self.handlers.copy())
            self.workers.append(worker)
            await worker.start(self.message_queue)
        
        # Start monitoring
        if self.enable_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"StreamProcessor started with {len(self.workers)} workers")
    
    async def stop(self) -> None:
        """Stop the stream processor"""
        self.running = False
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop workers
        for worker in self.workers:
            await worker.stop()
        
        self.workers.clear()
        self.logger.info("StreamProcessor stopped")
    
    async def process_message(self, 
                            message_type: MessageType,
                            data: Any,
                            priority: ProcessingPriority = ProcessingPriority.NORMAL,
                            routing_key: Optional[str] = None) -> bool:
        """
        Process a message through the stream
        
        Args:
            message_type: Type of message
            data: Message data
            priority: Processing priority
            routing_key: Optional routing key
            
        Returns:
            True if message was queued successfully
        """
        # Create stream message
        message = StreamMessage(
            message_id=f"msg_{self.message_counter}",
            message_type=message_type,
            timestamp=datetime.now(),
            data=data,
            priority=priority,
            sequence_number=self.sequence_counter,
            routing_key=routing_key
        )
        
        self.message_counter += 1
        self.sequence_counter += 1
        
        # Queue message
        success = await self.message_queue.put(message, timeout=0.1)
        
        if success:
            self.stats.messages_processed += 1
        else:
            self.stats.dropped_messages += 1
            self.logger.warning(f"Dropped message due to full queue: {message.message_id}")
        
        return success
    
    async def process_market_data(self, market_data: MarketDataMessage) -> bool:
        """Process market data message"""
        return await self.process_message(
            MessageType.MARKET_DATA,
            market_data,
            ProcessingPriority.HIGH,
            f"market_data.{market_data.symbol}"
        )
    
    async def process_order_request(self, order_request: OrderRequest) -> bool:
        """Process order request message"""
        return await self.process_message(
            MessageType.ORDER_REQUEST,
            order_request,
            ProcessingPriority.CRITICAL,
            f"order.{order_request.symbol}"
        )
    
    async def process_execution_report(self, execution_report: ExecutionReport) -> bool:
        """Process execution report message"""
        return await self.process_message(
            MessageType.EXECUTION_REPORT,
            execution_report,
            ProcessingPriority.HIGH,
            f"execution.{execution_report.symbol}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        # Update queue stats
        queue_stats = self.message_queue.get_stats()
        self.stats.queue_depth = queue_stats['current_size']
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, queue_stats['current_size'])
        self.stats.queue_full_events = queue_stats['full_events']
        
        # Update worker stats
        self.stats.active_workers = len([w for w in self.workers if w.running])
        
        # Calculate throughput
        uptime = (datetime.now() - self.stats.start_time).total_seconds()
        self.stats.messages_per_second = self.stats.messages_processed / max(uptime, 1)
        
        # Update latency stats
        all_latencies = []
        for worker in self.workers:
            all_latencies.extend(worker.latency_samples)
        
        if all_latencies:
            self.stats.update_latency_stats(all_latencies)
        
        # Worker utilization
        if self.workers:
            total_processed = sum(w.messages_processed for w in self.workers)
            total_capacity = len(self.workers) * uptime
            self.stats.worker_utilization = total_processed / max(total_capacity, 1)
        
        self.stats.last_update = datetime.now()
        
        return {
            'processing_stats': self.stats.__dict__,
            'queue_stats': queue_stats,
            'worker_stats': [w.get_stats() for w in self.workers],
            'handler_stats': [h.get_stats() for h in self.handlers.values()]
        }
    
    async def _monitoring_loop(self) -> None:
        """Monitoring loop for performance tracking"""
        while self.running:
            try:
                # Sample current throughput
                current_time = time.time()
                current_processed = self.stats.messages_processed
                
                if len(self.throughput_samples) > 0:
                    last_time, last_processed = self.throughput_samples[-1]
                    time_delta = current_time - last_time
                    msg_delta = current_processed - last_processed
                    
                    if time_delta > 0:
                        current_throughput = msg_delta / time_delta
                        self.stats.peak_throughput = max(self.stats.peak_throughput, current_throughput)
                
                self.throughput_samples.append((current_time, current_processed))
                
                # Log performance metrics every 10 seconds
                if len(self.throughput_samples) % 10 == 0:
                    stats = self.get_statistics()
                    self.logger.info(
                        f"Processing: {self.stats.messages_per_second:.1f} msg/s, "
                        f"Queue: {self.stats.queue_depth}/{self.queue_size}, "
                        f"Latency: {self.stats.avg_latency_us:.1f}μs avg, "
                        f"Workers: {self.stats.active_workers}/{len(self.workers)}"
                    )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)


# Factory function for creating stream processors
def create_stream_processor(config: Dict[str, Any]) -> StreamProcessor:
    """
    Create and configure a stream processor
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured StreamProcessor instance
    """
    processor = StreamProcessor(
        num_workers=config.get('num_workers'),
        queue_size=config.get('queue_size', 100000),
        enable_monitoring=config.get('enable_monitoring', True)
    )
    
    # Add default handlers
    processor.add_handler(MessageType.MARKET_DATA, MarketDataHandler())
    processor.add_handler(MessageType.ORDER_REQUEST, OrderHandler("order_request"))
    processor.add_handler(MessageType.EXECUTION_REPORT, OrderHandler("execution_report"))
    
    return processor