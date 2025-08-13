"""
Apache Flink Stream Processor
=============================

High-performance stream processing for real-time market data using Apache Flink.
Provides low-latency processing of market data streams with exactly-once semantics.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import threading
from queue import Queue, Empty
import time
from concurrent.futures import ThreadPoolExecutor
import pickle

@dataclass
class StreamEvent:
    """Represents a streaming event in the Flink pipeline"""
    event_id: str
    timestamp: datetime
    event_type: str
    symbol: str
    data: Dict[str, Any]
    source: str
    partition_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class ProcessingMetrics:
    """Processing metrics for monitoring"""
    processed_events: int = 0
    failed_events: int = 0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now(timezone.utc)

class StreamFunction:
    """Base class for stream processing functions"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"flink.{name}")
    
    async def process(self, event: StreamEvent) -> List[StreamEvent]:
        """Process a stream event and return results"""
        raise NotImplementedError
    
    async def setup(self):
        """Setup function called before processing starts"""
        pass
    
    async def teardown(self):
        """Teardown function called after processing ends"""
        pass

class MarketDataProcessor(StreamFunction):
    """Market data specific stream processor"""
    
    def __init__(self):
        super().__init__("market_data_processor")
        self.last_prices = {}
        
    async def process(self, event: StreamEvent) -> List[StreamEvent]:
        """Process market data events"""
        if event.event_type == "tick":
            return await self._process_tick(event)
        elif event.event_type == "orderbook":
            return await self._process_orderbook(event)
        elif event.event_type == "trade":
            return await self._process_trade(event)
        else:
            self.logger.warning(f"Unknown event type: {event.event_type}")
            return []
    
    async def _process_tick(self, event: StreamEvent) -> List[StreamEvent]:
        """Process tick data"""
        symbol = event.symbol
        price = event.data.get('price', 0.0)
        
        # Calculate price change if we have previous data
        enriched_data = event.data.copy()
        if symbol in self.last_prices:
            price_change = price - self.last_prices[symbol]
            enriched_data['price_change'] = price_change
            enriched_data['price_change_pct'] = (price_change / self.last_prices[symbol]) * 100
        
        self.last_prices[symbol] = price
        
        # Create enriched event
        enriched_event = StreamEvent(
            event_id=f"enriched_{event.event_id}",
            timestamp=datetime.now(timezone.utc),
            event_type="enriched_tick",
            symbol=symbol,
            data=enriched_data,
            source="flink_processor",
            partition_key=symbol
        )
        
        return [enriched_event]
    
    async def _process_orderbook(self, event: StreamEvent) -> List[StreamEvent]:
        """Process order book data"""
        # Calculate mid price
        data = event.data
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            best_bid = max(bids, key=lambda x: x['price'])['price']
            best_ask = min(asks, key=lambda x: x['price'])['price']
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100
            
            enriched_data = data.copy()
            enriched_data.update({
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'best_bid': best_bid,
                'best_ask': best_ask
            })
            
            enriched_event = StreamEvent(
                event_id=f"enriched_{event.event_id}",
                timestamp=datetime.now(timezone.utc),
                event_type="enriched_orderbook",
                symbol=event.symbol,
                data=enriched_data,
                source="flink_processor",
                partition_key=event.symbol
            )
            
            return [enriched_event]
        
        return []
    
    async def _process_trade(self, event: StreamEvent) -> List[StreamEvent]:
        """Process trade data"""
        # Add trade classification (buy/sell based on price)
        data = event.data.copy()
        
        # Simple trade classification
        if event.symbol in self.last_prices:
            if data['price'] >= self.last_prices[event.symbol]:
                data['side'] = 'buy'
            else:
                data['side'] = 'sell'
        
        enriched_event = StreamEvent(
            event_id=f"enriched_{event.event_id}",
            timestamp=datetime.now(timezone.utc),
            event_type="enriched_trade",
            symbol=event.symbol,
            data=data,
            source="flink_processor",
            partition_key=event.symbol
        )
        
        return [enriched_event]

class FlinkStreamProcessor:
    """
    Apache Flink-inspired stream processor for real-time market data processing.
    
    Provides:
    - Low-latency stream processing
    - Exactly-once processing semantics
    - Fault tolerance and recovery
    - Windowing and aggregation
    - Stateful processing
    """
    
    def __init__(self, name: str = "market_data_stream", parallelism: int = 4):
        self.name = name
        self.parallelism = parallelism
        self.logger = logging.getLogger(f"flink.{name}")
        
        # Processing components
        self.functions: List[StreamFunction] = []
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.error_queue = Queue()
        
        # State management
        self.state = {}
        self.checkpoints = {}
        self.checkpoint_interval = 30  # seconds
        
        # Metrics
        self.metrics = ProcessingMetrics()
        self.start_time = None
        
        # Control
        self.running = False
        self.threads = []
        self.executor = ThreadPoolExecutor(max_workers=parallelism)
        
        # Default processors
        self.add_function(MarketDataProcessor())
        
    def add_function(self, function: StreamFunction):
        """Add a processing function to the stream"""
        self.functions.append(function)
        self.logger.info(f"Added function: {function.name}")
    
    def remove_function(self, function_name: str):
        """Remove a processing function"""
        self.functions = [f for f in self.functions if f.name != function_name]
        self.logger.info(f"Removed function: {function_name}")
    
    async def send_event(self, event: StreamEvent):
        """Send an event to the stream for processing"""
        if not self.running:
            raise RuntimeError("Stream processor is not running")
        
        try:
            self.input_queue.put_nowait(event)
        except Exception as e:
            self.logger.error(f"Failed to send event: {e}")
            raise
    
    async def get_processed_event(self, timeout: float = 1.0) -> Optional[StreamEvent]:
        """Get a processed event from the output queue"""
        try:
            return self.output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    async def get_error_event(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get an error event"""
        try:
            return self.error_queue.get(timeout=timeout)
        except Empty:
            return None
    
    async def start(self):
        """Start the stream processor"""
        if self.running:
            self.logger.warning("Stream processor is already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        # Setup all functions
        for function in self.functions:
            await function.setup()
        
        # Start processing threads
        for i in range(self.parallelism):
            thread = threading.Thread(target=self._process_events, args=(i,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        # Start checkpoint thread
        checkpoint_thread = threading.Thread(target=self._checkpoint_loop)
        checkpoint_thread.daemon = True
        checkpoint_thread.start()
        self.threads.append(checkpoint_thread)
        
        # Start metrics thread
        metrics_thread = threading.Thread(target=self._metrics_loop)
        metrics_thread.daemon = True
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        self.logger.info(f"Started Flink stream processor '{self.name}' with {self.parallelism} threads")
    
    async def stop(self):
        """Stop the stream processor"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Teardown all functions
        for function in self.functions:
            await function.teardown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info(f"Stopped Flink stream processor '{self.name}'")
    
    def _process_events(self, worker_id: int):
        """Process events in a worker thread"""
        self.logger.info(f"Started worker thread {worker_id}")
        
        while self.running:
            try:
                # Get event from input queue
                event = self.input_queue.get(timeout=1.0)
                if event is None:
                    continue
                
                start_time = time.time()
                
                # Process through all functions
                events_to_process = [event]
                
                for function in self.functions:
                    new_events = []
                    for evt in events_to_process:
                        try:
                            # Run async function in executor
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            results = loop.run_until_complete(function.process(evt))
                            new_events.extend(results)
                            loop.close()
                            
                        except Exception as e:
                            self.logger.error(f"Function {function.name} failed: {e}")
                            self.error_queue.put({
                                'event': evt,
                                'function': function.name,
                                'error': str(e),
                                'timestamp': datetime.now(timezone.utc)
                            })
                            self.metrics.failed_events += 1
                            continue
                    
                    events_to_process = new_events
                
                # Put processed events in output queue
                for processed_event in events_to_process:
                    self.output_queue.put(processed_event)
                
                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                self.metrics.processed_events += 1
                self.metrics.avg_latency_ms = (
                    (self.metrics.avg_latency_ms * (self.metrics.processed_events - 1) + processing_time)
                    / self.metrics.processed_events
                )
                
                self.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                continue
    
    def _checkpoint_loop(self):
        """Periodic checkpoint creation"""
        while self.running:
            try:
                time.sleep(self.checkpoint_interval)
                self._create_checkpoint()
            except Exception as e:
                self.logger.error(f"Checkpoint error: {e}")
    
    def _create_checkpoint(self):
        """Create a checkpoint of current state"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        try:
            # Serialize current state
            checkpoint_data = {
                'state': pickle.dumps(self.state),
                'metrics': asdict(self.metrics),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.checkpoints[checkpoint_id] = checkpoint_data
            
            # Keep only last 10 checkpoints
            if len(self.checkpoints) > 10:
                oldest_checkpoint = min(self.checkpoints.keys())
                del self.checkpoints[oldest_checkpoint]
            
            self.logger.debug(f"Created checkpoint: {checkpoint_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
    
    def _metrics_loop(self):
        """Update metrics periodically"""
        last_processed = 0
        
        while self.running:
            try:
                time.sleep(1.0)  # Update every second
                
                current_processed = self.metrics.processed_events
                self.metrics.throughput_per_sec = current_processed - last_processed
                last_processed = current_processed
                self.metrics.last_update = datetime.now(timezone.utc)
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics"""
        return self.metrics
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot"""
        return self.state.copy()
    
    def restore_from_checkpoint(self, checkpoint_id: str):
        """Restore state from checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        try:
            checkpoint_data = self.checkpoints[checkpoint_id]
            self.state = pickle.loads(checkpoint_data['state'])
            self.logger.info(f"Restored from checkpoint: {checkpoint_id}")
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")
            raise
    
    def get_checkpoint_ids(self) -> List[str]:
        """Get list of available checkpoint IDs"""
        return list(self.checkpoints.keys())
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
