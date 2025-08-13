"""
Test Suite for Flink Stream Processor
=====================================

Comprehensive tests for the Apache Flink-inspired streaming infrastructure.
"""

import pytest
import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import List

# Set up basic logging for tests
logging.basicConfig(level=logging.INFO)

from src.streaming.flink.stream_processor import (
    FlinkStreamProcessor,
    StreamEvent,
    StreamFunction,
    ProcessingMetrics,
    MarketDataProcessor
)

class TestStreamFunction(StreamFunction):
    """Test stream function for unit tests"""
    
    def __init__(self, name: str = "test_function", delay_ms: float = 0):
        super().__init__(name)
        self.delay_ms = delay_ms
        self.processed_events = []
        
    async def process(self, event: StreamEvent) -> List[StreamEvent]:
        """Process event with optional delay"""
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)
        
        self.processed_events.append(event)
        
        # Create a processed event
        processed_event = StreamEvent(
            event_id=f"processed_{event.event_id}",
            timestamp=datetime.now(timezone.utc),
            event_type=f"processed_{event.event_type}",
            symbol=event.symbol,
            data={**event.data, "processed_by": self.name},
            source=f"test_processor_{self.name}",
            partition_key=event.symbol
        )
        
        return [processed_event]

@pytest.fixture
def sample_tick_event():
    """Create a sample tick event"""
    return StreamEvent(
        event_id="tick_001",
        timestamp=datetime.now(timezone.utc),
        event_type="tick",
        symbol="AAPL",
        data={
            "price": 150.25,
            "volume": 1000,
            "bid": 150.24,
            "ask": 150.26
        },
        source="test_data",
        partition_key="AAPL"
    )

@pytest.fixture
def sample_orderbook_event():
    """Create a sample orderbook event"""
    return StreamEvent(
        event_id="ob_001",
        timestamp=datetime.now(timezone.utc),
        event_type="orderbook",
        symbol="MSFT",
        data={
            "bids": [
                {"price": 300.50, "size": 100},
                {"price": 300.49, "size": 200}
            ],
            "asks": [
                {"price": 300.52, "size": 150},
                {"price": 300.53, "size": 300}
            ]
        },
        source="test_data",
        partition_key="MSFT"
    )

@pytest.fixture
def sample_trade_event():
    """Create a sample trade event"""
    return StreamEvent(
        event_id="trade_001",
        timestamp=datetime.now(timezone.utc),
        event_type="trade",
        symbol="GOOGL",
        data={
            "price": 2500.75,
            "size": 50,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        source="test_data",
        partition_key="GOOGL"
    )

class TestStreamEvent:
    """Test StreamEvent functionality"""
    
    def test_stream_event_creation(self, sample_tick_event):
        """Test basic stream event creation"""
        event = sample_tick_event
        assert event.event_id == "tick_001"
        assert event.symbol == "AAPL"
        assert event.data["price"] == 150.25
    
    def test_stream_event_serialization(self, sample_tick_event):
        """Test event serialization/deserialization"""
        event = sample_tick_event
        
        # Convert to dict
        event_dict = event.to_dict()
        assert isinstance(event_dict["timestamp"], str)
        
        # Convert back to event
        restored_event = StreamEvent.from_dict(event_dict)
        assert restored_event.event_id == event.event_id
        assert restored_event.symbol == event.symbol
        assert restored_event.data == event.data
        assert restored_event.timestamp == event.timestamp

class TestMarketDataProcessor:
    """Test MarketDataProcessor functionality"""
    
    @pytest.mark.asyncio
    async def test_tick_processing(self, sample_tick_event):
        """Test tick data processing"""
        processor = MarketDataProcessor()
        await processor.setup()
        
        # Process first tick
        results = await processor.process(sample_tick_event)
        assert len(results) == 1
        
        result = results[0]
        assert result.event_type == "enriched_tick"
        assert result.symbol == "AAPL"
        assert "price_change" not in result.data  # No previous price
        
        # Process second tick with same symbol
        second_event = StreamEvent(
            event_id="tick_002",
            timestamp=datetime.now(timezone.utc),
            event_type="tick",
            symbol="AAPL",
            data={"price": 151.00, "volume": 500},
            source="test_data"
        )
        
        results = await processor.process(second_event)
        assert len(results) == 1
        
        result = results[0]
        assert result.data["price_change"] == 0.75  # 151.00 - 150.25
        assert result.data["price_change_pct"] == pytest.approx(0.499, abs=0.01)
        
        await processor.teardown()
    
    @pytest.mark.asyncio
    async def test_orderbook_processing(self, sample_orderbook_event):
        """Test orderbook processing"""
        processor = MarketDataProcessor()
        await processor.setup()
        
        results = await processor.process(sample_orderbook_event)
        assert len(results) == 1
        
        result = results[0]
        assert result.event_type == "enriched_orderbook"
        assert result.symbol == "MSFT"
        
        # Check calculated values
        data = result.data
        assert data["best_bid"] == 300.50
        assert data["best_ask"] == 300.52
        assert data["mid_price"] == 300.51
        assert data["spread"] == pytest.approx(0.02, abs=0.0001)
        assert data["spread_pct"] == pytest.approx(0.00665, abs=0.00001)
        
        await processor.teardown()
    
    @pytest.mark.asyncio
    async def test_trade_processing(self, sample_trade_event):
        """Test trade processing"""
        processor = MarketDataProcessor()
        await processor.setup()
        
        # Set up a previous price for trade classification
        processor.last_prices["GOOGL"] = 2500.00
        
        results = await processor.process(sample_trade_event)
        assert len(results) == 1
        
        result = results[0]
        assert result.event_type == "enriched_trade"
        assert result.symbol == "GOOGL"
        assert result.data["side"] == "buy"  # price > last_price
        
        await processor.teardown()

class TestFlinkStreamProcessor:
    """Test FlinkStreamProcessor functionality"""
    
    @pytest.mark.asyncio
    async def test_processor_lifecycle(self):
        """Test starting and stopping the processor"""
        processor = FlinkStreamProcessor("test_stream", parallelism=2)
        
        assert not processor.running
        
        # Start processor
        await processor.start()
        assert processor.running
        assert len(processor.threads) > 0
        
        # Stop processor
        await processor.stop()
        assert not processor.running
        
        # Check metrics were initialized
        metrics = processor.get_metrics()
        assert isinstance(metrics, ProcessingMetrics)
    
    @pytest.mark.asyncio
    async def test_event_processing(self, sample_tick_event):
        """Test basic event processing"""
        processor = FlinkStreamProcessor("test_stream", parallelism=1)
        
        async with processor:
            # Send event
            await processor.send_event(sample_tick_event)
            
            # Wait a bit for processing
            await asyncio.sleep(0.1)
            
            # Get processed event
            processed_event = await processor.get_processed_event(timeout=2.0)
            assert processed_event is not None
            assert processed_event.event_type == "enriched_tick"
            assert processed_event.symbol == sample_tick_event.symbol
    
    @pytest.mark.asyncio
    async def test_custom_function(self, sample_tick_event):
        """Test adding custom processing functions"""
        processor = FlinkStreamProcessor("test_stream", parallelism=1)
        
        # Add custom test function
        test_function = TestStreamFunction("custom_processor")
        processor.add_function(test_function)
        
        async with processor:
            # Send event
            await processor.send_event(sample_tick_event)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Check that custom function was called
            assert len(test_function.processed_events) > 0
            
            # Get processed event
            processed_event = await processor.get_processed_event(timeout=2.0)
            assert processed_event is not None
            assert "processed_by" in processed_event.data
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in processing"""
        
        class ErrorFunction(StreamFunction):
            def __init__(self):
                super().__init__("error_function")
            
            async def process(self, event: StreamEvent):
                raise ValueError("Test error")
        
        processor = FlinkStreamProcessor("test_stream", parallelism=1)
        processor.add_function(ErrorFunction())
        
        async with processor:
            # Send event that will cause error
            error_event = StreamEvent(
                event_id="error_001",
                timestamp=datetime.now(timezone.utc),
                event_type="test",
                symbol="ERROR",
                data={},
                source="test"
            )
            
            await processor.send_event(error_event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check error was captured
            error = await processor.get_error_event(timeout=1.0)
            assert error is not None
            assert error["function"] == "error_function"
            assert "Test error" in error["error"]
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, sample_tick_event):
        """Test metrics tracking"""
        processor = FlinkStreamProcessor("test_stream", parallelism=1)
        
        async with processor:
            # Send multiple events
            for i in range(5):
                event = StreamEvent(
                    event_id=f"tick_{i:03d}",
                    timestamp=datetime.now(timezone.utc),
                    event_type="tick",
                    symbol="TEST",
                    data={"price": 100.0 + i, "volume": 1000},
                    source="test"
                )
                await processor.send_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Check metrics
            metrics = processor.get_metrics()
            assert metrics.processed_events >= 5
            assert metrics.avg_latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_functionality(self):
        """Test checkpoint creation and restoration"""
        processor = FlinkStreamProcessor("test_stream", parallelism=1)
        
        # Add some state
        processor.state["test_key"] = "test_value"
        
        # Create checkpoint manually
        processor._create_checkpoint()
        
        # Check checkpoint was created
        checkpoint_ids = processor.get_checkpoint_ids()
        assert len(checkpoint_ids) > 0
        
        # Modify state
        processor.state["test_key"] = "modified_value"
        
        # Restore from checkpoint
        processor.restore_from_checkpoint(checkpoint_ids[0])
        
        # Check state was restored
        assert processor.state["test_key"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_throughput_performance(self):
        """Test processing throughput"""
        processor = FlinkStreamProcessor("throughput_test", parallelism=4)
        
        async with processor:
            start_time = time.time()
            num_events = 100
            
            # Send events rapidly
            for i in range(num_events):
                event = StreamEvent(
                    event_id=f"perf_{i:04d}",
                    timestamp=datetime.now(timezone.utc),
                    event_type="tick",
                    symbol=f"SYMBOL_{i % 10}",  # 10 different symbols
                    data={"price": 100.0 + (i * 0.01), "volume": 1000},
                    source="perf_test"
                )
                await processor.send_event(event)
            
            # Wait for all events to be processed
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            metrics = processor.get_metrics()
            print(f"Processed {metrics.processed_events} events in {processing_time:.2f} seconds")
            print(f"Average latency: {metrics.avg_latency_ms:.2f} ms")
            print(f"Throughput: {metrics.throughput_per_sec} events/sec")
            
            # Basic performance assertions
            assert metrics.processed_events >= num_events
            assert metrics.avg_latency_ms < 100  # Should be under 100ms average
    
    def test_function_management(self):
        """Test adding and removing functions"""
        processor = FlinkStreamProcessor("function_test")
        
        initial_count = len(processor.functions)
        
        # Add function
        test_func = TestStreamFunction("test_func")
        processor.add_function(test_func)
        assert len(processor.functions) == initial_count + 1
        
        # Remove function
        processor.remove_function("test_func")
        assert len(processor.functions) == initial_count
        
        # Try to remove non-existent function
        processor.remove_function("non_existent")
        assert len(processor.functions) == initial_count

if __name__ == "__main__":
    # Run some basic tests
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    async def run_basic_test():
        print("Running basic Flink Stream Processor test...")
        
        # Create processor
        processor = FlinkStreamProcessor("basic_test", parallelism=2)
        
        async with processor:
            # Create test event
            event = StreamEvent(
                event_id="basic_001",
                timestamp=datetime.now(timezone.utc),
                event_type="tick",
                symbol="AAPL",
                data={"price": 150.00, "volume": 1000},
                source="basic_test"
            )
            
            # Send event
            await processor.send_event(event)
            print(f"Sent event: {event.event_id}")
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Get result
            result = await processor.get_processed_event(timeout=2.0)
            if result:
                print(f"Received processed event: {result.event_id} - {result.event_type}")
                print(f"Data: {result.data}")
            else:
                print("No processed event received")
            
            # Check metrics
            metrics = processor.get_metrics()
            print(f"Metrics: {metrics.processed_events} processed, {metrics.failed_events} failed")
            print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
    
    # Run the basic test
    asyncio.run(run_basic_test())
