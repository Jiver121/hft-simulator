"""
Integration Tests for Real-Time Trading System

This module contains comprehensive integration tests for the real-time HFT
trading system, testing the interaction between all components including
data feeds, order management, risk controls, and stream processing.

Test Categories:
- System initialization and startup
- Data feed integration
- Order execution workflow
- Risk management integration
- Stream processing pipeline
- Configuration management
- Error handling and recovery
- Performance benchmarks
"""

import pytest
import pytest_asyncio  # Import pytest_asyncio
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from src.realtime.trading_system import RealTimeTradingSystem, SystemState
from src.realtime.config import RealTimeConfig, Environment, DataFeedConfig, BrokerConfig, BrokerType
from src.realtime.order_management import OrderRequest, OrderPriority, ExecutionAlgorithm
from src.realtime.data_feeds import MarketDataMessage, create_data_feed
from src.realtime.brokers import create_broker
from src.realtime.risk_management import RiskViolation, ViolationType
from src.utils.constants import OrderSide, OrderType


class TestRealTimeTradingSystem:
    """Integration tests for the complete real-time trading system"""
    
    @pytest_asyncio.fixture
    async def test_config(self) -> RealTimeConfig:
        """Create test configuration"""
        config = RealTimeConfig(
            environment=Environment.TESTING,
            debug_mode=True,
            system_id="test-system-001"
        )
        
        # Configure mock data feed
        config.data_feeds["test_feed"] = DataFeedConfig(
            url="mock://test",
            symbols=["AAPL", "MSFT"],
            max_messages_per_second=50,
            buffer_size=1000
        )
        
        # Configure mock broker
        config.brokers["test_broker"] = BrokerConfig(
            broker_type=BrokerType.MOCK,
            api_key="test_key",
            sandbox_mode=True,
            enable_paper_trading=True
        )
        
        # Configure stream processing
        config.stream_processing.update({
            'num_workers': 2,
            'queue_size': 1000,
            'enable_monitoring': True
        })
        
        return config
    
    @pytest_asyncio.fixture
    async def trading_system(self, test_config: RealTimeConfig) -> RealTimeTradingSystem:
        """Create and initialize trading system"""
        print("Initializing trading_system fixture...")
        system = RealTimeTradingSystem(test_config)
        await system.initialize()
        print("trading_system fixture initialized successfully.")
        return system
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, test_config: RealTimeConfig):
        """Test system initialization process"""
        system = RealTimeTradingSystem(test_config)
        
        # Test initialization
        await system.initialize()
        
        # Verify components are initialized
        assert system.risk_manager is not None
        assert system.stream_processor is not None
        assert system.order_manager is not None
        assert len(system.data_feeds) > 0
        assert len(system.brokers) > 0
        
        # Verify component status
        assert 'risk_manager' in system.component_status
        assert 'stream_processor' in system.component_status
        assert 'order_manager' in system.component_status
    
    @pytest.mark.asyncio
    async def test_system_startup_shutdown(self, trading_system: RealTimeTradingSystem):
        """Test system startup and shutdown process"""
        # Test startup
        await trading_system.start()
        
        # Verify system is running
        assert trading_system.state == SystemState.RUNNING
        assert trading_system.start_time is not None
        
        # Wait a moment for components to stabilize
        await asyncio.sleep(1)
        
        # Verify components are active
        active_components = [
            status for status in trading_system.component_status.values()
            if status.state.value == 'active'
        ]
        assert len(active_components) > 0
        
        # Test shutdown
        await trading_system.stop()
        
        # Verify system is stopped
        assert trading_system.state == SystemState.STOPPED
    
    @pytest.mark.asyncio
    async def test_data_feed_integration(self, trading_system: RealTimeTradingSystem):
        """Test data feed integration and message processing"""
        market_data_received = []
        
        def on_market_data(message: MarketDataMessage):
            market_data_received.append(message)
        
        # Add callback
        trading_system.add_event_callback('market_data_received', on_market_data)
        
        # Start system
        await trading_system.start()
        
        # Wait for market data
        await asyncio.sleep(2)
        
        # Verify market data was received
        assert len(market_data_received) > 0
        
        # Verify message structure
        message = market_data_received[0]
        assert message.symbol in ["AAPL", "MSFT"]
        assert message.timestamp is not None
        assert message.message_type is not None
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_order_execution_workflow(self, trading_system: RealTimeTradingSystem):
        """Test complete order execution workflow"""
        order_events = []
        
        def on_order_filled(order_state):
            order_events.append(('filled', order_state))
        
        def on_order_rejected(order_state):
            order_events.append(('rejected', order_state))
        
        # Add callbacks
        trading_system.add_event_callback('order_filled', on_order_filled)
        trading_system.add_event_callback('order_rejected', on_order_rejected)
        
        # Start system
        await trading_system.start()
        
        # Wait for system to stabilize
        await asyncio.sleep(1)
        
        # Submit test order
        order_request = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            execution_algorithm=ExecutionAlgorithm.MARKET,
            priority=OrderPriority.NORMAL,
            strategy_id="test_strategy"
        )
        
        order_id = await trading_system.submit_order(order_request)
        assert order_id is not None
        
        # Wait for order processing
        await asyncio.sleep(2)
        
        # Verify order was processed
        assert len(order_events) > 0
        
        # Check order status
        order_status = await trading_system.order_manager.get_order_status(order_id)
        assert order_status is not None
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system: RealTimeTradingSystem):
        """Test risk management integration"""
        risk_violations = []
        
        def on_risk_violation(violation: RiskViolation):
            risk_violations.append(violation)
        
        # Add callback
        trading_system.add_event_callback('risk_violation', on_risk_violation)
        
        # Start system
        await trading_system.start()
        
        # Wait for system to stabilize
        await asyncio.sleep(1)
        
        # Submit order that should trigger risk violation
        large_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1000000,  # Very large order
            order_type=OrderType.MARKET,
            price=1000000.0,   # Very high price
            execution_algorithm=ExecutionAlgorithm.MARKET,
            priority=OrderPriority.NORMAL,
            strategy_id="test_strategy"
        )
        
        order_id = await trading_system.submit_order(large_order)
        
        # Wait for risk processing
        await asyncio.sleep(2)
        
        # Verify risk violation was triggered
        assert len(risk_violations) > 0
        
        # Verify violation details
        violation = risk_violations[0]
        assert violation.violation_type in [ViolationType.ORDER_SIZE_LIMIT, ViolationType.EXPOSURE_LIMIT]
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_stream_processing_performance(self, trading_system: RealTimeTradingSystem):
        """Test stream processing performance"""
        # Start system
        await trading_system.start()
        
        # Wait for processing to begin
        await asyncio.sleep(3)
        
        # Get stream processor statistics
        if trading_system.stream_processor:
            stats = trading_system.stream_processor.get_statistics()
            
            # Verify processing is occurring
            assert stats['processing_stats']['messages_processed'] > 0
            assert stats['processing_stats']['messages_per_second'] > 0
            
            # Verify latency is reasonable (< 1ms average)
            assert stats['processing_stats']['avg_latency_us'] < 1000
            
            # Verify workers are active
            assert stats['processing_stats']['active_workers'] > 0
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, trading_system: RealTimeTradingSystem):
        """Test system metrics collection"""
        # Start system
        await trading_system.start()
        
        # Wait for metrics to accumulate
        await asyncio.sleep(2)
        
        # Get system status
        status = trading_system.get_system_status()
        
        # Verify metrics structure
        assert 'system_state' in status
        assert 'metrics' in status
        assert 'component_status' in status
        
        # Verify metrics content
        metrics = status['metrics']
        assert 'uptime_seconds' in metrics
        assert 'messages_per_second' in metrics
        assert 'active_orders' in metrics
        
        # Verify component status
        component_status = status['component_status']
        assert len(component_status) > 0
        
        for component_name, component_info in component_status.items():
            assert 'state' in component_info
            assert 'is_healthy' in component_info
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, trading_system: RealTimeTradingSystem):
        """Test error handling and recovery mechanisms"""
        # Start system
        await trading_system.start()
        
        # Wait for system to stabilize
        await asyncio.sleep(1)
        
        # Simulate component error by stopping a data feed
        for feed_id, data_feed in trading_system.data_feeds.items():
            await data_feed.disconnect()
            break
        
        # Wait for error detection
        await asyncio.sleep(2)
        
        # Verify error was detected
        status = trading_system.get_system_status()
        error_components = [
            name for name, info in status['component_status'].items()
            if not info['is_healthy']
        ]
        
        # Should have at least one unhealthy component
        assert len(error_components) > 0
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_configuration_management(self, test_config: RealTimeConfig):
        """Test configuration management"""
        # Test configuration validation
        errors = test_config.validate()
        
        # Should have no errors for test config
        assert len(errors) == 0
        
        # Test configuration updates
        original_debug = test_config.debug_mode
        test_config.debug_mode = not original_debug
        
        # Verify update
        assert test_config.debug_mode != original_debug
        
        # Test environment-specific validation
        prod_config = RealTimeConfig(environment=Environment.PRODUCTION)
        prod_errors = prod_config.validate()
        
        # Production config should have validation errors (missing keys, etc.)
        assert len(prod_errors) > 0


class TestDataFeedIntegration:
    """Integration tests for data feed components"""
    
    @pytest.mark.asyncio
    async def test_mock_data_feed(self):
        """Test mock data feed functionality"""
        from src.realtime.data_feeds import DataFeedConfig, MockDataFeed
        
        config = DataFeedConfig(
            url="mock://test",
            symbols=["TEST"],
            max_messages_per_second=10
        )
        
        feed = MockDataFeed(config)
        messages_received = []
        
        def on_message(message):
            messages_received.append(message)
        
        feed.add_subscriber(on_message)
        
        # Connect and start streaming
        assert await feed.connect()
        
        # Collect messages for a short time
        stream_task = asyncio.create_task(self._collect_messages(feed, 1.0))
        await stream_task
        
        await feed.disconnect()
        
        # Verify messages were received
        assert len(messages_received) > 0
        
        # Verify message structure
        message = messages_received[0]
        assert message.symbol == "TEST"
        assert message.price > 0
        assert message.bid_price is not None
        assert message.ask_price is not None
    
    async def _collect_messages(self, feed, duration: float):
        """Helper to collect messages for specified duration"""
        start_time = time.time()
        async for message in feed.start_streaming():
            if time.time() - start_time > duration:
                break


class TestBrokerIntegration:
    """Integration tests for broker components"""
    
    @pytest.mark.asyncio
    async def test_mock_broker(self):
        """Test mock broker functionality"""
        from src.realtime.brokers import BrokerConfig, BrokerType, MockBroker
        from src.engine.order_types import Order
        
        config = BrokerConfig(
            broker_type=BrokerType.MOCK,
            api_key="test_key"
        )
        
        broker = MockBroker(config)
        
        # Test connection
        assert await broker.connect()
        assert await broker.authenticate()
        
        # Test account info
        account = await broker.get_account_info()
        assert account is not None
        assert account.cash_balance > 0
        
        # Test order submission
        order = Order(
            order_id="test_order_001",
            symbol="TEST",
            side=OrderSide.BUY,
            volume=100,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )
        
        response = await broker.submit_order(order)
        assert response.success
        assert response.broker_order_id is not None
        
        # Test order status
        status_response = await broker.get_order_status(response.broker_order_id)
        assert status_response.success
        
        # Test order cancellation
        cancel_response = await broker.cancel_order(response.broker_order_id)
        assert cancel_response.success
        
        await broker.disconnect()


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_message_processing_throughput(self, trading_system: RealTimeTradingSystem):
        """Benchmark message processing throughput"""
        # Start system
        await trading_system.start()
        
        # Wait for system to warm up
        await asyncio.sleep(2)
        
        # Measure throughput over 5 seconds
        start_time = time.time()
        initial_stats = trading_system.stream_processor.get_statistics()
        initial_processed = initial_stats['processing_stats']['messages_processed']
        
        await asyncio.sleep(5)
        
        end_time = time.time()
        final_stats = trading_system.stream_processor.get_statistics()
        final_processed = final_stats['processing_stats']['messages_processed']
        
        # Calculate throughput
        duration = end_time - start_time
        messages_processed = final_processed - initial_processed
        throughput = messages_processed / duration
        
        # Verify minimum throughput (should process at least 10 msg/sec)
        assert throughput >= 10
        
        print(f"Message processing throughput: {throughput:.1f} msg/sec")
        
        await trading_system.stop()
    
    @pytest.mark.asyncio
    async def test_order_processing_latency(self, trading_system: RealTimeTradingSystem):
        """Benchmark order processing latency"""
        order_latencies = []
        
        def on_order_filled(order_state):
            # Calculate latency from order creation to fill
            if order_state.created_time and order_state.last_update:
                latency = (order_state.last_update - order_state.created_time).total_seconds() * 1000
                order_latencies.append(latency)
        
        trading_system.add_event_callback('order_filled', on_order_filled)
        
        # Start system
        await trading_system.start()
        await asyncio.sleep(1)
        
        # Submit multiple orders
        for i in range(10):
            order_request = OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.MARKET,
                execution_algorithm=ExecutionAlgorithm.MARKET,
                priority=OrderPriority.NORMAL,
                strategy_id=f"benchmark_{i}"
            )
            
            await trading_system.submit_order(order_request)
            await asyncio.sleep(0.1)  # Small delay between orders
        
        # Wait for orders to process
        await asyncio.sleep(3)
        
        await trading_system.stop()
        
        # Analyze latencies
        if order_latencies:
            avg_latency = sum(order_latencies) / len(order_latencies)
            max_latency = max(order_latencies)
            
            print(f"Order processing - Avg: {avg_latency:.1f}ms, Max: {max_latency:.1f}ms")
            
            # Verify reasonable latency (< 100ms average)
            assert avg_latency < 100
            assert max_latency < 500


# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from some loggers during tests
    logging.getLogger('src.realtime.stream_processing').setLevel(logging.WARNING)
    logging.getLogger('src.realtime.data_feeds').setLevel(logging.WARNING)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])