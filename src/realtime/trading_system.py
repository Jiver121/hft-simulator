"""
Real-Time Trading System Orchestrator for HFT Simulator

This module provides the main orchestrator that coordinates all real-time
trading components including data feeds, order management, risk controls,
and stream processing for live HFT operations.

Key Features:
- Complete system lifecycle management
- Component coordination and communication
- Real-time market data processing
- Live order execution and management
- Comprehensive risk management
- Performance monitoring and alerting
- Graceful shutdown and error recovery
- Production-ready architecture

System Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Feeds    │───▶│ Stream Processor │───▶│   Strategies    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Market Data     │    │ Risk Manager     │    │ Order Manager   │
│ Manager         │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Monitoring     │    │    Brokers      │
                       │   & Alerting     │    │                 │
                       └──────────────────┘    └─────────────────┘
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from ..utils.logger import get_logger
from .config import RealTimeConfig, ConfigurationManager, get_config_manager
from .data_feeds import RealTimeDataFeed, create_data_feed, MarketDataMessage
from .brokers import BrokerAPI, create_broker, BrokerType
from .order_management import RealTimeOrderManager, OrderRequest, OrderState
from .risk_management import RealTimeRiskManager, RiskViolation
from .stream_processing import StreamProcessor, create_stream_processor, MessageType


class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentState(Enum):
    """Individual component states"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    
    # Throughput metrics
    messages_per_second: float = 0.0
    orders_per_second: float = 0.0
    trades_per_second: float = 0.0
    
    # Latency metrics
    avg_processing_latency_us: float = 0.0
    avg_order_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    
    # System health
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Component status
    active_data_feeds: int = 0
    active_brokers: int = 0
    active_strategies: int = 0
    
    # Trading metrics
    active_orders: int = 0
    total_positions: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Error metrics
    error_count: int = 0
    warning_count: int = 0
    
    # Timestamps
    system_start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate system uptime in seconds"""
        return (datetime.now() - self.system_start_time).total_seconds()


@dataclass
class ComponentStatus:
    """Status of individual system component"""
    
    component_name: str
    component_type: str
    state: ComponentState
    
    # Health metrics
    is_healthy: bool = True
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Performance metrics
    messages_processed: int = 0
    processing_rate: float = 0.0
    error_count: int = 0
    
    # Timestamps
    start_time: Optional[datetime] = None
    last_update: datetime = field(default_factory=datetime.now)


class RealTimeTradingSystem:
    """
    Main real-time trading system orchestrator
    
    Coordinates all components of the real-time HFT trading system,
    manages system lifecycle, monitors health, and ensures reliable operation.
    """
    
    def __init__(self, config: Optional[RealTimeConfig] = None):
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Configuration
        self.config = config
        self.config_manager: Optional[ConfigurationManager] = None
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.shutdown_requested = False
        
        # Core components
        self.data_feeds: Dict[str, RealTimeDataFeed] = {}
        self.brokers: Dict[str, BrokerAPI] = {}
        self.stream_processor: Optional[StreamProcessor] = None
        self.order_manager: Optional[RealTimeOrderManager] = None
        self.risk_manager: Optional[RealTimeRiskManager] = None
        
        # Component status tracking
        self.component_status: Dict[str, ComponentStatus] = {}
        
        # System metrics
        self.metrics = SystemMetrics()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'system_started': [],
            'system_stopped': [],
            'component_error': [],
            'risk_violation': [],
            'order_filled': [],
            'market_data_received': []
        }
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Strategies (placeholder for strategy integration)
        self.strategies: Dict[str, Any] = {}
        
        self.logger.info("RealTimeTradingSystem initialized")
    
    async def initialize(self, config_file: Optional[str] = None) -> None:
        """Initialize the trading system"""
        try:
            self.state = SystemState.INITIALIZING
            self.logger.info("Initializing real-time trading system...")
            
            # Initialize configuration
            if not self.config:
                self.config_manager = get_config_manager()
                if config_file:
                    self.config = await self.config_manager.load_configuration()
                else:
                    # Use default configuration
                    from .config import RealTimeConfig
                    self.config = RealTimeConfig()
            
            # Initialize core components
            await self._initialize_risk_manager()
            await self._initialize_stream_processor()
            await self._initialize_order_manager()
            await self._initialize_data_feeds()
            await self._initialize_brokers()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            self.logger.info("Real-time trading system initialized successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"Failed to initialize trading system: {e}")
            raise
    
    async def _initialize_risk_manager(self) -> None:
        """Initialize risk management system"""
        self.logger.info("Initializing risk manager...")
        
        self.risk_manager = RealTimeRiskManager()
        
        # Add risk violation callback
        self.risk_manager.risk_monitor.add_alert_callback(self._handle_risk_violation)
        
        # Update component status
        self.component_status['risk_manager'] = ComponentStatus(
            component_name='risk_manager',
            component_type='RiskManager',
            state=ComponentState.ACTIVE,
            start_time=datetime.now()
        )
        
        self.logger.info("Risk manager initialized")
    
    async def _initialize_stream_processor(self) -> None:
        """Initialize stream processing system"""
        self.logger.info("Initializing stream processor...")
        
        stream_config = self.config.stream_processing
        self.stream_processor = create_stream_processor(stream_config)
        
        # Update component status
        self.component_status['stream_processor'] = ComponentStatus(
            component_name='stream_processor',
            component_type='StreamProcessor',
            state=ComponentState.ACTIVE,
            start_time=datetime.now()
        )
        
        self.logger.info("Stream processor initialized")
    
    async def _initialize_order_manager(self) -> None:
        """Initialize order management system"""
        self.logger.info("Initializing order manager...")
        
        self.order_manager = RealTimeOrderManager(self.risk_manager)
        
        # Add order event callbacks
        self.order_manager.add_callback('order_filled', self._handle_order_filled)
        self.order_manager.add_callback('order_rejected', self._handle_order_rejected)
        
        # Update component status
        self.component_status['order_manager'] = ComponentStatus(
            component_name='order_manager',
            component_type='OrderManager',
            state=ComponentState.ACTIVE,
            start_time=datetime.now()
        )
        
        self.logger.info("Order manager initialized")
    
    async def _initialize_data_feeds(self) -> None:
        """Initialize market data feeds"""
        self.logger.info("Initializing data feeds...")
        
        for feed_id, feed_config in self.config.data_feeds.items():
            try:
                # Determine feed type from URL or configuration
                feed_type = 'websocket' if feed_config.url.startswith('ws') else 'mock'
                
                # Create data feed
                data_feed = create_data_feed(feed_type, feed_config)
                
                # Add market data callback
                data_feed.add_subscriber(self._handle_market_data)
                
                self.data_feeds[feed_id] = data_feed
                
                # Update component status
                self.component_status[f'data_feed_{feed_id}'] = ComponentStatus(
                    component_name=f'data_feed_{feed_id}',
                    component_type='DataFeed',
                    state=ComponentState.INACTIVE
                )
                
                self.logger.info(f"Data feed initialized: {feed_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize data feed {feed_id}: {e}")
    
    async def _initialize_brokers(self) -> None:
        """Initialize broker connections"""
        self.logger.info("Initializing brokers...")
        
        for broker_id, broker_config in self.config.brokers.items():
            try:
                # Create broker instance
                broker = create_broker(broker_config.broker_type, broker_config)
                
                # Add to order manager
                self.order_manager.add_broker(broker_id, broker)
                
                self.brokers[broker_id] = broker
                
                # Update component status
                self.component_status[f'broker_{broker_id}'] = ComponentStatus(
                    component_name=f'broker_{broker_id}',
                    component_type='Broker',
                    state=ComponentState.INACTIVE
                )
                
                self.logger.info(f"Broker initialized: {broker_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize broker {broker_id}: {e}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != 'win32':
            # Unix/Linux signal handling
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Signal handlers configured")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        
        # Create shutdown task
        asyncio.create_task(self.stop())
    
    async def start(self) -> None:
        """Start the real-time trading system"""
        try:
            self.state = SystemState.STARTING
            self.start_time = datetime.now()
            self.metrics.system_start_time = self.start_time
            
            self.logger.info("Starting real-time trading system...")
            
            # Start core components
            await self._start_risk_manager()
            await self._start_stream_processor()
            await self._start_order_manager()
            await self._start_data_feeds()
            await self._start_brokers()
            
            # Start monitoring
            await self._start_monitoring()
            
            # System is now running
            self.state = SystemState.RUNNING
            
            # Notify callbacks
            await self._notify_callbacks('system_started', self)
            
            self.logger.info("Real-time trading system started successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"Failed to start trading system: {e}")
            raise
    
    async def _start_risk_manager(self) -> None:
        """Start risk management system"""
        if self.risk_manager:
            await self.risk_manager.start_monitoring()
            self.component_status['risk_manager'].state = ComponentState.ACTIVE
            self.logger.info("Risk manager started")
    
    async def _start_stream_processor(self) -> None:
        """Start stream processing system"""
        if self.stream_processor:
            await self.stream_processor.start()
            self.component_status['stream_processor'].state = ComponentState.ACTIVE
            self.logger.info("Stream processor started")
    
    async def _start_order_manager(self) -> None:
        """Start order management system"""
        if self.order_manager:
            await self.order_manager.start()
            self.component_status['order_manager'].state = ComponentState.ACTIVE
            self.logger.info("Order manager started")
    
    async def _start_data_feeds(self) -> None:
        """Start all data feeds"""
        for feed_id, data_feed in self.data_feeds.items():
            try:
                # Connect to data feed
                if await data_feed.connect():
                    # Subscribe to symbols
                    await data_feed.subscribe(data_feed.config.symbols)
                    
                    # Start streaming (in background)
                    asyncio.create_task(self._run_data_feed(feed_id, data_feed))
                    
                    self.component_status[f'data_feed_{feed_id}'].state = ComponentState.ACTIVE
                    self.logger.info(f"Data feed started: {feed_id}")
                else:
                    self.component_status[f'data_feed_{feed_id}'].state = ComponentState.ERROR
                    self.logger.error(f"Failed to connect data feed: {feed_id}")
                    
            except Exception as e:
                self.component_status[f'data_feed_{feed_id}'].state = ComponentState.ERROR
                self.logger.error(f"Error starting data feed {feed_id}: {e}")
    
    async def _start_brokers(self) -> None:
        """Start all broker connections"""
        for broker_id, broker in self.brokers.items():
            try:
                # Connect and authenticate
                if await broker.connect() and await broker.authenticate():
                    self.component_status[f'broker_{broker_id}'].state = ComponentState.ACTIVE
                    self.logger.info(f"Broker started: {broker_id}")
                else:
                    self.component_status[f'broker_{broker_id}'].state = ComponentState.ERROR
                    self.logger.error(f"Failed to connect broker: {broker_id}")
                    
            except Exception as e:
                self.component_status[f'broker_{broker_id}'].state = ComponentState.ERROR
                self.logger.error(f"Error starting broker {broker_id}: {e}")
    
    async def _start_monitoring(self) -> None:
        """Start system monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("System monitoring started")
    
    async def _run_data_feed(self, feed_id: str, data_feed: RealTimeDataFeed) -> None:
        """Run individual data feed streaming"""
        try:
            async for message in data_feed.start_streaming():
                if self.shutdown_requested:
                    break
                
                # Update component metrics
                status = self.component_status[f'data_feed_{feed_id}']
                status.messages_processed += 1
                status.last_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error in data feed {feed_id}: {e}")
            self.component_status[f'data_feed_{feed_id}'].state = ComponentState.ERROR
            self.component_status[f'data_feed_{feed_id}'].error_message = str(e)
    
    async def stop(self) -> None:
        """Stop the real-time trading system"""
        try:
            self.state = SystemState.STOPPING
            self.logger.info("Stopping real-time trading system...")
            
            # Stop monitoring
            await self._stop_monitoring()
            
            # Stop components in reverse order
            await self._stop_data_feeds()
            await self._stop_brokers()
            await self._stop_order_manager()
            await self._stop_stream_processor()
            await self._stop_risk_manager()
            
            # System is now stopped
            self.state = SystemState.STOPPED
            
            # Notify callbacks
            await self._notify_callbacks('system_stopped', self)
            
            self.logger.info("Real-time trading system stopped successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"Error stopping trading system: {e}")
    
    async def _stop_monitoring(self) -> None:
        """Stop system monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System monitoring stopped")
    
    async def _stop_data_feeds(self) -> None:
        """Stop all data feeds"""
        for feed_id, data_feed in self.data_feeds.items():
            try:
                await data_feed.disconnect()
                self.component_status[f'data_feed_{feed_id}'].state = ComponentState.STOPPED
                self.logger.info(f"Data feed stopped: {feed_id}")
            except Exception as e:
                self.logger.error(f"Error stopping data feed {feed_id}: {e}")
    
    async def _stop_brokers(self) -> None:
        """Stop all broker connections"""
        for broker_id, broker in self.brokers.items():
            try:
                await broker.disconnect()
                self.component_status[f'broker_{broker_id}'].state = ComponentState.STOPPED
                self.logger.info(f"Broker stopped: {broker_id}")
            except Exception as e:
                self.logger.error(f"Error stopping broker {broker_id}: {e}")
    
    async def _stop_order_manager(self) -> None:
        """Stop order management system"""
        if self.order_manager:
            await self.order_manager.stop()
            self.component_status['order_manager'].state = ComponentState.STOPPED
            self.logger.info("Order manager stopped")
    
    async def _stop_stream_processor(self) -> None:
        """Stop stream processing system"""
        if self.stream_processor:
            await self.stream_processor.stop()
            self.component_status['stream_processor'].state = ComponentState.STOPPED
            self.logger.info("Stream processor stopped")
    
    async def _stop_risk_manager(self) -> None:
        """Stop risk management system"""
        if self.risk_manager:
            await self.risk_manager.stop_monitoring()
            self.component_status['risk_manager'].state = ComponentState.STOPPED
            self.logger.info("Risk manager stopped")
    
    async def _handle_market_data(self, message: MarketDataMessage) -> None:
        """Handle incoming market data"""
        try:
            # Process through stream processor
            if self.stream_processor:
                await self.stream_processor.process_market_data(message)
            
            # Update metrics
            self.metrics.messages_per_second += 1
            
            # Notify callbacks
            await self._notify_callbacks('market_data_received', message)
            
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
    
    async def _handle_order_filled(self, order_state: OrderState) -> None:
        """Handle order fill event"""
        try:
            self.logger.info(f"Order filled: {order_state.order.order_id}")
            
            # Update metrics
            self.metrics.trades_per_second += 1
            
            # Notify callbacks
            await self._notify_callbacks('order_filled', order_state)
            
        except Exception as e:
            self.logger.error(f"Error handling order fill: {e}")
    
    async def _handle_order_rejected(self, order_state: OrderState) -> None:
        """Handle order rejection event"""
        self.logger.warning(f"Order rejected: {order_state.order.order_id}")
        self.metrics.error_count += 1
    
    async def _handle_risk_violation(self, violation: RiskViolation) -> None:
        """Handle risk violation event"""
        self.logger.warning(f"Risk violation: {violation.violation_type.value}")
        
        # Update metrics
        if violation.risk_level.value in ['high', 'critical']:
            self.metrics.error_count += 1
        else:
            self.metrics.warning_count += 1
        
        # Notify callbacks
        await self._notify_callbacks('risk_violation', violation)
    
    async def _monitoring_loop(self) -> None:
        """Main system monitoring loop"""
        while not self.shutdown_requested:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check component health
                await self._check_component_health()
                
                # Log system status periodically
                if datetime.now().second % 30 == 0:  # Every 30 seconds
                    self._log_system_status()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _health_check_loop(self) -> None:
        """Health check loop for external monitoring"""
        while not self.shutdown_requested:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                # Update component status based on health checks
                for component_name, is_healthy in health_status.items():
                    if component_name in self.component_status:
                        self.component_status[component_name].is_healthy = is_healthy
                        self.component_status[component_name].last_heartbeat = datetime.now()
                
                await asyncio.sleep(10)  # Health check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        try:
            # Get metrics from components
            if self.stream_processor:
                stream_stats = self.stream_processor.get_statistics()
                self.metrics.messages_per_second = stream_stats['processing_stats']['messages_per_second']
                self.metrics.avg_processing_latency_us = stream_stats['processing_stats']['avg_latency_us']
            
            if self.order_manager:
                order_stats = self.order_manager.get_statistics()
                self.metrics.active_orders = order_stats['active_orders']
                
                position_summary = self.order_manager.get_position_summary()
                self.metrics.total_positions = position_summary['active_positions']
                self.metrics.total_pnl = position_summary['total_pnl']
            
            # Count active components
            self.metrics.active_data_feeds = len([
                s for s in self.component_status.values() 
                if s.component_type == 'DataFeed' and s.state == ComponentState.ACTIVE
            ])
            
            self.metrics.active_brokers = len([
                s for s in self.component_status.values() 
                if s.component_type == 'Broker' and s.state == ComponentState.ACTIVE
            ])
            
            self.metrics.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating system metrics: {e}")
    
    async def _check_component_health(self) -> None:
        """Check health of all components"""
        for component_name, status in self.component_status.items():
            try:
                # Check if component has been updated recently
                if status.last_update:
                    time_since_update = datetime.now() - status.last_update
                    if time_since_update > timedelta(minutes=5):
                        status.is_healthy = False
                        status.error_message = "Component not responding"
                
            except Exception as e:
                self.logger.error(f"Error checking health of {component_name}: {e}")
    
    async def _perform_health_checks(self) -> Dict[str, bool]:
        """Perform detailed health checks"""
        health_status = {}
        
        # Check data feeds
        for feed_id, data_feed in self.data_feeds.items():
            try:
                # Simple connectivity check
                health_status[f'data_feed_{feed_id}'] = data_feed.connected
            except Exception:
                health_status[f'data_feed_{feed_id}'] = False
        
        # Check brokers
        for broker_id, broker in self.brokers.items():
            try:
                health_status[f'broker_{broker_id}'] = broker.connected and broker.authenticated
            except Exception:
                health_status[f'broker_{broker_id}'] = False
        
        return health_status
    
    def _log_system_status(self) -> None:
        """Log current system status"""
        active_components = len([
            s for s in self.component_status.values() 
            if s.state == ComponentState.ACTIVE
        ])
        
        self.logger.info(
            f"System Status: {self.state.value} | "
            f"Uptime: {self.metrics.uptime_seconds:.0f}s | "
            f"Active Components: {active_components}/{len(self.component_status)} | "
            f"Messages/s: {self.metrics.messages_per_second:.1f} | "
            f"Active Orders: {self.metrics.active_orders} | "
            f"Total P&L: ${self.metrics.total_pnl:.2f}"
        )
    
    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for system events"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
            self.logger.info(f"Added callback for {event_type}")
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify registered callbacks of events"""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_state': self.state.value,
            'start_time': self.start_time,
            'uptime_seconds': self.metrics.uptime_seconds,
            'metrics': {
                **self.metrics.__dict__,
                # Ensure tests see uptime within the metrics dict as well
                'uptime_seconds': self.metrics.uptime_seconds,
            },
            'component_status': {
                name: {
                    'state': status.state.value,
                    'is_healthy': status.is_healthy,
                    'messages_processed': status.messages_processed,
                    'error_count': status.error_count,
                    'last_update': status.last_update
                }
                for name, status in self.component_status.items()
            }
        }
    
    async def submit_order(self, order_request: OrderRequest) -> str:
        """Submit order through the system"""
        if not self.order_manager:
            raise RuntimeError("Order manager not initialized")
        
        if self.state != SystemState.RUNNING:
            raise RuntimeError(f"System not running (state: {self.state.value})")
        
        return await self.order_manager.submit_order(order_request)
    
    async def run(self) -> None:
        """Run the trading system until shutdown"""
        try:
            await self.start()
            
            # Keep running until shutdown requested
            while not self.shutdown_requested and self.state == SystemState.RUNNING:
                await asyncio.sleep(1)
            
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main run loop: {e}")
        finally:
            await self.stop()


# Convenience function for running the system
async def run_trading_system(config_file: Optional[str] = None) -> None:
    """
    Run the real-time trading system
    
    Args:
        config_file: Optional configuration file path
    """
    system = RealTimeTradingSystem()
    
    try:
        await system.initialize(config_file)
        await system.run()
    except Exception as e:
        logging.error(f"Failed to run trading system: {e}")
        raise


if __name__ == "__main__":
    # Run the trading system
    asyncio.run(run_trading_system())