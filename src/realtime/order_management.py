"""
Real-Time Order Management System for HFT Simulator

This module provides a comprehensive order management system for live trading,
including order routing, execution tracking, risk controls, and portfolio management.

Key Features:
- Real-time order execution and tracking
- Pre-trade and post-trade risk controls
- Position and portfolio management
- Order routing to multiple brokers
- Comprehensive audit trail
- Performance monitoring
- Automatic risk limit enforcement

Components:
- RealTimeOrderManager: Main order management orchestrator
- OrderRouter: Routes orders to appropriate brokers
- PositionManager: Tracks positions and P&L
- RiskController: Enforces risk limits and controls
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.engine.order_types import Order, Trade
from src.realtime.brokers import BrokerAPI, OrderResponse, Account
from src.realtime.types import OrderRequest, Position, RiskViolation, OrderPriority, ExecutionAlgorithm

# Forward reference to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .risk_management import RealTimeRiskManager


@dataclass
class ExecutionReport:
    """Execution report for order fills"""
    
    order_id: str
    execution_id: str
    symbol: str
    side: OrderSide
    
    # Execution details
    fill_quantity: float
    fill_price: float
    remaining_quantity: float
    
    # Timing
    execution_time: datetime
    
    # Costs
    commission: float = 0.0
    fees: float = 0.0
    
    # Market data
    market_price: Optional[float] = None
    slippage: Optional[float] = None
    
    # Metadata
    broker_execution_id: Optional[str] = None
    venue: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderState:
    """Complete state of an order in the system"""
    
    request: OrderRequest
    order: Order
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Execution tracking
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    executions: List[ExecutionReport] = field(default_factory=list)
    
    # Broker information
    broker_order_id: Optional[str] = None
    broker_responses: List[OrderResponse] = field(default_factory=list)
    
    # Risk and validation
    risk_checks_passed: bool = False
    risk_violations: List[RiskViolation] = field(default_factory=list)
    
    # Performance metrics
    total_commission: float = 0.0
    total_fees: float = 0.0
    realized_slippage: float = 0.0
    
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to fill"""
        return max(0, self.request.quantity - self.filled_quantity)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.remaining_quantity <= 0
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate as percentage"""
        if self.request.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.request.quantity) * 100


class OrderRouter:
    """
    Routes orders to appropriate brokers based on various criteria
    """
    
    def __init__(self):
        self.logger = get_logger(f"{self.__class__.__name__}")
        self.brokers: Dict[str, BrokerAPI] = {}
        self.routing_rules: Dict[str, Any] = {}
        
        # Default routing configuration
        self.default_broker = None
        self.symbol_routing: Dict[str, str] = {}  # symbol -> broker_id
        self.order_type_routing: Dict[OrderType, str] = {}
        
    def add_broker(self, broker_id: str, broker: BrokerAPI) -> None:
        """Add broker to routing table"""
        self.brokers[broker_id] = broker
        if self.default_broker is None:
            self.default_broker = broker_id
        self.logger.info(f"Added broker: {broker_id}")
    
    def set_symbol_routing(self, symbol: str, broker_id: str) -> None:
        """Set specific broker for symbol"""
        if broker_id not in self.brokers:
            raise ValueError(f"Unknown broker: {broker_id}")
        self.symbol_routing[symbol] = broker_id
        self.logger.info(f"Set routing: {symbol} -> {broker_id}")
    
    def route_order(self, order_request: OrderRequest) -> Optional[str]:
        """
        Determine which broker should handle the order
        
        Args:
            order_request: Order to route
            
        Returns:
            Broker ID or None if no suitable broker found
        """
        # Check symbol-specific routing
        if order_request.symbol in self.symbol_routing:
            broker_id = self.symbol_routing[order_request.symbol]
            if broker_id in self.brokers:
                return broker_id
        
        # Check order type routing
        if order_request.order_type in self.order_type_routing:
            broker_id = self.order_type_routing[order_request.order_type]
            if broker_id in self.brokers:
                return broker_id
        
        # Use default broker
        return self.default_broker
    
    def get_broker(self, broker_id: str) -> Optional[BrokerAPI]:
        """Get broker instance by ID"""
        return self.brokers.get(broker_id)


class PositionManager:
    """
    Manages positions and portfolio state in real-time
    """
    
    def __init__(self):
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.daily_pnl = 0.0
        
        # Risk metrics
        self.max_position_value = 0.0
        self.total_exposure = 0.0
        
        # Performance tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def update_position(self, execution: ExecutionReport) -> None:
        """Update position based on execution"""
        symbol = execution.symbol
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                average_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                side='flat',
                opened_at=execution.execution_time,
                updated_at=execution.execution_time
            )
        
        position = self.positions[symbol]
        
        # Calculate new position
        old_quantity = position.quantity
        fill_quantity = execution.fill_quantity
        
        if execution.side == OrderSide.BUY:
            new_quantity = old_quantity + fill_quantity
        else:
            new_quantity = old_quantity - fill_quantity
        
        # Update position
        if new_quantity == 0:
            # Position closed - calculate realized P&L
            if old_quantity != 0:
                realized_pnl = (execution.fill_price - position.average_price) * fill_quantity
                if execution.side == OrderSide.SELL:
                    realized_pnl = -realized_pnl
                
                position.realized_pnl += realized_pnl
                self.realized_pnl += realized_pnl
                
                if realized_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            position.quantity = 0.0
            position.average_price = 0.0
            position.side = 'flat'
        
        else:
            # Update average price
            if (old_quantity >= 0 and new_quantity >= 0) or (old_quantity <= 0 and new_quantity <= 0):
                # Same side - update average price
                total_cost = (old_quantity * position.average_price) + (fill_quantity * execution.fill_price)
                position.average_price = total_cost / new_quantity if new_quantity != 0 else 0.0
            else:
                # Crossing zero - use new fill price
                position.average_price = execution.fill_price
            
            position.quantity = new_quantity
            position.side = 'long' if new_quantity > 0 else 'short'
        
        position.updated_at = execution.execution_time
        self.trade_count += 1
        
        self.logger.debug(f"Updated position {symbol}: {position.quantity}@{position.average_price:.4f}")
    
    def calculate_unrealized_pnl(self, market_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L based on current market prices"""
        total_unrealized = 0.0
        
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in market_prices:
                market_price = market_prices[symbol]
                unrealized = (market_price - position.average_price) * position.quantity
                position.unrealized_pnl = unrealized
                position.market_value = position.quantity * market_price
                total_unrealized += unrealized
        
        self.unrealized_pnl = total_unrealized
        return total_unrealized
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        active_positions = {k: v for k, v in self.positions.items() if v.quantity != 0}
        
        return {
            'active_positions': len(active_positions),
            'total_positions': len(self.positions),
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.realized_pnl + self.unrealized_pnl,
            'trade_count': self.trade_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(self.trade_count, 1) * 100,
            'positions': {k: v for k, v in active_positions.items()}
        }


class RealTimeOrderManager:
    """
    Main order management system coordinating all order lifecycle activities
    """
    
    def __init__(self, risk_manager: 'RealTimeRiskManager'):
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Core components
        self.risk_manager = risk_manager
        self.order_router = OrderRouter()
        self.position_manager = PositionManager()
        
        # Order tracking
        self.active_orders: Dict[str, OrderState] = {}
        self.completed_orders: Dict[str, OrderState] = {}
        self.order_queue: asyncio.Queue = asyncio.Queue()
        
        # Event callbacks
        self.order_callbacks: Dict[str, List[Callable]] = {
            'order_submitted': [],
            'order_filled': [],
            'order_cancelled': [],
            'order_rejected': [],
            'execution_received': []
        }
        
        # Performance tracking
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'average_fill_time': 0.0,
            'last_update': None
        }
        
        # Processing control
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
    
    def add_broker(self, broker_id: str, broker: BrokerAPI) -> None:
        """Add broker to the order management system"""
        self.order_router.add_broker(broker_id, broker)
        self.logger.info(f"Added broker to order manager: {broker_id}")
    
    def add_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for order events"""
        if event_type in self.order_callbacks:
            self.order_callbacks[event_type].append(callback)
            self.logger.info(f"Added callback for {event_type}")
    
    async def start(self) -> None:
        """Start the order management system"""
        if self.running:
            return
        
        self.running = True
        self.processing_task = asyncio.create_task(self._process_orders())
        self.logger.info("Order management system started")
    
    async def stop(self) -> None:
        """Stop the order management system"""
        self.running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Order management system stopped")
    
    async def submit_order(self, order_request: OrderRequest) -> str:
        """
        Submit order for execution
        
        Args:
            order_request: Order request to submit
            
        Returns:
            Order ID for tracking
        """
        # Create order ID
        order_id = order_request.client_order_id or str(uuid.uuid4())
        
        # Create internal order object
        order = Order(
            order_id=order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            volume=int(order_request.quantity),
            order_type=order_request.order_type,
            price=order_request.price,
            timestamp=pd.Timestamp.now()
        )
        
        # Create order state
        order_state = OrderState(
            request=order_request,
            order=order,
            status=OrderStatus.PENDING
        )
        
        # Pre-trade risk checks
        risk_result = await self.risk_manager.validate_order(order_request, self.position_manager.positions)
        
        if not risk_result.approved:
            order_state.status = OrderStatus.REJECTED
            order_state.risk_violations = risk_result.violations
            self.completed_orders[order_id] = order_state
            
            await self._notify_callbacks('order_rejected', order_state)
            
            self.stats['orders_rejected'] += 1
            self.logger.warning(f"Order rejected by risk manager: {order_id}")
            return order_id
        
        order_state.risk_checks_passed = True
        self.active_orders[order_id] = order_state
        
        # Add to processing queue
        await self.order_queue.put(order_state)
        
        self.logger.info(f"Order submitted: {order_id} - {order_request.symbol} {order_request.side} {order_request.quantity}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel active order"""
        if order_id not in self.active_orders:
            self.logger.warning(f"Cannot cancel order - not found: {order_id}")
            return False
        
        order_state = self.active_orders[order_id]
        
        # Route to appropriate broker
        broker_id = self.order_router.route_order(order_state.request)
        if not broker_id:
            self.logger.error(f"No broker available for cancellation: {order_id}")
            return False
        
        broker = self.order_router.get_broker(broker_id)
        if not broker:
            self.logger.error(f"Broker not found: {broker_id}")
            return False
        
        # Send cancellation to broker
        if order_state.broker_order_id:
            response = await broker.cancel_order(order_state.broker_order_id)
            order_state.broker_responses.append(response)
            
            if response.success:
                order_state.status = OrderStatus.CANCELLED
                self._move_to_completed(order_id)
                
                await self._notify_callbacks('order_cancelled', order_state)
                
                self.stats['orders_cancelled'] += 1
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id} - {response.message}")
                return False
        
        return False
    
    async def get_order_status(self, order_id: str) -> Optional[OrderState]:
        """Get current order status"""
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        else:
            return None
    
    def get_active_orders(self) -> Dict[str, OrderState]:
        """Get all active orders"""
        return self.active_orders.copy()
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary"""
        return self.position_manager.get_position_summary()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order management statistics"""
        return {
            **self.stats,
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'queue_size': self.order_queue.qsize(),
            'position_summary': self.position_manager.get_position_summary()
        }
    
    async def _process_orders(self) -> None:
        """Main order processing loop"""
        while self.running:
            try:
                # Get next order from queue
                order_state = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                
                # Process the order
                await self._execute_order(order_state)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing orders: {e}")
                await asyncio.sleep(1)
    
    async def _execute_order(self, order_state: OrderState) -> None:
        """Execute individual order"""
        order_id = order_state.order.order_id
        
        try:
            # Route to broker
            broker_id = self.order_router.route_order(order_state.request)
            if not broker_id:
                await self._handle_order_error(order_state, "No broker available")
                return
            
            broker = self.order_router.get_broker(broker_id)
            if not broker:
                await self._handle_order_error(order_state, f"Broker not found: {broker_id}")
                return
            
            # Submit to broker
            response = await broker.submit_order(order_state.order)
            order_state.broker_responses.append(response)
            
            if response.success:
                order_state.broker_order_id = response.broker_order_id
                order_state.status = OrderStatus.SUBMITTED
                
                await self._notify_callbacks('order_submitted', order_state)
                
                self.stats['orders_submitted'] += 1
                self.logger.info(f"Order submitted to broker: {order_id}")
                
                # Start monitoring for fills
                asyncio.create_task(self._monitor_order_fills(order_state, broker))
                
            else:
                await self._handle_order_error(order_state, response.message or "Broker rejected order")
        
        except Exception as e:
            await self._handle_order_error(order_state, str(e))
    
    async def _monitor_order_fills(self, order_state: OrderState, broker: BrokerAPI) -> None:
        """Monitor order for fills and updates"""
        order_id = order_state.order.order_id
        
        while order_id in self.active_orders and not order_state.is_complete:
            try:
                # Check order status
                if order_state.broker_order_id:
                    response = await broker.get_order_status(order_state.broker_order_id)
                    
                    if response.success and response.status:
                        old_status = order_state.status
                        order_state.status = response.status
                        order_state.last_update = datetime.now()
                        
                        # Handle status changes
                        if response.status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
                            await self._handle_order_fill(order_state, response)
                        elif response.status == OrderStatus.PARTIALLY_FILLED:
                            await self._handle_partial_fill(order_state, response)
                        elif response.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                            self._move_to_completed(order_id)
                            break
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _handle_order_fill(self, order_state: OrderState, response: OrderResponse) -> None:
        """Handle complete order fill"""
        order_id = order_state.order.order_id
        
        # Create execution report
        execution = ExecutionReport(
            order_id=order_id,
            execution_id=str(uuid.uuid4()),
            symbol=order_state.request.symbol,
            side=order_state.request.side,
            fill_quantity=response.quantity or order_state.request.quantity,
            fill_price=response.price or order_state.request.price or 0.0,
            remaining_quantity=0.0,
            execution_time=datetime.now(),
            broker_execution_id=response.broker_order_id
        )
        
        # Update order state
        order_state.executions.append(execution)
        order_state.filled_quantity = execution.fill_quantity
        order_state.average_fill_price = execution.fill_price
        
        # Update positions
        self.position_manager.update_position(execution)
        
        # Move to completed
        self._move_to_completed(order_id)
        
        # Notify callbacks
        await self._notify_callbacks('order_filled', order_state)
        await self._notify_callbacks('execution_received', execution)
        
        # Update statistics
        self.stats['orders_filled'] += 1
        self.stats['total_volume'] += execution.fill_quantity
        self.stats['last_update'] = datetime.now()
        
        self.logger.info(f"Order filled: {order_id} - {execution.fill_quantity}@{execution.fill_price:.4f}")
    
    async def _handle_partial_fill(self, order_state: OrderState, response: OrderResponse) -> None:
        """Handle partial order fill"""
        # Implementation for partial fills would go here
        # For now, treat as complete fill
        await self._handle_order_fill(order_state, response)
    
    async def _handle_order_error(self, order_state: OrderState, error_message: str) -> None:
        """Handle order execution error"""
        order_id = order_state.order.order_id
        
        order_state.status = OrderStatus.REJECTED
        order_state.last_update = datetime.now()
        
        self._move_to_completed(order_id)
        
        await self._notify_callbacks('order_rejected', order_state)
        
        self.stats['orders_rejected'] += 1
        self.logger.error(f"Order error {order_id}: {error_message}")
    
    def _move_to_completed(self, order_id: str) -> None:
        """Move order from active to completed"""
        if order_id in self.active_orders:
            order_state = self.active_orders.pop(order_id)
            self.completed_orders[order_id] = order_state
    
    async def _notify_callbacks(self, event_type: str, data: Any) -> None:
        """Notify registered callbacks of events"""
        for callback in self.order_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")