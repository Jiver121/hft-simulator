"""
CQRS (Command Query Responsibility Segregation) implementation for HFT system.
Separates read and write operations with event sourcing integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type, Generic, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum

from .event_store import Event, EventType, EventStore, EventBus, InMemoryEventStore

logger = logging.getLogger(__name__)

# Type variables for generics
TCommand = TypeVar('TCommand')
TQuery = TypeVar('TQuery')
TAggregate = TypeVar('TAggregate')


class CommandResult:
    """Result of command execution"""
    
    def __init__(self, success: bool, events: List[Event] = None, error: str = None, aggregate_id: str = None):
        self.success = success
        self.events = events or []
        self.error = error
        self.aggregate_id = aggregate_id
    
    @classmethod
    def success_result(cls, events: List[Event], aggregate_id: str = None) -> 'CommandResult':
        """Create successful result"""
        return cls(True, events, aggregate_id=aggregate_id)
    
    @classmethod
    def failure_result(cls, error: str) -> 'CommandResult':
        """Create failure result"""
        return cls(False, error=error)


# Base command and query classes
@dataclass
class Command:
    """Base command class"""
    command_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Query:
    """Base query class"""
    query_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Trading-specific commands
@dataclass
class CreateOrderCommand(Command):
    """Command to create new order"""
    order_id: str
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    price: Optional[float] = None
    order_type: str = "LIMIT"
    strategy_id: Optional[str] = None


@dataclass
class CancelOrderCommand(Command):
    """Command to cancel order"""
    order_id: str
    reason: Optional[str] = None


@dataclass
class ModifyOrderCommand(Command):
    """Command to modify order"""
    order_id: str
    new_quantity: Optional[float] = None
    new_price: Optional[float] = None


@dataclass
class ExecuteTradeCommand(Command):
    """Command to execute trade"""
    execution_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float = 0.0


@dataclass
class UpdatePositionCommand(Command):
    """Command to update position"""
    symbol: str
    quantity_delta: float
    price: float
    strategy_id: Optional[str] = None


# Trading-specific queries
@dataclass
class GetOrderQuery(Query):
    """Query to get order details"""
    order_id: str


@dataclass
class GetOrdersBySymbolQuery(Query):
    """Query to get orders by symbol"""
    symbol: str
    status: Optional[str] = None


@dataclass
class GetPositionQuery(Query):
    """Query to get position"""
    symbol: str
    strategy_id: Optional[str] = None


@dataclass
class GetPortfolioQuery(Query):
    """Query to get portfolio summary"""
    strategy_id: Optional[str] = None


@dataclass
class GetPnLQuery(Query):
    """Query to get P&L data"""
    symbol: Optional[str] = None
    strategy_id: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None


# Command and query handlers
class CommandHandler(ABC, Generic[TCommand]):
    """Abstract base class for command handlers"""
    
    @abstractmethod
    async def handle(self, command: TCommand) -> CommandResult:
        """Handle command and return result"""
        pass


class QueryHandler(ABC, Generic[TQuery]):
    """Abstract base class for query handlers"""
    
    @abstractmethod
    async def handle(self, query: TQuery) -> Any:
        """Handle query and return result"""
        pass


# Aggregate base class
class Aggregate:
    """Base class for aggregates in event sourcing"""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events: List[Event] = []
    
    def apply_event(self, event: Event):
        """Apply event to aggregate"""
        self.version = event.version
        self._handle_event(event)
    
    def _handle_event(self, event: Event):
        """Override in subclasses to handle specific events"""
        pass
    
    def add_event(self, event_type: EventType, event_data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Add new event to uncommitted events"""
        event = Event(
            event_id=str(__import__('uuid').uuid4()),
            event_type=event_type,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.__class__.__name__,
            event_data=event_data,
            timestamp=datetime.now(timezone.utc),
            version=self.version + 1,
            metadata=metadata
        )
        
        self.uncommitted_events.append(event)
        self.apply_event(event)
    
    def get_uncommitted_events(self) -> List[Event]:
        """Get uncommitted events"""
        return self.uncommitted_events.copy()
    
    def mark_events_as_committed(self):
        """Mark events as committed"""
        self.uncommitted_events.clear()


# Trading aggregates
class OrderAggregate(Aggregate):
    """Order aggregate for order management"""
    
    def __init__(self, order_id: str):
        super().__init__(order_id)
        self.symbol = None
        self.side = None
        self.quantity = 0.0
        self.price = None
        self.order_type = "LIMIT"
        self.status = "NEW"
        self.filled_quantity = 0.0
        self.remaining_quantity = 0.0
        self.created_time = None
        self.strategy_id = None
    
    def create_order(self, symbol: str, side: str, quantity: float, 
                    price: float = None, order_type: str = "LIMIT", 
                    strategy_id: str = None):
        """Create new order"""
        if self.status != "NEW" or self.symbol is not None:
            raise ValueError("Order already created")
        
        self.add_event(EventType.ORDER_CREATED, {
            "order_id": self.aggregate_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "strategy_id": strategy_id
        })
    
    def cancel_order(self, reason: str = None):
        """Cancel order"""
        if self.status not in ["NEW", "PARTIALLY_FILLED"]:
            raise ValueError(f"Cannot cancel order in status: {self.status}")
        
        self.add_event(EventType.ORDER_CANCELLED, {
            "order_id": self.aggregate_id,
            "reason": reason,
            "cancelled_quantity": self.remaining_quantity
        })
    
    def fill_order(self, quantity: float, price: float, execution_id: str):
        """Fill order (partial or complete)"""
        if self.status not in ["NEW", "PARTIALLY_FILLED"]:
            raise ValueError(f"Cannot fill order in status: {self.status}")
        
        if quantity > self.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining quantity")
        
        if quantity == self.remaining_quantity:
            self.add_event(EventType.ORDER_FILLED, {
                "order_id": self.aggregate_id,
                "execution_id": execution_id,
                "quantity": quantity,
                "price": price,
                "total_filled": self.filled_quantity + quantity
            })
        else:
            self.add_event(EventType.ORDER_PARTIALLY_FILLED, {
                "order_id": self.aggregate_id,
                "execution_id": execution_id,
                "quantity": quantity,
                "price": price,
                "total_filled": self.filled_quantity + quantity,
                "remaining": self.remaining_quantity - quantity
            })
    
    def _handle_event(self, event: Event):
        """Handle events to update state"""
        if event.event_type == EventType.ORDER_CREATED:
            data = event.event_data
            self.symbol = data["symbol"]
            self.side = data["side"]
            self.quantity = data["quantity"]
            self.price = data["price"]
            self.order_type = data["order_type"]
            self.strategy_id = data.get("strategy_id")
            self.remaining_quantity = self.quantity
            self.created_time = event.timestamp
            
        elif event.event_type == EventType.ORDER_CANCELLED:
            self.status = "CANCELLED"
            
        elif event.event_type == EventType.ORDER_FILLED:
            data = event.event_data
            fill_qty = data["quantity"]
            self.filled_quantity += fill_qty
            self.remaining_quantity -= fill_qty
            self.status = "FILLED"
            
        elif event.event_type == EventType.ORDER_PARTIALLY_FILLED:
            data = event.event_data
            fill_qty = data["quantity"]
            self.filled_quantity += fill_qty
            self.remaining_quantity -= fill_qty
            self.status = "PARTIALLY_FILLED"


class PositionAggregate(Aggregate):
    """Position aggregate for position management"""
    
    def __init__(self, position_id: str):
        super().__init__(position_id)
        self.symbol = None
        self.quantity = 0.0
        self.average_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.last_update_time = None
        self.strategy_id = None
    
    def update_position(self, symbol: str, quantity_delta: float, price: float, strategy_id: str = None):
        """Update position with new trade"""
        if self.symbol is None:
            self.symbol = symbol
            self.strategy_id = strategy_id
        elif self.symbol != symbol:
            raise ValueError("Symbol mismatch")
        
        self.add_event(EventType.POSITION_MODIFIED, {
            "symbol": symbol,
            "quantity_delta": quantity_delta,
            "price": price,
            "previous_quantity": self.quantity,
            "previous_average_price": self.average_price,
            "strategy_id": strategy_id
        })
    
    def close_position(self, price: float):
        """Close entire position"""
        if self.quantity == 0:
            raise ValueError("No position to close")
        
        self.add_event(EventType.POSITION_CLOSED, {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": price,
            "realized_pnl": (price - self.average_price) * self.quantity
        })
    
    def _handle_event(self, event: Event):
        """Handle events to update position state"""
        if event.event_type == EventType.POSITION_MODIFIED:
            data = event.event_data
            quantity_delta = data["quantity_delta"]
            price = data["price"]
            
            if self.symbol is None:
                self.symbol = data["symbol"]
                self.strategy_id = data.get("strategy_id")
            
            # Update average price using weighted average
            if self.quantity == 0:
                self.average_price = price
            elif (self.quantity > 0 and quantity_delta > 0) or (self.quantity < 0 and quantity_delta < 0):
                # Same direction - update average price
                total_cost = self.quantity * self.average_price + quantity_delta * price
                self.quantity += quantity_delta
                if self.quantity != 0:
                    self.average_price = total_cost / self.quantity
            else:
                # Opposite direction - realize P&L
                if abs(quantity_delta) <= abs(self.quantity):
                    # Partial close
                    realized_pnl = quantity_delta * (price - self.average_price)
                    self.realized_pnl += realized_pnl
                    self.quantity += quantity_delta
                else:
                    # Full close and reverse
                    realized_pnl = -self.quantity * (price - self.average_price)
                    self.realized_pnl += realized_pnl
                    remaining_delta = quantity_delta + self.quantity
                    self.quantity = remaining_delta
                    self.average_price = price if remaining_delta != 0 else 0.0
            
            self.last_update_time = event.timestamp
            
        elif event.event_type == EventType.POSITION_CLOSED:
            data = event.event_data
            self.realized_pnl += data["realized_pnl"]
            self.quantity = 0.0
            self.average_price = 0.0
            self.last_update_time = event.timestamp


# Command handlers
class CreateOrderCommandHandler(CommandHandler[CreateOrderCommand]):
    """Handler for create order commands"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, command: CreateOrderCommand) -> CommandResult:
        """Handle create order command"""
        try:
            # Create order aggregate
            order = OrderAggregate(command.order_id)
            
            # Execute business logic
            order.create_order(
                command.symbol,
                command.side,
                command.quantity,
                command.price,
                command.order_type,
                command.strategy_id
            )
            
            # Save events
            events = order.get_uncommitted_events()
            for event in events:
                await self.event_store.append_event(event)
            
            order.mark_events_as_committed()
            
            return CommandResult.success_result(events, command.order_id)
            
        except Exception as e:
            logger.error(f"Error handling create order command: {e}")
            return CommandResult.failure_result(str(e))


class CancelOrderCommandHandler(CommandHandler[CancelOrderCommand]):
    """Handler for cancel order commands"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, command: CancelOrderCommand) -> CommandResult:
        """Handle cancel order command"""
        try:
            # Load order aggregate from events
            order_events = await self.event_store.get_events(command.order_id)
            if not order_events:
                return CommandResult.failure_result("Order not found")
            
            order = OrderAggregate(command.order_id)
            for event in order_events:
                order.apply_event(event)
            
            # Execute business logic
            order.cancel_order(command.reason)
            
            # Save events
            events = order.get_uncommitted_events()
            for event in events:
                await self.event_store.append_event(event)
            
            order.mark_events_as_committed()
            
            return CommandResult.success_result(events, command.order_id)
            
        except Exception as e:
            logger.error(f"Error handling cancel order command: {e}")
            return CommandResult.failure_result(str(e))


class ExecuteTradeCommandHandler(CommandHandler[ExecuteTradeCommand]):
    """Handler for execute trade commands"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, command: ExecuteTradeCommand) -> CommandResult:
        """Handle execute trade command"""
        try:
            # Load order aggregate
            order_events = await self.event_store.get_events(command.order_id)
            if not order_events:
                return CommandResult.failure_result("Order not found")
            
            order = OrderAggregate(command.order_id)
            for event in order_events:
                order.apply_event(event)
            
            # Execute order fill
            order.fill_order(command.quantity, command.price, command.execution_id)
            
            # Save order events
            events = order.get_uncommitted_events()
            for event in events:
                await self.event_store.append_event(event)
            
            order.mark_events_as_committed()
            
            # Update position
            position_id = f"{command.symbol}_{order.strategy_id or 'default'}"
            position_events = await self.event_store.get_events(position_id)
            
            position = PositionAggregate(position_id)
            for event in position_events:
                position.apply_event(event)
            
            # Calculate position delta
            quantity_delta = command.quantity if order.side == "BUY" else -command.quantity
            position.update_position(command.symbol, quantity_delta, command.price, order.strategy_id)
            
            # Save position events
            position_events = position.get_uncommitted_events()
            for event in position_events:
                await self.event_store.append_event(event)
            
            position.mark_events_as_committed()
            
            # Create trade execution event
            trade_event = Event(
                event_id=command.execution_id,
                event_type=EventType.TRADE_EXECUTED,
                aggregate_id=command.execution_id,
                aggregate_type="Trade",
                event_data={
                    "execution_id": command.execution_id,
                    "order_id": command.order_id,
                    "symbol": command.symbol,
                    "side": command.side,
                    "quantity": command.quantity,
                    "price": command.price,
                    "commission": command.commission,
                    "notional": command.quantity * command.price
                },
                timestamp=datetime.now(timezone.utc),
                version=1
            )
            
            await self.event_store.append_event(trade_event)
            events.extend(position_events)
            events.append(trade_event)
            
            return CommandResult.success_result(events, command.order_id)
            
        except Exception as e:
            logger.error(f"Error handling execute trade command: {e}")
            return CommandResult.failure_result(str(e))


# Query handlers (read models)
class GetOrderQueryHandler(QueryHandler[GetOrderQuery]):
    """Handler for get order queries"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, query: GetOrderQuery) -> Optional[Dict[str, Any]]:
        """Handle get order query"""
        try:
            # Load order from events
            order_events = await self.event_store.get_events(query.order_id)
            if not order_events:
                return None
            
            order = OrderAggregate(query.order_id)
            for event in order_events:
                order.apply_event(event)
            
            return {
                "order_id": order.aggregate_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "order_type": order.order_type,
                "status": order.status,
                "filled_quantity": order.filled_quantity,
                "remaining_quantity": order.remaining_quantity,
                "created_time": order.created_time.isoformat() if order.created_time else None,
                "strategy_id": order.strategy_id
            }
            
        except Exception as e:
            logger.error(f"Error handling get order query: {e}")
            return None


class GetPositionQueryHandler(QueryHandler[GetPositionQuery]):
    """Handler for get position queries"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, query: GetPositionQuery) -> Optional[Dict[str, Any]]:
        """Handle get position query"""
        try:
            position_id = f"{query.symbol}_{query.strategy_id or 'default'}"
            position_events = await self.event_store.get_events(position_id)
            
            if not position_events:
                return {
                    "symbol": query.symbol,
                    "quantity": 0.0,
                    "average_price": 0.0,
                    "market_value": 0.0,
                    "realized_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "total_pnl": 0.0,
                    "strategy_id": query.strategy_id
                }
            
            position = PositionAggregate(position_id)
            for event in position_events:
                position.apply_event(event)
            
            # For demo, use last price as current market price
            # In real system, this would come from market data
            current_price = position.average_price
            market_value = position.quantity * current_price
            unrealized_pnl = (current_price - position.average_price) * position.quantity
            
            return {
                "symbol": position.symbol,
                "quantity": position.quantity,
                "average_price": position.average_price,
                "market_value": market_value,
                "realized_pnl": position.realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": position.realized_pnl + unrealized_pnl,
                "last_update_time": position.last_update_time.isoformat() if position.last_update_time else None,
                "strategy_id": position.strategy_id
            }
            
        except Exception as e:
            logger.error(f"Error handling get position query: {e}")
            return None


# Command and Query Bus
class CommandBus:
    """Command bus for dispatching commands to handlers"""
    
    def __init__(self):
        self.handlers: Dict[Type, CommandHandler] = {}
    
    def register_handler(self, command_type: Type[TCommand], handler: CommandHandler[TCommand]):
        """Register command handler"""
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.__name__}")
    
    async def send(self, command: Command) -> CommandResult:
        """Send command to appropriate handler"""
        handler = self.handlers.get(type(command))
        if not handler:
            return CommandResult.failure_result(f"No handler for {type(command).__name__}")
        
        return await handler.handle(command)


class QueryBus:
    """Query bus for dispatching queries to handlers"""
    
    def __init__(self):
        self.handlers: Dict[Type, QueryHandler] = {}
    
    def register_handler(self, query_type: Type[TQuery], handler: QueryHandler[TQuery]):
        """Register query handler"""
        self.handlers[query_type] = handler
        logger.info(f"Registered handler for {query_type.__name__}")
    
    async def send(self, query: Query) -> Any:
        """Send query to appropriate handler"""
        handler = self.handlers.get(type(query))
        if not handler:
            raise ValueError(f"No handler for {type(query).__name__}")
        
        return await handler.handle(query)


# CQRS Facade
class CQRSSystem:
    """Main CQRS system facade"""
    
    def __init__(self, event_store: EventStore, event_bus: EventBus = None):
        self.event_store = event_store
        self.event_bus = event_bus
        self.command_bus = CommandBus()
        self.query_bus = QueryBus()
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default command and query handlers"""
        # Command handlers
        self.command_bus.register_handler(
            CreateOrderCommand, 
            CreateOrderCommandHandler(self.event_store)
        )
        self.command_bus.register_handler(
            CancelOrderCommand,
            CancelOrderCommandHandler(self.event_store)
        )
        self.command_bus.register_handler(
            ExecuteTradeCommand,
            ExecuteTradeCommandHandler(self.event_store)
        )
        
        # Query handlers
        self.query_bus.register_handler(
            GetOrderQuery,
            GetOrderQueryHandler(self.event_store)
        )
        self.query_bus.register_handler(
            GetPositionQuery,
            GetPositionQueryHandler(self.event_store)
        )
    
    async def send_command(self, command: Command) -> CommandResult:
        """Send command"""
        result = await self.command_bus.send(command)
        
        # Publish events to event bus
        if result.success and self.event_bus:
            for event in result.events:
                await self.event_bus.publish(event)
        
        return result
    
    async def send_query(self, query: Query) -> Any:
        """Send query"""
        return await self.query_bus.send(query)


# Example usage
async def main():
    """Example usage of CQRS system"""
    # Create components
    event_store = InMemoryEventStore()
    event_bus = EventBus()
    cqrs = CQRSSystem(event_store, event_bus)
    
    # Subscribe to events
    async def handle_order_events(event: Event):
        print(f"Event: {event.event_type.value} - {event.event_data}")
    
    event_bus.subscribe_all(handle_order_events)
    
    # Create order
    create_cmd = CreateOrderCommand(
        order_id="order-001",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=45000.0,
        strategy_id="strategy-001"
    )
    
    result = await cqrs.send_command(create_cmd)
    print(f"Create order result: {result.success}")
    
    # Query order
    order_query = GetOrderQuery(order_id="order-001")
    order_data = await cqrs.send_query(order_query)
    print(f"Order data: {order_data}")
    
    # Execute trade
    execute_cmd = ExecuteTradeCommand(
        execution_id="exec-001",
        order_id="order-001",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=45000.0
    )
    
    result = await cqrs.send_command(execute_cmd)
    print(f"Execute trade result: {result.success}")
    
    # Query position
    position_query = GetPositionQuery(symbol="BTCUSDT", strategy_id="strategy-001")
    position_data = await cqrs.send_query(position_query)
    print(f"Position data: {position_data}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
