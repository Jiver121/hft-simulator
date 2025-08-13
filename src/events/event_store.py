"""
Event sourcing implementation for distributed HFT system.
Provides event storage, replay, and projection capabilities.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Type, AsyncIterator
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from enum import Enum

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from ..utils.serialization import MessageSerializer

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types in the HFT system"""
    # Market data events
    MARKET_DATA_RECEIVED = "market_data_received"
    ORDER_BOOK_UPDATED = "order_book_updated"
    
    # Order events
    ORDER_CREATED = "order_created"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_REJECTED = "order_rejected"
    
    # Execution events
    TRADE_EXECUTED = "trade_executed"
    SETTLEMENT_COMPLETED = "settlement_completed"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_MODIFIED = "position_modified"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    RISK_CHECK_FAILED = "risk_check_failed"
    
    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    
    # System events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class Event:
    """Base event class"""
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    event_data: Dict[str, Any]
    timestamp: datetime
    version: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "event_data": self.event_data,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            event_data=data["event_data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data["version"],
            metadata=data.get("metadata")
        )


class EventStore(ABC):
    """Abstract base class for event stores"""
    
    @abstractmethod
    async def append_event(self, event: Event) -> bool:
        """Append event to store"""
        pass
    
    @abstractmethod
    async def get_events(self, 
                        aggregate_id: str, 
                        from_version: int = 0,
                        to_version: Optional[int] = None) -> List[Event]:
        """Get events for aggregate"""
        pass
    
    @abstractmethod
    async def get_events_by_type(self, 
                               event_type: EventType,
                               from_time: Optional[datetime] = None,
                               to_time: Optional[datetime] = None) -> List[Event]:
        """Get events by type within time range"""
        pass
    
    @abstractmethod
    async def stream_events(self,
                           from_position: int = 0,
                           batch_size: int = 100) -> AsyncIterator[List[Event]]:
        """Stream events from position"""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store for testing and development"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.aggregate_events: Dict[str, List[Event]] = {}
        self.event_types: Dict[EventType, List[Event]] = {}
        self._lock = asyncio.Lock()
    
    async def append_event(self, event: Event) -> bool:
        """Append event to in-memory store"""
        async with self._lock:
            try:
                # Add to main events list
                self.events.append(event)
                
                # Index by aggregate
                if event.aggregate_id not in self.aggregate_events:
                    self.aggregate_events[event.aggregate_id] = []
                self.aggregate_events[event.aggregate_id].append(event)
                
                # Index by event type
                if event.event_type not in self.event_types:
                    self.event_types[event.event_type] = []
                self.event_types[event.event_type].append(event)
                
                logger.debug(f"Event appended: {event.event_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to append event: {e}")
                return False
    
    async def get_events(self, 
                        aggregate_id: str, 
                        from_version: int = 0,
                        to_version: Optional[int] = None) -> List[Event]:
        """Get events for aggregate"""
        events = self.aggregate_events.get(aggregate_id, [])
        
        # Filter by version
        filtered_events = [
            e for e in events 
            if e.version >= from_version and (to_version is None or e.version <= to_version)
        ]
        
        return sorted(filtered_events, key=lambda e: e.version)
    
    async def get_events_by_type(self, 
                               event_type: EventType,
                               from_time: Optional[datetime] = None,
                               to_time: Optional[datetime] = None) -> List[Event]:
        """Get events by type within time range"""
        events = self.event_types.get(event_type, [])
        
        # Filter by time range
        if from_time or to_time:
            filtered_events = []
            for event in events:
                if from_time and event.timestamp < from_time:
                    continue
                if to_time and event.timestamp > to_time:
                    continue
                filtered_events.append(event)
            events = filtered_events
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def stream_events(self,
                           from_position: int = 0,
                           batch_size: int = 100) -> AsyncIterator[List[Event]]:
        """Stream events from position"""
        position = from_position
        
        while position < len(self.events):
            batch = self.events[position:position + batch_size]
            if batch:
                yield batch
                position += len(batch)
            else:
                break


class RedisEventStore(EventStore):
    """Redis-based event store for production use"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("redis library not available")
            
        self.redis_url = redis_url
        self.redis = None
        self.serializer = MessageSerializer("json")
    
    async def connect(self):
        """Connect to Redis"""
        self.redis = redis.from_url(self.redis_url)
        await self.redis.ping()
        logger.info("Connected to Redis event store")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
    
    async def append_event(self, event: Event) -> bool:
        """Append event to Redis store"""
        try:
            if not self.redis:
                await self.connect()
            
            # Serialize event
            event_data = json.dumps(event.to_dict())
            
            # Use Redis transaction for atomicity
            async with self.redis.pipeline(transaction=True) as pipe:
                # Add to main events stream
                await pipe.xadd("events:stream", {"data": event_data})
                
                # Add to aggregate stream
                aggregate_key = f"events:aggregate:{event.aggregate_id}"
                await pipe.xadd(aggregate_key, {"data": event_data})
                
                # Add to event type index
                event_type_key = f"events:type:{event.event_type.value}"
                await pipe.zadd(event_type_key, {event.event_id: event.timestamp.timestamp()})
                
                # Store event details
                event_key = f"event:{event.event_id}"
                await pipe.set(event_key, event_data)
                
                # Execute transaction
                await pipe.execute()
            
            logger.debug(f"Event appended to Redis: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event to Redis: {e}")
            return False
    
    async def get_events(self, 
                        aggregate_id: str, 
                        from_version: int = 0,
                        to_version: Optional[int] = None) -> List[Event]:
        """Get events for aggregate from Redis"""
        try:
            if not self.redis:
                await self.connect()
            
            aggregate_key = f"events:aggregate:{aggregate_id}"
            
            # Get events from stream
            events_data = await self.redis.xrange(aggregate_key)
            
            events = []
            for event_id, fields in events_data:
                event_dict = json.loads(fields[b"data"])
                event = Event.from_dict(event_dict)
                
                # Filter by version
                if (event.version >= from_version and 
                    (to_version is None or event.version <= to_version)):
                    events.append(event)
            
            return sorted(events, key=lambda e: e.version)
            
        except Exception as e:
            logger.error(f"Failed to get events from Redis: {e}")
            return []
    
    async def get_events_by_type(self, 
                               event_type: EventType,
                               from_time: Optional[datetime] = None,
                               to_time: Optional[datetime] = None) -> List[Event]:
        """Get events by type from Redis"""
        try:
            if not self.redis:
                await self.connect()
            
            event_type_key = f"events:type:{event_type.value}"
            
            # Determine time range
            min_score = from_time.timestamp() if from_time else 0
            max_score = to_time.timestamp() if to_time else "+inf"
            
            # Get event IDs in time range
            event_ids = await self.redis.zrangebyscore(event_type_key, min_score, max_score)
            
            # Get event details
            events = []
            for event_id in event_ids:
                event_key = f"event:{event_id.decode()}"
                event_data = await self.redis.get(event_key)
                if event_data:
                    event_dict = json.loads(event_data)
                    event = Event.from_dict(event_dict)
                    events.append(event)
            
            return sorted(events, key=lambda e: e.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to get events by type from Redis: {e}")
            return []
    
    async def stream_events(self,
                           from_position: int = 0,
                           batch_size: int = 100) -> AsyncIterator[List[Event]]:
        """Stream events from Redis"""
        try:
            if not self.redis:
                await self.connect()
            
            # Convert position to Redis stream ID
            last_id = f"{from_position}-0"
            
            while True:
                # Read from main events stream
                results = await self.redis.xread(
                    {"events:stream": last_id}, 
                    count=batch_size,
                    block=1000  # Block for 1 second if no new events
                )
                
                if not results:
                    break  # No more events
                
                stream_name, events_data = results[0]
                if not events_data:
                    break
                
                # Convert to Event objects
                events = []
                for event_id, fields in events_data:
                    event_dict = json.loads(fields[b"data"])
                    event = Event.from_dict(event_dict)
                    events.append(event)
                    last_id = event_id.decode()
                
                if events:
                    yield events
                
        except Exception as e:
            logger.error(f"Failed to stream events from Redis: {e}")


class FileEventStore(EventStore):
    """File-based event store for simple deployments"""
    
    def __init__(self, file_path: str = "events.jsonl"):
        self.file_path = file_path
        self.serializer = MessageSerializer("json")
        self._lock = asyncio.Lock()
    
    async def append_event(self, event: Event) -> bool:
        """Append event to file"""
        try:
            if not AIOFILES_AVAILABLE:
                # Fallback to synchronous file operations
                with open(self.file_path, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            else:
                async with aiofiles.open(self.file_path, "a") as f:
                    await f.write(json.dumps(event.to_dict()) + "\n")
            
            logger.debug(f"Event appended to file: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append event to file: {e}")
            return False
    
    async def get_events(self, 
                        aggregate_id: str, 
                        from_version: int = 0,
                        to_version: Optional[int] = None) -> List[Event]:
        """Get events for aggregate from file"""
        events = []
        
        try:
            if not AIOFILES_AVAILABLE:
                # Fallback to synchronous file operations
                with open(self.file_path, "r") as f:
                    for line in f:
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        
                        if (event.aggregate_id == aggregate_id and
                            event.version >= from_version and
                            (to_version is None or event.version <= to_version)):
                            events.append(event)
            else:
                async with aiofiles.open(self.file_path, "r") as f:
                    async for line in f:
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        
                        if (event.aggregate_id == aggregate_id and
                            event.version >= from_version and
                            (to_version is None or event.version <= to_version)):
                            events.append(event)
            
        except FileNotFoundError:
            logger.warning(f"Event store file not found: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to read events from file: {e}")
        
        return sorted(events, key=lambda e: e.version)
    
    async def get_events_by_type(self, 
                               event_type: EventType,
                               from_time: Optional[datetime] = None,
                               to_time: Optional[datetime] = None) -> List[Event]:
        """Get events by type from file"""
        events = []
        
        try:
            if not AIOFILES_AVAILABLE:
                with open(self.file_path, "r") as f:
                    for line in f:
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        
                        if event.event_type == event_type:
                            # Check time range
                            if from_time and event.timestamp < from_time:
                                continue
                            if to_time and event.timestamp > to_time:
                                continue
                            events.append(event)
            else:
                async with aiofiles.open(self.file_path, "r") as f:
                    async for line in f:
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        
                        if event.event_type == event_type:
                            # Check time range
                            if from_time and event.timestamp < from_time:
                                continue
                            if to_time and event.timestamp > to_time:
                                continue
                            events.append(event)
            
        except FileNotFoundError:
            logger.warning(f"Event store file not found: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to read events from file: {e}")
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def stream_events(self,
                           from_position: int = 0,
                           batch_size: int = 100) -> AsyncIterator[List[Event]]:
        """Stream events from file"""
        try:
            position = 0
            batch = []
            
            if not AIOFILES_AVAILABLE:
                with open(self.file_path, "r") as f:
                    for line in f:
                        if position < from_position:
                            position += 1
                            continue
                        
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        batch.append(event)
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                        
                        position += 1
            else:
                async with aiofiles.open(self.file_path, "r") as f:
                    async for line in f:
                        if position < from_position:
                            position += 1
                            continue
                        
                        event_dict = json.loads(line.strip())
                        event = Event.from_dict(event_dict)
                        batch.append(event)
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                        
                        position += 1
            
            # Yield remaining events
            if batch:
                yield batch
                
        except FileNotFoundError:
            logger.warning(f"Event store file not found: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to stream events from file: {e}")


class EventBus:
    """Event bus for publishing and subscribing to events"""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[callable]] = {}
        self.wildcard_subscribers: List[callable] = []
    
    def subscribe(self, event_type: EventType, handler: callable):
        """Subscribe to specific event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        logger.info(f"Subscribed to {event_type.value}")
    
    def subscribe_all(self, handler: callable):
        """Subscribe to all events"""
        self.wildcard_subscribers.append(handler)
        logger.info("Subscribed to all events")
    
    async def publish(self, event: Event):
        """Publish event to subscribers"""
        try:
            # Notify specific event type subscribers
            if event.event_type in self.subscribers:
                for handler in self.subscribers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
            
            # Notify wildcard subscribers
            for handler in self.wildcard_subscribers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in wildcard event handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error publishing event: {e}")


class EventRepository:
    """High-level interface for event operations"""
    
    def __init__(self, event_store: EventStore, event_bus: EventBus = None):
        self.event_store = event_store
        self.event_bus = event_bus or EventBus()
    
    async def save_event(self, 
                        event_type: EventType,
                        aggregate_id: str,
                        aggregate_type: str,
                        event_data: Dict[str, Any],
                        version: int = 1,
                        metadata: Optional[Dict[str, Any]] = None) -> Event:
        """Create and save event"""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            aggregate_id=aggregate_id,
            aggregate_type=aggregate_type,
            event_data=event_data,
            timestamp=datetime.now(timezone.utc),
            version=version,
            metadata=metadata
        )
        
        # Save to store
        success = await self.event_store.append_event(event)
        if not success:
            raise RuntimeError(f"Failed to save event: {event.event_id}")
        
        # Publish to bus
        await self.event_bus.publish(event)
        
        return event
    
    async def get_aggregate_events(self, aggregate_id: str) -> List[Event]:
        """Get all events for an aggregate"""
        return await self.event_store.get_events(aggregate_id)
    
    async def replay_events(self, 
                           from_time: Optional[datetime] = None,
                           to_time: Optional[datetime] = None) -> AsyncIterator[Event]:
        """Replay events within time range"""
        # Stream all events and filter by time
        async for event_batch in self.event_store.stream_events():
            for event in event_batch:
                if from_time and event.timestamp < from_time:
                    continue
                if to_time and event.timestamp > to_time:
                    continue
                yield event


# Example usage
async def main():
    """Example usage of event store"""
    # Create event store (use InMemoryEventStore for demo)
    event_store = InMemoryEventStore()
    event_bus = EventBus()
    repository = EventRepository(event_store, event_bus)
    
    # Subscribe to events
    async def handle_order_event(event: Event):
        print(f"Order event: {event.event_type.value} for {event.aggregate_id}")
    
    event_bus.subscribe(EventType.ORDER_CREATED, handle_order_event)
    event_bus.subscribe(EventType.ORDER_FILLED, handle_order_event)
    
    # Save some events
    await repository.save_event(
        EventType.ORDER_CREATED,
        "order123",
        "Order",
        {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": 0.1,
            "price": 45000.0
        },
        version=1
    )
    
    await repository.save_event(
        EventType.ORDER_FILLED,
        "order123",
        "Order",
        {
            "symbol": "BTCUSDT",
            "quantity": 0.1,
            "price": 45000.0,
            "execution_id": "exec123"
        },
        version=2
    )
    
    # Get events for aggregate
    events = await repository.get_aggregate_events("order123")
    print(f"\nFound {len(events)} events for order123")
    
    # Replay events
    print("\nReplaying events:")
    async for event in repository.replay_events():
        print(f"  {event.timestamp}: {event.event_type.value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
