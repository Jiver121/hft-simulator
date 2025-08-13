"""
Kafka-based message broker for distributed HFT system.
Handles order events, market data streams, and strategy communications.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from contextlib import asynccontextmanager

try:
    from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
    from aiokafka.errors import KafkaError
except ImportError:
    # Fallback for development without Kafka
    class AIOKafkaProducer:
        def __init__(self, **kwargs): pass
        async def start(self): pass
        async def stop(self): pass
        async def send_and_wait(self, topic, value): pass
    
    class AIOKafkaConsumer:
        def __init__(self, **kwargs): pass
        async def start(self): pass
        async def stop(self): pass
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration
    
    class KafkaError(Exception): pass

from ..utils.serialization import MessageSerializer

logger = logging.getLogger(__name__)


@dataclass
class MessageMetadata:
    """Metadata for Kafka messages"""
    timestamp: datetime
    producer_id: str
    message_id: str
    version: str = "1.0"
    trace_id: Optional[str] = None


@dataclass 
class KafkaMessage:
    """Standard Kafka message format"""
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    metadata: MessageMetadata
    headers: Optional[Dict[str, str]] = None


class KafkaTopics:
    """Centralized topic definitions"""
    # Market data streams
    MARKET_DATA_LEVEL1 = "market-data-level1"
    MARKET_DATA_LEVEL2 = "market-data-level2"
    MARKET_DATA_TRADES = "market-data-trades"
    
    # Order management
    ORDER_REQUESTS = "order-requests"
    ORDER_EXECUTIONS = "order-executions"
    ORDER_UPDATES = "order-updates"
    
    # Strategy events
    STRATEGY_SIGNALS = "strategy-signals"
    STRATEGY_POSITIONS = "strategy-positions"
    STRATEGY_PNL = "strategy-pnl"
    
    # Risk management
    RISK_ALERTS = "risk-alerts"
    RISK_LIMITS = "risk-limits"
    
    # System events
    SYSTEM_HEALTH = "system-health"
    SYSTEM_ALERTS = "system-alerts"


class KafkaConfig:
    """Kafka configuration settings"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 security_protocol: str = "PLAINTEXT",
                 sasl_mechanism: Optional[str] = None,
                 sasl_username: Optional[str] = None,
                 sasl_password: Optional[str] = None,
                 ssl_cafile: Optional[str] = None,
                 ssl_certfile: Optional[str] = None,
                 ssl_keyfile: Optional[str] = None):
        
        self.bootstrap_servers = bootstrap_servers
        self.security_protocol = security_protocol
        self.sasl_mechanism = sasl_mechanism
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
    
    def get_producer_config(self) -> Dict[str, Any]:
        """Get producer configuration"""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
            "value_serializer": lambda v: json.dumps(v).encode(),
            "key_serializer": lambda k: k.encode() if k else None,
            "compression_type": "snappy",
            "max_batch_size": 16384,
            "linger_ms": 10,  # Low latency for HFT
        }
        
        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })
            
        return config
    
    def get_consumer_config(self, group_id: str) -> Dict[str, Any]:
        """Get consumer configuration"""
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
            "value_deserializer": lambda m: json.loads(m.decode()),
            "key_deserializer": lambda k: k.decode() if k else None,
            "group_id": group_id,
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,  # Manual commit for reliability
            "max_poll_records": 500,
            "fetch_min_bytes": 1,
            "fetch_max_wait_ms": 100,  # Low latency
        }
        
        if self.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.sasl_mechanism,
                "sasl_plain_username": self.sasl_username,
                "sasl_plain_password": self.sasl_password,
            })
            
        return config


class KafkaProducer:
    """High-performance Kafka producer for HFT messages"""
    
    def __init__(self, config: KafkaConfig, producer_id: str):
        self.config = config
        self.producer_id = producer_id
        self.producer = None
        self.serializer = MessageSerializer()
        self._message_counter = 0
    
    async def start(self):
        """Start the producer"""
        try:
            self.producer = AIOKafkaProducer(**self.config.get_producer_config())
            await self.producer.start()
            logger.info(f"Kafka producer {self.producer_id} started")
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self):
        """Stop the producer"""
        if self.producer:
            await self.producer.stop()
            logger.info(f"Kafka producer {self.producer_id} stopped")
    
    async def send_message(self,
                          topic: str,
                          value: Dict[str, Any],
                          key: Optional[str] = None,
                          headers: Optional[Dict[str, str]] = None,
                          trace_id: Optional[str] = None) -> bool:
        """Send a message to Kafka"""
        try:
            self._message_counter += 1
            
            # Create message metadata
            metadata = MessageMetadata(
                timestamp=datetime.now(timezone.utc),
                producer_id=self.producer_id,
                message_id=f"{self.producer_id}-{self._message_counter}",
                trace_id=trace_id
            )
            
            # Create standardized message
            message = KafkaMessage(
                topic=topic,
                key=key,
                value=value,
                metadata=metadata,
                headers=headers
            )
            
            # Serialize message
            serialized_message = asdict(message)
            
            # Send to Kafka
            await self.producer.send_and_wait(
                topic,
                value=serialized_message,
                key=key,
                headers=headers
            )
            
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending message to {topic}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message to {topic}: {e}")
            return False
    
    async def send_market_data(self, symbol: str, data: Dict[str, Any], data_type: str = "level1"):
        """Send market data message"""
        topic_map = {
            "level1": KafkaTopics.MARKET_DATA_LEVEL1,
            "level2": KafkaTopics.MARKET_DATA_LEVEL2,
            "trades": KafkaTopics.MARKET_DATA_TRADES
        }
        
        topic = topic_map.get(data_type, KafkaTopics.MARKET_DATA_LEVEL1)
        return await self.send_message(
            topic=topic,
            value=data,
            key=symbol
        )
    
    async def send_order_event(self, order_id: str, event_data: Dict[str, Any], event_type: str = "request"):
        """Send order-related event"""
        topic_map = {
            "request": KafkaTopics.ORDER_REQUESTS,
            "execution": KafkaTopics.ORDER_EXECUTIONS,
            "update": KafkaTopics.ORDER_UPDATES
        }
        
        topic = topic_map.get(event_type, KafkaTopics.ORDER_REQUESTS)
        return await self.send_message(
            topic=topic,
            value=event_data,
            key=order_id
        )
    
    async def send_strategy_signal(self, strategy_id: str, signal_data: Dict[str, Any]):
        """Send strategy signal"""
        return await self.send_message(
            topic=KafkaTopics.STRATEGY_SIGNALS,
            value=signal_data,
            key=strategy_id
        )


class KafkaConsumer:
    """High-performance Kafka consumer for HFT messages"""
    
    def __init__(self, config: KafkaConfig, group_id: str, topics: List[str]):
        self.config = config
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        self.serializer = MessageSerializer()
        self.message_handlers: Dict[str, Callable] = {}
        self._running = False
    
    async def start(self):
        """Start the consumer"""
        try:
            consumer_config = self.config.get_consumer_config(self.group_id)
            self.consumer = AIOKafkaConsumer(*self.topics, **consumer_config)
            await self.consumer.start()
            logger.info(f"Kafka consumer {self.group_id} started for topics: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self):
        """Stop the consumer"""
        self._running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info(f"Kafka consumer {self.group_id} stopped")
    
    def register_handler(self, topic: str, handler: Callable):
        """Register message handler for a topic"""
        self.message_handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")
    
    async def consume_messages(self):
        """Consume messages from Kafka"""
        self._running = True
        
        try:
            async for msg in self.consumer:
                if not self._running:
                    break
                
                try:
                    # Deserialize message
                    message_data = msg.value
                    topic = msg.topic
                    
                    # Get handler for topic
                    handler = self.message_handlers.get(topic)
                    if handler:
                        await handler(message_data)
                    else:
                        logger.warning(f"No handler registered for topic: {topic}")
                    
                    # Commit the message
                    await self.consumer.commit()
                    
                except Exception as e:
                    logger.error(f"Error processing message from {msg.topic}: {e}")
                    # Continue processing other messages
                    
        except Exception as e:
            logger.error(f"Error in message consumption loop: {e}")
        finally:
            self._running = False


class MessageBroker:
    """High-level message broker interface"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producers: Dict[str, KafkaProducer] = {}
        self.consumers: Dict[str, KafkaConsumer] = {}
    
    def create_producer(self, producer_id: str) -> KafkaProducer:
        """Create a new producer"""
        producer = KafkaProducer(self.config, producer_id)
        self.producers[producer_id] = producer
        return producer
    
    def create_consumer(self, group_id: str, topics: List[str]) -> KafkaConsumer:
        """Create a new consumer"""
        consumer = KafkaConsumer(self.config, group_id, topics)
        self.consumers[group_id] = consumer
        return consumer
    
    async def start_all(self):
        """Start all producers and consumers"""
        # Start producers
        for producer in self.producers.values():
            await producer.start()
        
        # Start consumers
        for consumer in self.consumers.values():
            await consumer.start()
    
    async def stop_all(self):
        """Stop all producers and consumers"""
        # Stop consumers
        for consumer in self.consumers.values():
            await consumer.stop()
        
        # Stop producers
        for producer in self.producers.values():
            await producer.stop()
    
    @asynccontextmanager
    async def managed_broker(self):
        """Context manager for broker lifecycle"""
        try:
            await self.start_all()
            yield self
        finally:
            await self.stop_all()


# Example usage and testing
async def main():
    """Example usage of Kafka broker"""
    config = KafkaConfig()
    broker = MessageBroker(config)
    
    # Create producer and consumer
    producer = broker.create_producer("test-producer")
    consumer = broker.create_consumer("test-group", [KafkaTopics.MARKET_DATA_LEVEL1])
    
    # Register message handler
    async def handle_market_data(message):
        print(f"Received market data: {message}")
    
    consumer.register_handler(KafkaTopics.MARKET_DATA_LEVEL1, handle_market_data)
    
    async with broker.managed_broker():
        # Send test message
        await producer.send_market_data("BTCUSDT", {
            "symbol": "BTCUSDT",
            "bid": 45000.0,
            "ask": 45001.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Consume for a short time
        await asyncio.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
