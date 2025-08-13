"""
RabbitMQ-based message broker as an alternative to Kafka.
Provides similar functionality with AMQP protocol.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from contextlib import asynccontextmanager

try:
    import aio_pika
    from aio_pika import Message, ExchangeType, DeliveryMode
except ImportError:
    # Fallback for development without RabbitMQ
    class Message:
        def __init__(self, **kwargs): pass
    class ExchangeType:
        TOPIC = "topic"
        DIRECT = "direct"
    class DeliveryMode:
        PERSISTENT = 2

from ..utils.serialization import MessageSerializer

logger = logging.getLogger(__name__)


@dataclass
class RabbitMQConfig:
    """RabbitMQ configuration settings"""
    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    ssl: bool = False
    connection_timeout: int = 30
    heartbeat: int = 600
    
    def get_connection_url(self) -> str:
        """Get RabbitMQ connection URL"""
        protocol = "amqps" if self.ssl else "amqp"
        return f"{protocol}://{self.username}:{self.password}@{self.host}:{self.port}{self.virtual_host}"


class RabbitMQExchanges:
    """Centralized exchange definitions"""
    MARKET_DATA = "market_data"
    ORDERS = "orders"
    STRATEGIES = "strategies"
    RISK = "risk"
    SYSTEM = "system"


class RabbitMQRoutingKeys:
    """Routing key definitions"""
    # Market data
    MARKET_DATA_LEVEL1 = "market.data.level1"
    MARKET_DATA_LEVEL2 = "market.data.level2"
    MARKET_DATA_TRADES = "market.data.trades"
    
    # Orders
    ORDER_REQUESTS = "order.requests"
    ORDER_EXECUTIONS = "order.executions" 
    ORDER_UPDATES = "order.updates"
    
    # Strategies
    STRATEGY_SIGNALS = "strategy.signals"
    STRATEGY_POSITIONS = "strategy.positions"
    STRATEGY_PNL = "strategy.pnl"
    
    # Risk
    RISK_ALERTS = "risk.alerts"
    RISK_LIMITS = "risk.limits"
    
    # System
    SYSTEM_HEALTH = "system.health"
    SYSTEM_ALERTS = "system.alerts"


class RabbitMQProducer:
    """RabbitMQ producer for HFT messages"""
    
    def __init__(self, config: RabbitMQConfig, producer_id: str):
        self.config = config
        self.producer_id = producer_id
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self.serializer = MessageSerializer()
        self._message_counter = 0
    
    async def start(self):
        """Start the producer"""
        try:
            self.connection = await aio_pika.connect_robust(
                self.config.get_connection_url(),
                timeout=self.config.connection_timeout,
                heartbeat=self.config.heartbeat
            )
            self.channel = await self.connection.channel()
            
            # Declare exchanges
            await self._declare_exchanges()
            
            logger.info(f"RabbitMQ producer {self.producer_id} started")
        except Exception as e:
            logger.error(f"Failed to start RabbitMQ producer: {e}")
            raise
    
    async def stop(self):
        """Stop the producer"""
        if self.connection:
            await self.connection.close()
            logger.info(f"RabbitMQ producer {self.producer_id} stopped")
    
    async def _declare_exchanges(self):
        """Declare all necessary exchanges"""
        exchanges = [
            (RabbitMQExchanges.MARKET_DATA, ExchangeType.TOPIC),
            (RabbitMQExchanges.ORDERS, ExchangeType.TOPIC),
            (RabbitMQExchanges.STRATEGIES, ExchangeType.TOPIC),
            (RabbitMQExchanges.RISK, ExchangeType.TOPIC),
            (RabbitMQExchanges.SYSTEM, ExchangeType.TOPIC),
        ]
        
        for exchange_name, exchange_type in exchanges:
            exchange = await self.channel.declare_exchange(
                exchange_name, exchange_type, durable=True
            )
            self.exchanges[exchange_name] = exchange
    
    async def publish_message(self,
                             exchange: str,
                             routing_key: str,
                             message_body: Dict[str, Any],
                             persistent: bool = True) -> bool:
        """Publish a message to RabbitMQ"""
        try:
            self._message_counter += 1
            
            # Add metadata
            message_body["_metadata"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "producer_id": self.producer_id,
                "message_id": f"{self.producer_id}-{self._message_counter}"
            }
            
            # Create message
            message = Message(
                json.dumps(message_body).encode(),
                delivery_mode=DeliveryMode.PERSISTENT if persistent else DeliveryMode.NOT_PERSISTENT,
                headers={
                    "producer_id": self.producer_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Publish to exchange
            exchange_obj = self.exchanges[exchange]
            await exchange_obj.publish(message, routing_key=routing_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing message to {exchange}.{routing_key}: {e}")
            return False
    
    async def send_market_data(self, symbol: str, data: Dict[str, Any], data_type: str = "level1"):
        """Send market data message"""
        routing_key_map = {
            "level1": RabbitMQRoutingKeys.MARKET_DATA_LEVEL1,
            "level2": RabbitMQRoutingKeys.MARKET_DATA_LEVEL2,
            "trades": RabbitMQRoutingKeys.MARKET_DATA_TRADES
        }
        
        routing_key = routing_key_map.get(data_type, RabbitMQRoutingKeys.MARKET_DATA_LEVEL1)
        
        # Add symbol to routing key for better routing
        full_routing_key = f"{routing_key}.{symbol}"
        
        return await self.publish_message(
            exchange=RabbitMQExchanges.MARKET_DATA,
            routing_key=full_routing_key,
            message_body=data
        )
    
    async def send_order_event(self, order_id: str, event_data: Dict[str, Any], event_type: str = "request"):
        """Send order-related event"""
        routing_key_map = {
            "request": RabbitMQRoutingKeys.ORDER_REQUESTS,
            "execution": RabbitMQRoutingKeys.ORDER_EXECUTIONS,
            "update": RabbitMQRoutingKeys.ORDER_UPDATES
        }
        
        routing_key = routing_key_map.get(event_type, RabbitMQRoutingKeys.ORDER_REQUESTS)
        
        return await self.publish_message(
            exchange=RabbitMQExchanges.ORDERS,
            routing_key=routing_key,
            message_body=event_data
        )


class RabbitMQConsumer:
    """RabbitMQ consumer for HFT messages"""
    
    def __init__(self, config: RabbitMQConfig, consumer_id: str):
        self.config = config
        self.consumer_id = consumer_id
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self.queues = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.serializer = MessageSerializer()
        self._running = False
    
    async def start(self):
        """Start the consumer"""
        try:
            self.connection = await aio_pika.connect_robust(
                self.config.get_connection_url(),
                timeout=self.config.connection_timeout,
                heartbeat=self.config.heartbeat
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=100)  # For performance
            
            # Declare exchanges
            await self._declare_exchanges()
            
            logger.info(f"RabbitMQ consumer {self.consumer_id} started")
        except Exception as e:
            logger.error(f"Failed to start RabbitMQ consumer: {e}")
            raise
    
    async def stop(self):
        """Stop the consumer"""
        self._running = False
        if self.connection:
            await self.connection.close()
            logger.info(f"RabbitMQ consumer {self.consumer_id} stopped")
    
    async def _declare_exchanges(self):
        """Declare all necessary exchanges"""
        exchanges = [
            (RabbitMQExchanges.MARKET_DATA, ExchangeType.TOPIC),
            (RabbitMQExchanges.ORDERS, ExchangeType.TOPIC),
            (RabbitMQExchanges.STRATEGIES, ExchangeType.TOPIC),
            (RabbitMQExchanges.RISK, ExchangeType.TOPIC),
            (RabbitMQExchanges.SYSTEM, ExchangeType.TOPIC),
        ]
        
        for exchange_name, exchange_type in exchanges:
            exchange = await self.channel.declare_exchange(
                exchange_name, exchange_type, durable=True
            )
            self.exchanges[exchange_name] = exchange
    
    async def subscribe_to_routing_key(self, exchange: str, routing_key: str, queue_name: str = None):
        """Subscribe to messages with specific routing key"""
        if queue_name is None:
            queue_name = f"{self.consumer_id}_{exchange}_{routing_key}".replace(".", "_").replace("*", "all")
        
        # Declare queue
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        # Bind queue to exchange with routing key
        exchange_obj = self.exchanges[exchange]
        await queue.bind(exchange_obj, routing_key=routing_key)
        
        # Store queue reference
        self.queues[f"{exchange}.{routing_key}"] = queue
        
        logger.info(f"Subscribed to {exchange}.{routing_key} with queue {queue_name}")
    
    def register_handler(self, exchange: str, routing_key: str, handler: Callable):
        """Register message handler for exchange and routing key"""
        key = f"{exchange}.{routing_key}"
        self.message_handlers[key] = handler
        logger.info(f"Registered handler for {key}")
    
    async def start_consuming(self):
        """Start consuming messages"""
        self._running = True
        
        async def process_message(message: aio_pika.IncomingMessage):
            """Process incoming message"""
            try:
                # Parse message body
                message_data = json.loads(message.body.decode())
                
                # Extract exchange and routing key info
                exchange = message.exchange
                routing_key = message.routing_key
                key = f"{exchange}.{routing_key}"
                
                # Find appropriate handler
                handler = self.message_handlers.get(key)
                if handler:
                    await handler(message_data)
                else:
                    logger.warning(f"No handler registered for {key}")
                
                # Acknowledge message
                message.ack()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                message.reject(requeue=False)  # Dead letter or discard
        
        # Start consuming from all queues
        for queue in self.queues.values():
            await queue.consume(process_message)
        
        # Keep consuming until stopped
        try:
            while self._running:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in consuming loop: {e}")
        finally:
            self._running = False


class RabbitMQBroker:
    """High-level RabbitMQ broker interface"""
    
    def __init__(self, config: RabbitMQConfig):
        self.config = config
        self.producers: Dict[str, RabbitMQProducer] = {}
        self.consumers: Dict[str, RabbitMQConsumer] = {}
    
    def create_producer(self, producer_id: str) -> RabbitMQProducer:
        """Create a new producer"""
        producer = RabbitMQProducer(self.config, producer_id)
        self.producers[producer_id] = producer
        return producer
    
    def create_consumer(self, consumer_id: str) -> RabbitMQConsumer:
        """Create a new consumer"""
        consumer = RabbitMQConsumer(self.config, consumer_id)
        self.consumers[consumer_id] = consumer
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


# Example usage
async def main():
    """Example usage of RabbitMQ broker"""
    config = RabbitMQConfig()
    broker = RabbitMQBroker(config)
    
    # Create producer and consumer
    producer = broker.create_producer("test-producer")
    consumer = broker.create_consumer("test-consumer")
    
    async with broker.managed_broker():
        # Subscribe to market data
        await consumer.subscribe_to_routing_key(
            RabbitMQExchanges.MARKET_DATA,
            "market.data.level1.*"
        )
        
        # Register message handler
        async def handle_market_data(message):
            print(f"Received market data: {message}")
        
        consumer.register_handler(
            RabbitMQExchanges.MARKET_DATA,
            "market.data.level1.*",
            handle_market_data
        )
        
        # Start consuming
        consume_task = asyncio.create_task(consumer.start_consuming())
        
        # Send test message
        await producer.send_market_data("BTCUSDT", {
            "symbol": "BTCUSDT",
            "bid": 45000.0,
            "ask": 45001.0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Wait briefly then stop
        await asyncio.sleep(5)
        await consumer.stop()
        consume_task.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
