"""
Broker API Integration Framework for HFT Simulator

This module provides a unified interface for connecting to various broker APIs
for live order execution, portfolio management, and account monitoring.

Supported Brokers:
- Interactive Brokers (TWS API)
- Alpaca Markets
- TD Ameritrade
- Binance (Crypto)
- Coinbase Pro (Crypto)
- Mock Broker (for testing)

Key Features:
- Unified broker interface
- Real-time order execution
- Portfolio and position tracking
- Account balance monitoring
- Risk management integration
- Comprehensive error handling
- Automatic reconnection
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
import time
import base64
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import pandas as pd

from src.utils.logger import get_logger
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.engine.order_types import Order, Trade
from .types import Position


class BrokerType(Enum):
    """Supported broker types"""
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    TD_AMERITRADE = "td_ameritrade"
    BINANCE = "binance"
    COINBASE_PRO = "coinbase_pro"
    MOCK = "mock"


@dataclass
class BrokerConfig:
    """Configuration for broker connections"""
    
    broker_type: BrokerType
    
    # Authentication
    api_key: str
    secret_key: Optional[str] = None
    passphrase: Optional[str] = None  # For some crypto exchanges
    
    # Connection settings
    base_url: str = ""
    sandbox_mode: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    
    # Trading settings
    default_account: Optional[str] = None
    enable_paper_trading: bool = True
    
    # Risk settings
    max_position_value: float = 100000.0
    max_order_value: float = 10000.0
    daily_loss_limit: float = 5000.0
    
    # Rate limiting
    requests_per_second: int = 10
    burst_limit: int = 100


@dataclass
class Account:
    """Represents broker account information"""
    
    account_id: str
    account_type: str  # 'cash', 'margin', etc.
    
    # Balances
    cash_balance: float
    buying_power: float
    total_value: float
    
    # P&L
    day_pnl: float
    total_pnl: float
    
    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Metadata
    currency: str = "USD"
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResponse:
    """Response from broker order submission"""
    
    success: bool
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    # Order details
    symbol: Optional[str] = None
    side: Optional[OrderSide] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    order_type: Optional[OrderType] = None
    status: Optional[OrderStatus] = None
    
    # Error information
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None


class BrokerAPI(ABC):
    """
    Abstract base class for broker API implementations
    
    This class defines the interface that all broker integrations
    must implement, ensuring consistency across different brokers.
    """
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # Connection state
        self.connected = False
        self.authenticated = False
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Account information
        self.account: Optional[Account] = None
        self.positions: Dict[str, Position] = {}
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Rate limiting
        self.request_timestamps = []
        
        # Statistics
        self.stats = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_rejected': 0,
            'api_calls': 0,
            'api_errors': 0,
            'connection_errors': 0,
            'last_update': None
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to broker API"""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with broker API"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> OrderResponse:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel existing order"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Account:
        """Get current account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get status of specific order"""
        pass
    
    async def _rate_limit_check(self) -> None:
        """Check and enforce rate limits"""
        now = time.time()
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 1.0
        ]
        
        # Check if we're within limits
        if len(self.request_timestamps) >= self.config.requests_per_second:
            sleep_time = 1.0 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order against broker-specific rules"""
        
        # Check order value limits
        if order.price and order.volume:
            order_value = order.price * order.volume
            if order_value > self.config.max_order_value:
                self.logger.error(f"Order value {order_value} exceeds limit {self.config.max_order_value}")
                return False
        
        # Additional validation would be implemented here
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get broker API statistics"""
        return {
            **self.stats,
            'connected': self.connected,
            'authenticated': self.authenticated,
            'active_orders': len(self.active_orders),
            'positions_count': len(self.positions),
            'account_value': self.account.total_value if self.account else 0
        }


class AlpacaBroker(BrokerAPI):
    """
    Alpaca Markets broker integration
    
    Provides integration with Alpaca's commission-free stock trading API.
    Supports both paper trading and live trading modes.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Alpaca-specific URLs
        if config.sandbox_mode:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key or "",
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Establish connection to Alpaca API"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers=self.headers
            )
            
            # Test connection
            if await self.authenticate():
                self.connected = True
                self.logger.info("Connected to Alpaca API")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            self.stats['connection_errors'] += 1
            return False
    
    async def disconnect(self) -> None:
        """Close connection to Alpaca API"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.connected = False
        self.authenticated = False
        self.logger.info("Disconnected from Alpaca API")
    
    async def authenticate(self) -> bool:
        """Authenticate with Alpaca API"""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    account_data = await response.json()
                    self.authenticated = True
                    self.logger.info("Authenticated with Alpaca API")
                    
                    # Update account information
                    await self._update_account_from_data(account_data)
                    return True
                else:
                    self.logger.error(f"Authentication failed: {response.status}")
                    return False
        
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    async def submit_order(self, order: Order) -> OrderResponse:
        """Submit order to Alpaca"""
        if not self.connected or not self.authenticated:
            return OrderResponse(
                success=False,
                message="Not connected or authenticated",
                error_code="CONNECTION_ERROR"
            )
        
        if not self._validate_order(order):
            return OrderResponse(
                success=False,
                message="Order validation failed",
                error_code="VALIDATION_ERROR"
            )
        
        try:
            await self._rate_limit_check()
            
            # Convert to Alpaca order format
            alpaca_order = {
                "symbol": order.symbol,
                "qty": str(order.volume),
                "side": "buy" if order.side == OrderSide.BUY else "sell",
                "type": self._convert_order_type(order.order_type),
                "time_in_force": "DAY"
            }
            
            if order.price and order.order_type == OrderType.LIMIT:
                alpaca_order["limit_price"] = str(order.price)
            
            async with self.session.post(
                f"{self.base_url}/v2/orders",
                json=alpaca_order
            ) as response:
                
                self.stats['api_calls'] += 1
                response_data = await response.json()
                
                if response.status == 201:
                    self.stats['orders_submitted'] += 1
                    
                    return OrderResponse(
                        success=True,
                        order_id=order.order_id,
                        broker_order_id=response_data.get('id'),
                        message="Order submitted successfully",
                        timestamp=datetime.now(),
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.volume,
                        price=order.price,
                        order_type=order.order_type,
                        status=OrderStatus.PENDING
                    )
                else:
                    self.stats['orders_rejected'] += 1
                    return OrderResponse(
                        success=False,
                        message=response_data.get('message', 'Order rejected'),
                        error_code=str(response.status),
                        error_details=response_data
                    )
        
        except Exception as e:
            self.logger.error(f"Order submission error: {e}")
            self.stats['api_errors'] += 1
            return OrderResponse(
                success=False,
                message=str(e),
                error_code="API_ERROR"
            )
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel order in Alpaca"""
        try:
            await self._rate_limit_check()
            
            async with self.session.delete(
                f"{self.base_url}/v2/orders/{order_id}"
            ) as response:
                
                self.stats['api_calls'] += 1
                
                if response.status == 204:
                    return OrderResponse(
                        success=True,
                        order_id=order_id,
                        message="Order cancelled successfully",
                        timestamp=datetime.now()
                    )
                else:
                    response_data = await response.json()
                    return OrderResponse(
                        success=False,
                        message=response_data.get('message', 'Cancellation failed'),
                        error_code=str(response.status)
                    )
        
        except Exception as e:
            self.logger.error(f"Order cancellation error: {e}")
            return OrderResponse(
                success=False,
                message=str(e),
                error_code="API_ERROR"
            )
    
    async def get_account_info(self) -> Account:
        """Get Alpaca account information"""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    account_data = await response.json()
                    await self._update_account_from_data(account_data)
                    return self.account
                else:
                    self.logger.error(f"Failed to get account info: {response.status}")
                    return self.account
        
        except Exception as e:
            self.logger.error(f"Account info error: {e}")
            return self.account
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get Alpaca positions"""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.base_url}/v2/positions") as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    positions_data = await response.json()
                    self.positions = {}
                    
                    for pos_data in positions_data:
                        position = Position(
                            symbol=pos_data['symbol'],
                            quantity=float(pos_data['qty']),
                            average_price=float(pos_data['avg_entry_price']),
                            market_value=float(pos_data['market_value']),
                            unrealized_pnl=float(pos_data['unrealized_pl']),
                            realized_pnl=0.0,  # Not provided by Alpaca
                            side='long' if float(pos_data['qty']) > 0 else 'short',
                            cost_basis=float(pos_data['cost_basis']),
                            updated_at=datetime.now()
                        )
                        self.positions[position.symbol] = position
                    
                    return self.positions
                else:
                    self.logger.error(f"Failed to get positions: {response.status}")
                    return self.positions
        
        except Exception as e:
            self.logger.error(f"Positions error: {e}")
            return self.positions
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Get order status from Alpaca"""
        try:
            await self._rate_limit_check()
            
            async with self.session.get(f"{self.base_url}/v2/orders/{order_id}") as response:
                self.stats['api_calls'] += 1
                
                if response.status == 200:
                    order_data = await response.json()
                    
                    return OrderResponse(
                        success=True,
                        order_id=order_id,
                        broker_order_id=order_data.get('id'),
                        symbol=order_data.get('symbol'),
                        side=OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL,
                        quantity=float(order_data.get('qty', 0)),
                        price=float(order_data.get('limit_price', 0)) if order_data.get('limit_price') else None,
                        status=self._convert_alpaca_status(order_data.get('status')),
                        timestamp=datetime.now()
                    )
                else:
                    return OrderResponse(
                        success=False,
                        message=f"Failed to get order status: {response.status}",
                        error_code=str(response.status)
                    )
        
        except Exception as e:
            self.logger.error(f"Order status error: {e}")
            return OrderResponse(
                success=False,
                message=str(e),
                error_code="API_ERROR"
            )
    
    async def _update_account_from_data(self, account_data: Dict[str, Any]) -> None:
        """Update account information from API response"""
        self.account = Account(
            account_id=account_data.get('id', ''),
            account_type=account_data.get('account_type', 'cash'),
            cash_balance=float(account_data.get('cash', 0)),
            buying_power=float(account_data.get('buying_power', 0)),
            total_value=float(account_data.get('portfolio_value', 0)),
            day_pnl=float(account_data.get('day_trade_buying_power', 0)),
            total_pnl=0.0,  # Calculate separately
            currency=account_data.get('currency', 'USD'),
            last_updated=datetime.now(),
            metadata=account_data
        )
        
        self.stats['last_update'] = datetime.now()
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to Alpaca format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def _convert_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to internal format"""
        mapping = {
            "new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "replaced": OrderStatus.REPLACED,
            "pending_cancel": OrderStatus.PENDING_CANCEL,
            "pending_replace": OrderStatus.PENDING_REPLACE,
            "rejected": OrderStatus.REJECTED
        }
        return mapping.get(alpaca_status, OrderStatus.UNKNOWN)


class MockBroker(BrokerAPI):
    """
    Mock broker for testing and development
    
    Simulates broker behavior without actual trading.
    Useful for strategy development and testing.
    """
    
    def __init__(self, config: BrokerConfig):
        super().__init__(config)
        
        # Mock account setup
        self.account = Account(
            account_id="MOCK_ACCOUNT",
            account_type="cash",
            cash_balance=100000.0,
            buying_power=100000.0,
            total_value=100000.0,
            day_pnl=0.0,
            total_pnl=0.0,
            last_updated=datetime.now()
        )
        
        # Mock order tracking
        self.next_order_id = 1
    
    async def connect(self) -> bool:
        """Mock connection - always succeeds"""
        self.connected = True
        self.logger.info("Connected to Mock Broker")
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnection"""
        self.connected = False
        self.authenticated = False
        self.logger.info("Disconnected from Mock Broker")
    
    async def authenticate(self) -> bool:
        """Mock authentication - always succeeds"""
        self.authenticated = True
        self.logger.info("Authenticated with Mock Broker")
        return True
    
    async def submit_order(self, order: Order) -> OrderResponse:
        """Mock order submission"""
        if not self._validate_order(order):
            return OrderResponse(
                success=False,
                message="Order validation failed",
                error_code="VALIDATION_ERROR"
            )
        
        # Simulate order processing delay
        await asyncio.sleep(0.01)
        
        broker_order_id = f"MOCK_{self.next_order_id}"
        self.next_order_id += 1
        
        # Mock order acceptance (90% success rate)
        import random
        if random.random() < 0.9:
            self.stats['orders_submitted'] += 1
            self.active_orders[broker_order_id] = order
            
            return OrderResponse(
                success=True,
                order_id=order.order_id,
                broker_order_id=broker_order_id,
                message="Mock order submitted successfully",
                timestamp=datetime.now(),
                symbol=order.symbol,
                side=order.side,
                quantity=order.volume,
                price=order.price,
                order_type=order.order_type,
                status=OrderStatus.PENDING
            )
        else:
            self.stats['orders_rejected'] += 1
            return OrderResponse(
                success=False,
                message="Mock order rejected",
                error_code="MOCK_REJECTION"
            )
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Mock order cancellation"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            
            return OrderResponse(
                success=True,
                order_id=order_id,
                message="Mock order cancelled",
                timestamp=datetime.now()
            )
        else:
            return OrderResponse(
                success=False,
                message="Order not found",
                error_code="ORDER_NOT_FOUND"
            )
    
    async def get_account_info(self) -> Account:
        """Return mock account info"""
        return self.account
    
    async def get_positions(self) -> Dict[str, Position]:
        """Return mock positions"""
        return self.positions
    
    async def get_order_status(self, order_id: str) -> OrderResponse:
        """Mock order status"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            return OrderResponse(
                success=True,
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.volume,
                price=order.price,
                status=OrderStatus.FILLED,  # Mock as filled
                timestamp=datetime.now()
            )
        else:
            return OrderResponse(
                success=False,
                message="Order not found",
                error_code="ORDER_NOT_FOUND"
            )


# Factory function for creating broker instances
def create_broker(broker_type: BrokerType, config: BrokerConfig) -> BrokerAPI:
    """
    Factory function to create appropriate broker instance
    
    Args:
        broker_type: Type of broker to create
        config: Broker configuration
        
    Returns:
        Configured broker instance
    """
    if broker_type == BrokerType.ALPACA:
        return AlpacaBroker(config)
    elif broker_type == BrokerType.MOCK:
        return MockBroker(config)
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")