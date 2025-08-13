# API Reference

This document provides comprehensive API documentation for the HFT Trading Simulator. All classes, methods, and functions are documented with parameters, return values, and usage examples.

## Table of Contents

- [Core Engine](#core-engine)
- [Trading Strategies](#trading-strategies)  
- [Execution System](#execution-system)
- [Performance Analytics](#performance-analytics)
- [Real-Time Trading](#real-time-trading)
- [Data Processing](#data-processing)
- [Visualization](#visualization)
- [Utilities](#utilities)

---

## Core Engine

### OrderBook

Main order book implementation for managing bid/ask levels and order matching.

```python
from src.engine.order_book import OrderBook
```

#### Class: `OrderBook`

**Constructor:**
```python
OrderBook(symbol: str, tick_size: float = 0.01)
```

**Parameters:**
- `symbol` (str): Trading symbol (e.g., 'BTCUSDT')
- `tick_size` (float): Minimum price increment

**Methods:**

##### `add_order(order: Order) -> List[Trade]`

Adds an order to the book and returns any resulting trades.

**Parameters:**
- `order` (Order): Order to add

**Returns:**
- `List[Trade]`: List of trades generated from the order

**Example:**
```python
order_book = OrderBook('BTCUSDT')
order = Order('ORDER001', 'BTCUSDT', OrderSide.BUY, OrderType.LIMIT, 100, timestamp, price=50000.0)
trades = order_book.add_order(order)
```

##### `cancel_order(order_id: str) -> bool`

Cancels an order by ID.

**Parameters:**
- `order_id` (str): Unique order identifier

**Returns:**
- `bool`: True if order was cancelled successfully

##### `get_depth(side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]`

Gets market depth for specified side.

**Parameters:**
- `side` (OrderSide): BID or ASK side
- `levels` (int): Number of price levels to return

**Returns:**
- `List[Tuple[float, int]]`: List of (price, volume) tuples

##### `get_snapshot() -> BookSnapshot`

Returns current order book snapshot.

**Returns:**
- `BookSnapshot`: Current state of the order book

##### `get_best_bid() -> Optional[float]`

Returns the best bid price.

##### `get_best_ask() -> Optional[float]`

Returns the best ask price.

##### `get_spread() -> Optional[float]`

Returns the bid-ask spread.

##### `get_mid_price() -> Optional[float]`

Returns the mid-price between best bid and ask.

---

### Order Types

Core order and trade data structures.

```python
from src.engine.order_types import Order, Trade, OrderSide, OrderType
```

#### Class: `Order`

Represents a trading order.

**Constructor:**
```python
Order(order_id: str, symbol: str, side: OrderSide, order_type: OrderType, 
      volume: int, timestamp: pd.Timestamp, price: Optional[float] = None, 
      time_in_force: str = 'GTC')
```

**Parameters:**
- `order_id` (str): Unique identifier
- `symbol` (str): Trading symbol
- `side` (OrderSide): BUY or SELL
- `order_type` (OrderType): LIMIT, MARKET, STOP, etc.
- `volume` (int): Order quantity
- `timestamp` (pd.Timestamp): Order timestamp
- `price` (Optional[float]): Price (required for limit orders)
- `time_in_force` (str): Order duration ('GTC', 'IOC', 'FOK')

**Properties:**
- `is_buy` (bool): True if buy order
- `is_sell` (bool): True if sell order
- `is_limit_order` (bool): True if limit order
- `is_market_order` (bool): True if market order
- `is_filled` (bool): True if completely filled
- `remaining_volume` (int): Unfilled quantity

**Methods:**

##### `fill(volume: int, price: float) -> Trade`

Fills the order partially or completely.

**Parameters:**
- `volume` (int): Volume to fill
- `price` (float): Fill price

**Returns:**
- `Trade`: Trade object representing the fill

#### Class: `Trade`

Represents a completed trade.

**Constructor:**
```python
Trade(trade_id: str, symbol: str, buyer_order_id: str, seller_order_id: str,
      volume: int, price: float, timestamp: pd.Timestamp, aggressor_side: OrderSide)
```

**Properties:**
- `value` (float): Trade value (volume * price)
- `is_buy_aggressor` (bool): True if buy side was aggressor

#### Enums

##### `OrderSide`
- `BUY` / `BID`: Buy/bid order
- `SELL` / `ASK`: Sell/ask order

##### `OrderType`
- `LIMIT`: Limit order
- `MARKET`: Market order
- `STOP`: Stop order
- `STOP_LIMIT`: Stop limit order

---

### Market Data

Market data structures and utilities.

```python
from src.engine.market_data import BookSnapshot, MarketData
```

#### Class: `BookSnapshot`

Represents order book state at a point in time.

**Constructor:**
```python
BookSnapshot(symbol: str, timestamp: pd.Timestamp, 
            bids: List[PriceLevel], asks: List[PriceLevel])
```

**Properties:**
- `best_bid` (float): Best bid price
- `best_ask` (float): Best ask price
- `mid_price` (float): Mid-price
- `spread` (float): Bid-ask spread
- `spread_bps` (float): Spread in basis points
- `total_bid_volume` (int): Total bid volume
- `total_ask_volume` (int): Total ask volume

**Methods:**

##### `get_depth(side: OrderSide, levels: int = 5) -> List[Tuple[float, int]]`

Gets market depth for specified side.

##### `get_market_impact(side: OrderSide, volume: int) -> Tuple[float, float]`

Calculates market impact for given order size.

**Returns:**
- `Tuple[float, float]`: (average_price, total_cost)

##### `is_crossed() -> bool`

Checks if market is crossed (bid >= ask).

##### `is_locked() -> bool`

Checks if market is locked (bid == ask).

---

## Trading Strategies

### Base Strategy

Abstract base class for all trading strategies.

```python
from src.strategies.base_strategy import BaseStrategy
```

#### Class: `BaseStrategy`

Abstract base class defining the strategy interface.

**Methods:**

##### `generate_signals(market_data: Dict, context: Optional[Dict] = None) -> List[Dict]`

Generates trading signals based on market data.

**Parameters:**
- `market_data` (Dict): Current market data
- `context` (Optional[Dict]): Additional context information

**Returns:**
- `List[Dict]`: List of trading signals

##### `update_position(trade_data: Dict) -> None`

Updates strategy position after trade execution.

##### `get_parameters() -> Dict`

Returns current strategy parameters.

##### `set_parameters(params: Dict) -> None`

Updates strategy parameters.

---

### Market Making Strategy

Liquidity provision strategy with inventory management.

```python
from src.strategies.market_making import MarketMakingStrategy
```

#### Class: `MarketMakingStrategy(BaseStrategy)`

Professional market making strategy with advanced features.

**Constructor:**
```python
MarketMakingStrategy(target_spread: float = 0.01, max_position: int = 1000,
                    inventory_target: int = 0, skew_adjustment: bool = True,
                    adverse_selection_protection: bool = True)
```

**Parameters:**
- `target_spread` (float): Target bid-ask spread
- `max_position` (int): Maximum position size
- `inventory_target` (int): Target inventory level
- `skew_adjustment` (bool): Enable inventory skewing
- `adverse_selection_protection` (bool): Enable adverse selection protection

**Methods:**

##### `generate_signals(market_data: Dict, context: Optional[Dict] = None) -> List[Dict]`

Generates market making signals.

**Returns:**
- List of signals with keys: 'side', 'price', 'quantity', 'order_type'

##### `calculate_fair_value(market_data: Dict) -> float`

Calculates fair value estimate.

##### `calculate_inventory_skew() -> float`

Calculates inventory-based price skew.

##### `update_spreads(market_data: Dict) -> None`

Updates spread based on market conditions.

**Example:**
```python
strategy = MarketMakingStrategy(
    target_spread=0.01,
    max_position=1000,
    inventory_target=0
)

# Generate signals
signals = strategy.generate_signals(market_data)
for signal in signals:
    print(f"Signal: {signal['side']} {signal['quantity']}@{signal['price']}")
```

---

### Liquidity Taking Strategy

Aggressive execution strategy for alpha capture.

```python
from src.strategies.liquidity_taking import LiquidityTakingStrategy
```

#### Class: `LiquidityTakingStrategy(BaseStrategy)`

Strategy for taking liquidity based on directional signals.

**Constructor:**
```python
LiquidityTakingStrategy(signal_threshold: float = 0.005, max_order_size: int = 500,
                       execution_delay: float = 0.1, signal_decay: float = 0.95)
```

**Parameters:**
- `signal_threshold` (float): Minimum signal strength to act
- `max_order_size` (int): Maximum single order size
- `execution_delay` (float): Delay before execution (seconds)
- `signal_decay` (float): Signal decay rate

**Methods:**

##### `calculate_signal_strength(market_data: Dict) -> float`

Calculates directional signal strength.

##### `determine_order_size(signal_strength: float) -> int`

Determines optimal order size based on signal.

**Example:**
```python
strategy = LiquidityTakingStrategy(
    signal_threshold=0.005,
    max_order_size=500
)

signals = strategy.generate_signals(market_data)
```

---

### ML Strategy

Machine learning-powered trading strategy.

```python
from src.strategies.ml_strategy import MLStrategy
```

#### Class: `MLStrategy(BaseStrategy)`

Strategy using machine learning models for prediction.

**Constructor:**
```python
MLStrategy(model_path: str, feature_config: str, prediction_threshold: float = 0.6,
          feature_window: int = 100, retrain_interval: int = 1000)
```

**Parameters:**
- `model_path` (str): Path to trained model file
- `feature_config` (str): Path to feature configuration
- `prediction_threshold` (float): Minimum prediction confidence
- `feature_window` (int): Window size for feature calculation
- `retrain_interval` (int): Trades between model retraining

**Methods:**

##### `calculate_features(market_data: Dict) -> np.ndarray`

Calculates feature vector from market data.

##### `make_prediction(features: np.ndarray) -> Tuple[float, float]`

Makes price/direction prediction.

**Returns:**
- `Tuple[float, float]`: (prediction, confidence)

##### `retrain_model(recent_data: List[Dict]) -> None`

Retrains model with recent data.

**Example:**
```python
strategy = MLStrategy(
    model_path='models/trained_model.pkl',
    feature_config='features/ml_features.yaml',
    prediction_threshold=0.7
)

# Model automatically calculates features and makes predictions
signals = strategy.generate_signals(market_data)
```

---

## Execution System

### Execution Simulator

Realistic order execution simulation with market impact.

```python
from src.execution.simulator import ExecutionSimulator
```

#### Class: `ExecutionSimulator`

Simulates realistic order execution with slippage and latency.

**Constructor:**
```python
ExecutionSimulator(order_book: OrderBook, portfolio: Portfolio, 
                  slippage_model: str = 'linear', latency_model: str = 'constant')
```

**Parameters:**
- `order_book` (OrderBook): Order book instance
- `portfolio` (Portfolio): Portfolio instance
- `slippage_model` (str): Slippage model type
- `latency_model` (str): Latency model type

**Methods:**

##### `execute_order(order: Order) -> Optional[Dict]`

Executes a single order.

**Returns:**
- `Optional[Dict]`: Fill information or None if not filled

##### `run_backtest(data: pd.DataFrame) -> Dict`

Runs complete backtest on historical data.

**Parameters:**
- `data` (pd.DataFrame): Historical market data

**Returns:**
- `Dict`: Backtest results including PnL, metrics, and trades

##### `calculate_slippage(order: Order, market_data: Dict) -> float`

Calculates slippage for order execution.

**Example:**
```python
simulator = ExecutionSimulator(order_book, portfolio)

# Execute single order
order = Order('ORDER001', 'BTCUSDT', OrderSide.BUY, OrderType.MARKET, 100, timestamp)
fill = simulator.execute_order(order)

# Run backtest
results = simulator.run_backtest(historical_data)
print(f"Total PnL: ${results['total_pnl']:.2f}")
```

---

### Matching Engine

High-performance order matching engine.

```python
from src.execution.matching_engine import MatchingEngine
```

#### Class: `MatchingEngine`

Professional-grade order matching with advanced features.

**Constructor:**
```python
MatchingEngine(symbol: str, tick_size: float = 0.01, lot_size: int = 1)
```

**Methods:**

##### `process_order(order: Order) -> Tuple[List[Trade], OrderUpdate]`

Processes order and returns trades and status update.

##### `cancel_order(order_id: str) -> bool`

Cancels order by ID.

##### `get_market_snapshot() -> BookSnapshot`

Gets current market snapshot.

##### `get_statistics() -> Dict`

Returns engine performance statistics.

**Example:**
```python
engine = MatchingEngine('BTCUSDT')

order = Order('ORDER001', 'BTCUSDT', OrderSide.BUY, OrderType.LIMIT, 
              100, timestamp, price=50000.0)
trades, update = engine.process_order(order)

print(f"Generated {len(trades)} trades")
```

---

## Performance Analytics

### Performance Analyzer

Comprehensive performance and risk analytics.

```python
from src.performance.metrics import PerformanceAnalyzer
```

#### Class: `PerformanceAnalyzer`

Calculates trading performance metrics and risk measures.

**Constructor:**
```python
PerformanceAnalyzer(initial_capital: float = 100000.0, risk_free_rate: float = 0.02)
```

**Parameters:**
- `initial_capital` (float): Starting capital
- `risk_free_rate` (float): Risk-free rate for Sharpe calculation

**Methods:**

##### `calculate_metrics() -> PerformanceMetrics`

Calculates comprehensive performance metrics.

**Returns:**
- `PerformanceMetrics`: Object containing all metrics

##### `calculate_sharpe_ratio() -> float`

Calculates Sharpe ratio.

##### `calculate_max_drawdown() -> float`

Calculates maximum drawdown.

##### `calculate_var(confidence: float = 0.05) -> float`

Calculates Value at Risk.

##### `calculate_win_rate() -> float`

Calculates percentage of winning trades.

**Example:**
```python
analyzer = PerformanceAnalyzer(initial_capital=100000)

# Add trade data
for trade in trades:
    analyzer.add_trade(trade)

metrics = analyzer.calculate_metrics()
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

---

### Portfolio

Portfolio management and P&L tracking.

```python
from src.performance.portfolio import Portfolio
```

#### Class: `Portfolio`

Manages positions, cash, and P&L calculations.

**Constructor:**
```python
Portfolio(initial_balance: float = 100000.0, currency: str = 'USD')
```

**Methods:**

##### `update_position(fill_data: Dict) -> None`

Updates position based on trade execution.

##### `get_position(symbol: str) -> float`

Gets current position for symbol.

##### `get_total_pnl() -> float`

Gets total unrealized + realized P&L.

##### `get_summary() -> Dict`

Gets portfolio summary with positions and P&L.

##### `get_trade_history() -> List[Dict]`

Gets complete trade history.

**Example:**
```python
portfolio = Portfolio(initial_balance=100000)

# Update position after trade
fill = {
    'symbol': 'BTCUSDT',
    'side': OrderSide.BUY,
    'quantity': 100,
    'price': 50000.0,
    'timestamp': pd.Timestamp.now()
}
portfolio.update_position(fill)

summary = portfolio.get_summary()
print(f"Total Value: ${summary['total_value']:,.2f}")
```

---

### Risk Manager

Real-time risk management and monitoring.

```python
from src.performance.risk_manager import RiskManager
```

#### Class: `RiskManager`

Monitors and controls trading risks.

**Constructor:**
```python
RiskManager(max_position_size: int = 1000, max_drawdown: float = 0.05,
           var_threshold: float = 0.02, circuit_breaker: bool = True)
```

**Methods:**

##### `validate_order(order: Order) -> Dict`

Validates order against risk limits.

**Returns:**
- `Dict`: {'approved': bool, 'reason': str}

##### `calculate_risk_metrics() -> Dict`

Calculates current risk metrics.

##### `check_circuit_breaker() -> bool`

Checks if circuit breaker should trigger.

**Example:**
```python
risk_manager = RiskManager(max_position_size=1000, max_drawdown=0.05)

# Validate order
order = Order('ORDER001', 'BTCUSDT', OrderSide.BUY, OrderType.LIMIT, 2000, timestamp)
validation = risk_manager.validate_order(order)

if not validation['approved']:
    print(f"Order rejected: {validation['reason']}")
```

---

## Real-Time Trading

### Real-Time Trading System

Complete real-time trading system orchestrator.

```python
from src.realtime.trading_system import RealTimeTradingSystem
```

#### Class: `RealTimeTradingSystem`

Orchestrates real-time trading components.

**Constructor:**
```python
RealTimeTradingSystem(config: Optional[RealTimeConfig] = None)
```

**Methods:**

##### `async start() -> None`

Starts the real-time trading system.

##### `async stop() -> None`

Stops the trading system gracefully.

##### `add_strategy(strategy: BaseStrategy) -> None`

Adds trading strategy to system.

##### `get_system_metrics() -> SystemMetrics`

Gets system performance metrics.

**Example:**
```python
import asyncio
from src.realtime.config import RealTimeConfig

config = RealTimeConfig.from_file('config.yaml')
system = RealTimeTradingSystem(config)

async def main():
    await system.start()
    # System runs until stopped
    
asyncio.run(main())
```

---

### Data Feeds

Real-time market data feeds.

```python
from src.realtime.data_feeds import WebSocketDataFeed, RESTDataFeed
```

#### Class: `WebSocketDataFeed`

WebSocket-based real-time data feed.

**Constructor:**
```python
WebSocketDataFeed(url: str, symbols: List[str], reconnect_attempts: int = 5)
```

**Methods:**

##### `async connect() -> None`

Establishes WebSocket connection.

##### `async disconnect() -> None`

Closes WebSocket connection.

##### `subscribe(symbol: str) -> None`

Subscribes to symbol updates.

##### `set_message_handler(handler: Callable) -> None`

Sets message processing handler.

**Example:**
```python
feed = WebSocketDataFeed(
    url='wss://stream.binance.com:9443/ws/btcusdt@ticker',
    symbols=['BTCUSDT', 'ETHUSDT']
)

def handle_message(message):
    print(f"Received: {message}")

feed.set_message_handler(handle_message)
await feed.connect()
```

---

## Data Processing

### Data Ingestion

Data loading and preprocessing utilities.

```python
from src.data.ingestion import DataIngestion
```

#### Class: `DataIngestion`

Handles data loading from various sources.

**Methods:**

##### `load_csv(file_path: str, **kwargs) -> pd.DataFrame`

Loads data from CSV file.

##### `load_parquet(file_path: str) -> pd.DataFrame`

Loads data from Parquet file.

##### `validate_data(df: pd.DataFrame) -> bool`

Validates data format and completeness.

##### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

Preprocesses and cleans data.

**Example:**
```python
ingestion = DataIngestion()

# Load and preprocess data
data = ingestion.load_csv('data/market_data.csv')
data = ingestion.preprocess_data(data)

if ingestion.validate_data(data):
    print("Data ready for processing")
```

---

### Data Preprocessor

Data cleaning and transformation utilities.

```python
from src.data.preprocessor import DataPreprocessor
```

#### Class: `DataPreprocessor`

Advanced data preprocessing and feature engineering.

**Methods:**

##### `clean_and_validate(df: pd.DataFrame) -> pd.DataFrame`

Cleans and validates market data.

##### `calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame`

Calculates technical indicators.

##### `resample_data(df: pd.DataFrame, frequency: str) -> pd.DataFrame`

Resamples data to different frequency.

---

## Visualization

### Dashboard

Real-time trading dashboard.

```python
from src.visualization.realtime_dashboard import RealtimeDashboard
```

#### Class: `RealtimeDashboard`

Web-based real-time trading dashboard.

**Constructor:**
```python
RealtimeDashboard(port: int = 8080, debug: bool = False)
```

**Methods:**

##### `start_server() -> None`

Starts the dashboard server.

##### `add_chart(chart_config: Dict) -> None`

Adds chart to dashboard.

##### `update_data(data: Dict) -> None`

Updates dashboard with new data.

**Example:**
```python
dashboard = RealtimeDashboard(port=8080)

# Add market data chart
chart_config = {
    'type': 'line',
    'title': 'Price Chart',
    'x_axis': 'timestamp',
    'y_axis': 'price'
}
dashboard.add_chart(chart_config)
dashboard.start_server()
```

---

### Charts

Chart generation utilities.

```python
from src.visualization.charts import ChartGenerator
```

#### Class: `ChartGenerator`

Generates various types of trading charts.

**Methods:**

##### `create_price_chart(data: pd.DataFrame) -> Dict`

Creates price chart configuration.

##### `create_volume_chart(data: pd.DataFrame) -> Dict`

Creates volume chart configuration.

##### `create_pnl_chart(trades: List[Dict]) -> Dict`

Creates P&L chart configuration.

---

## Utilities

### Logger

Centralized logging utilities.

```python
from src.utils.logger import get_logger, setup_logging
```

#### Functions:

##### `get_logger(name: str) -> logging.Logger`

Gets configured logger instance.

##### `setup_logging(level: str = 'INFO', log_file: str = None) -> None`

Sets up logging configuration.

**Example:**
```python
logger = get_logger(__name__)
logger.info("Starting trading system")
logger.error("Connection failed", exc_info=True)
```

---

### Performance Monitor

System performance monitoring utilities.

```python
from src.utils.performance_monitor import PerformanceMonitor
```

#### Class: `PerformanceMonitor`

Monitors system performance metrics.

**Methods:**

##### `start_monitoring() -> None`

Starts performance monitoring.

##### `get_metrics() -> Dict`

Gets current performance metrics.

##### `log_event(event: str, data: Dict) -> None`

Logs performance event.

---

### Configuration

Configuration management utilities.

```python
from src.utils.config import ConfigManager
```

#### Class: `ConfigManager`

Manages application configuration.

**Methods:**

##### `load_config(file_path: str) -> Dict`

Loads configuration from file.

##### `validate_config(config: Dict) -> bool`

Validates configuration structure.

##### `get_parameter(key: str, default=None) -> Any`

Gets configuration parameter.

---

## Error Handling

### Exception Classes

Custom exception classes for specific error conditions.

```python
from src.utils.exceptions import (
    TradingSystemError, InvalidOrderError, 
    RiskLimitExceededError, DataValidationError
)
```

#### Exceptions:

- `TradingSystemError`: Base exception for trading system errors
- `InvalidOrderError`: Invalid order parameters
- `RiskLimitExceededError`: Risk limits exceeded
- `DataValidationError`: Data validation failures
- `ConnectionError`: Network/connection issues

---

## Type Hints

Common type definitions used throughout the API.

```python
from typing import Dict, List, Optional, Union, Tuple, Callable
from src.utils.types import (
    OrderDict, TradeDict, MarketDataDict, 
    SignalDict, ConfigDict
)
```

### Type Aliases:

- `OrderDict`: Dictionary representing order data
- `TradeDict`: Dictionary representing trade data
- `MarketDataDict`: Dictionary representing market data
- `SignalDict`: Dictionary representing trading signal
- `ConfigDict`: Dictionary representing configuration

---

## Usage Examples

### Complete Trading System Setup

```python
import asyncio
from src.engine.order_book import OrderBook
from src.strategies.market_making import MarketMakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.performance.portfolio import Portfolio
from src.realtime.trading_system import RealTimeTradingSystem

# Create components
order_book = OrderBook('BTCUSDT')
portfolio = Portfolio(initial_balance=100000)
strategy = MarketMakingStrategy(target_spread=0.01, max_position=1000)
simulator = ExecutionSimulator(order_book, portfolio)

# Set up real-time system
trading_system = RealTimeTradingSystem()
trading_system.add_strategy(strategy)

# Start trading
async def main():
    await trading_system.start()

asyncio.run(main())
```

### Backtesting Example

```python
import pandas as pd
from src.data.ingestion import DataIngestion
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.performance.metrics import PerformanceAnalyzer

# Load data
ingestion = DataIngestion()
data = ingestion.load_csv('data/historical_data.csv')

# Set up strategy and execution
strategy = LiquidityTakingStrategy(signal_threshold=0.005)
simulator = ExecutionSimulator(order_book, portfolio)

# Run backtest
results = simulator.run_backtest(data)

# Analyze performance
analyzer = PerformanceAnalyzer()
metrics = analyzer.calculate_metrics()

print(f"Total Return: {metrics.total_return:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
```

---

## Best Practices

### Error Handling
Always wrap trading operations in try-catch blocks:

```python
try:
    trades = order_book.add_order(order)
except InvalidOrderError as e:
    logger.error(f"Invalid order: {e}")
except RiskLimitExceededError as e:
    logger.warning(f"Risk limit exceeded: {e}")
```

### Resource Management
Use context managers for resource cleanup:

```python
from src.utils.context_managers import trading_session

with trading_session() as session:
    # Trading operations
    pass
# Resources automatically cleaned up
```

### Async Programming
Use async/await for real-time operations:

```python
async def process_market_data(data_feed):
    async for message in data_feed:
        await process_message(message)
```

---

For more examples and detailed usage, see the [examples](../examples/) directory and [notebooks](../notebooks/) folder.
