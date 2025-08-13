"""
Pytest Configuration and Fixtures for HFT Simulator Tests

This module provides shared test fixtures and configuration for all tests
in the HFT simulator test suite.

Educational Notes:
- Fixtures provide reusable test data and setup
- Shared fixtures reduce code duplication across tests
- Proper test configuration ensures consistent test environment
- Fixtures can be scoped to control their lifecycle
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

from src.engine.order_book import OrderBook
from src.engine.order_types import Order, OrderSide, OrderType, Trade
from src.engine.market_data import BookSnapshot, MarketData
from src.performance.portfolio import Portfolio
from src.performance.risk_manager import RiskManager
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from src.strategies.liquidity_taking import LiquidityTakingStrategy, LiquidityTakingConfig


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_symbol():
    """Standard test symbol"""
    return "AAPL"


@pytest.fixture
def sample_timestamp():
    """Standard test timestamp"""
    return pd.Timestamp('2024-01-01 09:30:00')


@pytest.fixture
def empty_order_book(sample_symbol):
    """Empty order book for testing"""
    return OrderBook(sample_symbol)


@pytest.fixture
def populated_order_book(sample_symbol, sample_timestamp):
    """Order book with sample orders"""
    import uuid
    import time
    from src.engine.order_types import clear_order_id_registry
    
    # Clear the registry to avoid conflicts in tests
    clear_order_id_registry()
    
    order_book = OrderBook(sample_symbol)
    
    # Generate unique order IDs using UUID and timestamp
    timestamp_ns = int(time.time_ns())
    
    # Add sample bid orders with unique IDs
    bid_orders = [
        Order(f"BID001_{timestamp_ns}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.BUY, OrderType.LIMIT, 100, 149.95, sample_timestamp),
        Order(f"BID002_{timestamp_ns + 1}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.BUY, OrderType.LIMIT, 200, 149.90, sample_timestamp),
        Order(f"BID003_{timestamp_ns + 2}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.BUY, OrderType.LIMIT, 150, 149.85, sample_timestamp),
    ]
    
    # Add sample ask orders with unique IDs
    ask_orders = [
        Order(f"ASK001_{timestamp_ns + 3}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.SELL, OrderType.LIMIT, 100, 150.05, sample_timestamp),
        Order(f"ASK002_{timestamp_ns + 4}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.SELL, OrderType.LIMIT, 200, 150.10, sample_timestamp),
        Order(f"ASK003_{timestamp_ns + 5}_{str(uuid.uuid4())[:8]}", sample_symbol, OrderSide.SELL, OrderType.LIMIT, 150, 150.15, sample_timestamp),
    ]
    
    # Add orders to book
    for order in bid_orders + ask_orders:
        order_book.add_order(order)
    
    return order_book


@pytest.fixture
def sample_book_snapshot(sample_symbol, sample_timestamp):
    """Sample book snapshot for testing"""
    from src.engine.order_types import PriceLevel
    
    # Create price levels for bids and asks
    bids = [
        PriceLevel(149.95, 100, 1),
        PriceLevel(149.90, 200, 1),
        PriceLevel(149.85, 150, 1)
    ]
    asks = [
        PriceLevel(150.05, 100, 1),
        PriceLevel(150.10, 200, 1),
        PriceLevel(150.15, 150, 1)
    ]
    
    return BookSnapshot(
        symbol=sample_symbol,
        timestamp=sample_timestamp,
        bids=bids,
        asks=asks,
        last_trade_price=150.0,
        last_trade_volume=100,
        last_trade_timestamp=sample_timestamp
    )


@pytest.fixture
def sample_market_data(sample_symbol, sample_book_snapshot):
    """Sample market data for testing"""
    market_data = MarketData(symbol=sample_symbol)
    market_data.update_snapshot(sample_book_snapshot)
    return market_data


@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing"""
    return Portfolio(initial_cash=100000.0, name="Test Portfolio")


@pytest.fixture
def sample_risk_manager():
    """Sample risk manager for testing"""
    return RiskManager(initial_capital=100000.0)


@pytest.fixture
def market_making_strategy(sample_symbol, sample_portfolio):
    """Market making strategy for testing"""
    config = MarketMakingConfig(
        spread_target=0.02,
        position_limit=1000,
        inventory_target=0,
        risk_aversion=0.1
    )
    
    return MarketMakingStrategy(
        symbols=[sample_symbol],
        portfolio=sample_portfolio,
        config=config
    )


@pytest.fixture
def liquidity_taking_strategy(sample_symbol, sample_portfolio):
    """Liquidity taking strategy for testing"""
    config = LiquidityTakingConfig(
        momentum_threshold=0.01,
        mean_reversion_threshold=0.02,
        volume_threshold=500,
        position_limit=500
    )
    
    return LiquidityTakingStrategy(
        symbols=[sample_symbol],
        portfolio=sample_portfolio,
        config=config
    )


@pytest.fixture
def sample_trades(sample_symbol, sample_timestamp):
    """Sample trades for testing"""
    return [
        Trade("T001", sample_symbol, 100, 150.00, sample_timestamp, "BUY001", "SELL001"),
        Trade("T002", sample_symbol, 200, 150.05, sample_timestamp, "BUY002", "SELL002"),
        Trade("T003", sample_symbol, 150, 149.95, sample_timestamp, "BUY003", "SELL003"),
    ]


@pytest.fixture
def price_history():
    """Sample price history for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    base_price = 150.0
    
    # Generate realistic price movement
    np.random.seed(42)  # For reproducible tests
    price_changes = np.random.normal(0, 0.01, 100)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    return pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': np.random.randint(100, 1000, 100)
    })


@pytest.fixture
def ohlcv_data():
    """Sample OHLCV data for testing"""
    dates = pd.date_range('2024-01-01', periods=50, freq='1H')
    base_price = 150.0
    
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.01, 50)
    close_prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLC from close prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.005, 50)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.005, 50)))
    volumes = np.random.randint(1000, 10000, 50)
    
    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


@pytest.fixture
def performance_data():
    """Sample performance data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    base_value = 100000.0
    
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, 100)  # Small positive drift
    values = base_value * np.exp(np.cumsum(returns))
    
    return [(timestamp, value) for timestamp, value in zip(dates, values)]


@pytest.fixture
def large_dataset():
    """Large dataset for performance testing"""
    n_points = 10000
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1S')
    base_price = 150.0
    
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.0001, n_points)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'AAPL',
        'price': prices,
        'volume': np.random.randint(100, 1000, n_points),
        'bid_price': prices - 0.01,
        'ask_price': prices + 0.01,
        'bid_volume': np.random.randint(100, 1000, n_points),
        'ask_volume': np.random.randint(100, 1000, n_points)
    })


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on their location
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# Custom assertions
def assert_order_book_valid(order_book: OrderBook):
    """Assert that order book is in valid state"""
    # Check bid ordering (descending)
    bid_prices = list(order_book.bids.keys())
    assert bid_prices == sorted(bid_prices, reverse=True), "Bids should be in descending order"
    
    # Check ask ordering (ascending)
    ask_prices = list(order_book.asks.keys())
    assert ask_prices == sorted(ask_prices), "Asks should be in ascending order"
    
    # Check spread is non-negative
    if order_book.best_bid and order_book.best_ask:
        spread = order_book.best_ask[0] - order_book.best_bid[0]
        assert spread >= 0, "Spread should be non-negative"


def assert_portfolio_valid(portfolio: Portfolio):
    """Assert that portfolio is in valid state"""
    assert portfolio.total_value >= 0, "Portfolio value should be non-negative"
    assert portfolio.current_cash >= 0, "Cash should be non-negative"
    
    # Check P&L consistency
    calculated_pnl = portfolio.total_realized_pnl + portfolio.total_unrealized_pnl
    assert abs(calculated_pnl - portfolio.total_pnl) < 0.01, "P&L calculation should be consistent"


def assert_trade_valid(trade: Trade):
    """Assert that trade is valid"""
    assert trade.volume > 0, "Trade volume should be positive"
    assert trade.price > 0, "Trade price should be positive"
    assert trade.timestamp is not None, "Trade should have timestamp"
    assert trade.symbol is not None, "Trade should have symbol"


# Performance test helpers
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.name} took {self.duration:.3f} seconds")


@pytest.fixture
def performance_timer():
    """Performance timer fixture"""
    return PerformanceTimer