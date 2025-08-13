"""
HFT Order Book Simulator

A comprehensive high-frequency trading simulation environment for educational,
research, and strategy development purposes.

This package provides:
- Data ingestion and preprocessing for HFT datasets
- Real-time order book reconstruction and management
- Order execution simulation with realistic latency and slippage
- Trading strategy implementations (market making, liquidity taking)
- Performance tracking and risk management
- Visualization and reporting tools

Example Usage:
    >>> from src.engine.order_book import OrderBook
    >>> from src.strategies.market_making import MarketMakingStrategy
    >>> from src.execution.simulator import ExecutionSimulator
    >>> 
    >>> # Initialize components
    >>> order_book = OrderBook()
    >>> strategy = MarketMakingStrategy()
    >>> simulator = ExecutionSimulator(order_book, strategy)
    >>> 
    >>> # Run backtest
    >>> results = simulator.run_backtest('data/sample/hft_data.csv')
    >>> print(f"Total PnL: ${results.total_pnl:.2f}")

For detailed documentation and tutorials, see the notebooks/ directory.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import key classes for easy access
from .engine.order_book import OrderBook
from .strategies.base_strategy import BaseStrategy
from .execution.simulator import ExecutionSimulator
from .performance.metrics import PerformanceMetrics
from .data.ingestion import DataIngestion

__all__ = [
    'OrderBook',
    'BaseStrategy', 
    'ExecutionSimulator',
    'PerformanceMetrics',
    'DataIngestion',
    '__version__',
    '__author__',
    '__email__',
]