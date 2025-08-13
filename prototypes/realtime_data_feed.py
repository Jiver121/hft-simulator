"""
Real-Time Data Feed Prototype

This prototype demonstrates the foundation for real-time market data integration,
showing how the HFT simulator can be extended to handle live data streams.

Key Features:
- WebSocket data feed simulation
- Asynchronous data processing
- Integration with existing order book engine
- Real-time strategy execution
"""

import asyncio
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from collections import deque
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.engine.order_book import OrderBook
    from src.engine.order_types import Order, MarketDataPoint
    from src.utils.constants import OrderSide, OrderType
    from src.strategies.market_making import MarketMakingStrategy
except ImportError:
    # Fallback for simplified demo
    print("Note: Running in simplified mode without full HFT simulator imports")
    OrderBook = None
    Order = None
    MarketDataPoint = None
    OrderSide = None
    OrderType = None
    MarketMakingStrategy = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple MarketDataPoint for demo if not imported
if MarketDataPoint is None:
    @dataclass
    class MarketDataPoint:
        timestamp: pd.Timestamp
        price: float
        volume: int
        best_bid: Optional[float] = None
        best_ask: Optional[float] = None
        bid_volume: Optional[int] = None
        ask_volume: Optional[int] = None
        metadata: Dict[str, Any] = None
        
        @property
        def spread(self) -> Optional[float]:
            if self.best_bid is not None and self.best_ask is not None:
                return self.best_ask - self.best_bid
            return None

@dataclass
class RealTimeConfig:
    """Configuration for real-time data feed"""
    symbols: List[str]
    update_frequency_ms: int = 100  # Update frequency in milliseconds
    buffer_size: int = 1000
    enable_strategies: bool = True
    risk_limits: Dict[str, float] = None

class MockDataFeed:
    """
    Mock real-time data feed that simulates live market data
    
    In production, this would connect to actual exchange feeds:
    - WebSocket connections to exchanges
    - FIX protocol feeds
    - Market data vendor APIs (Bloomberg, Refinitiv)
    """
    
    def __init__(self, symbols: List[str], update_frequency_ms: int = 100):
        self.symbols = symbols
        self.update_frequency_ms = update_frequency_ms
        self.subscribers: List[Callable] = []
        self.running = False
        
        # Initialize price tracking
        self.current_prices = {symbol: 100.0 for symbol in symbols}
        self.price_history = {symbol: deque(maxlen=100) for symbol in symbols}
        
        logger.info(f"MockDataFeed initialized for symbols: {symbols}")
    
    def subscribe(self, callback: Callable[[MarketDataPoint], None]) -> None:
        """Subscribe to real-time market data updates"""
        self.subscribers.append(callback)
        logger.info(f"Added subscriber, total: {len(self.subscribers)}")
    
    async def start(self) -> None:
        """Start the real-time data feed"""
        self.running = True
        logger.info("Starting real-time data feed...")
        
        while self.running:
            try:
                # Generate market data for each symbol
                for symbol in self.symbols:
                    market_data = self._generate_market_data(symbol)
                    
                    # Notify all subscribers
                    for callback in self.subscribers:
                        try:
                            await callback(market_data)
                        except Exception as e:
                            logger.error(f"Error in subscriber callback: {e}")
                
                # Wait for next update
                await asyncio.sleep(self.update_frequency_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in data feed: {e}")
                await asyncio.sleep(1)  # Brief pause before retry
    
    def stop(self) -> None:
        """Stop the data feed"""
        self.running = False
        logger.info("Data feed stopped")
    
    def _generate_market_data(self, symbol: str) -> MarketDataPoint:
        """Generate realistic market data point"""
        # Get current price
        current_price = self.current_prices[symbol]
        
        # Generate price movement (random walk with mean reversion)
        price_change = np.random.normal(0, 0.001)  # Small random changes
        mean_reversion = -0.0001 * (current_price - 100.0)  # Revert to 100
        
        new_price = current_price + price_change + mean_reversion
        new_price = max(95.0, min(105.0, new_price))  # Keep in bounds
        
        self.current_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Generate bid/ask with realistic spread
        spread = np.random.uniform(0.01, 0.03)
        best_bid = new_price - spread/2
        best_ask = new_price + spread/2
        
        # Generate volume
        volume = np.random.randint(100, 1000)
        bid_volume = np.random.randint(500, 2000)
        ask_volume = np.random.randint(500, 2000)
        
        return MarketDataPoint(
            timestamp=pd.Timestamp.now(),
            price=new_price,
            volume=volume,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            metadata={'symbol': symbol, 'source': 'mock_feed'}
        )

class RealTimeOrderBook:
    """
    Real-time order book that processes live market data
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.order_book = OrderBook(symbol) if OrderBook else None
        self.last_update = None
        self.update_count = 0
        
        logger.info(f"RealTimeOrderBook initialized for {symbol}")
    
    async def process_market_data(self, market_data: MarketDataPoint) -> None:
        """Process incoming market data and update order book"""
        try:
            self.last_update = market_data.timestamp
            self.update_count += 1
            
            # In a real system, this would process actual order book updates
            # For now, we'll simulate by updating our internal state
            
            if self.update_count % 100 == 0:  # Log every 100 updates
                logger.info(f"{self.symbol}: Price={market_data.price:.4f}, "
                           f"Spread={market_data.spread:.4f}, Updates={self.update_count}")
        
        except Exception as e:
            logger.error(f"Error processing market data for {self.symbol}: {e}")

class RealTimeStrategyEngine:
    """
    Real-time strategy execution engine
    """
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.strategies: Dict[str, Any] = {}
        self.order_books: Dict[str, RealTimeOrderBook] = {}
        self.active_orders: Dict[str, Order] = {}
        
        # Initialize order books for each symbol
        for symbol in config.symbols:
            self.order_books[symbol] = RealTimeOrderBook(symbol)
        
        logger.info(f"RealTimeStrategyEngine initialized for {len(config.symbols)} symbols")
    
    def add_strategy(self, name: str, strategy: Any) -> None:
        """Add a trading strategy"""
        self.strategies[name] = strategy
        logger.info(f"Added strategy: {name}")
    
    async def process_market_data(self, market_data: MarketDataPoint) -> None:
        """Process market data and execute strategies"""
        try:
            symbol = market_data.metadata.get('symbol')
            if not symbol or symbol not in self.order_books:
                return
            
            # Update order book
            await self.order_books[symbol].process_market_data(market_data)
            
            # Execute strategies if enabled
            if self.config.enable_strategies:
                await self._execute_strategies(market_data)
        
        except Exception as e:
            logger.error(f"Error in strategy engine: {e}")
    
    async def _execute_strategies(self, market_data: MarketDataPoint) -> None:
        """Execute all active strategies"""
        for strategy_name, strategy in self.strategies.items():
            try:
                # In a real system, this would generate and execute orders
                # For now, we'll just log strategy activity
                
                if hasattr(strategy, 'should_trade'):
                    if strategy.should_trade(market_data):
                        logger.info(f"Strategy {strategy_name} generated signal for "
                                   f"{market_data.metadata.get('symbol')} at {market_data.price:.4f}")
            
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_name}: {e}")

class RealTimeSimulator:
    """
    Main real-time simulation coordinator
    """
    
    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.data_feed = MockDataFeed(config.symbols, config.update_frequency_ms)
        self.strategy_engine = RealTimeStrategyEngine(config)
        self.running = False
        
        # Connect data feed to strategy engine
        self.data_feed.subscribe(self.strategy_engine.process_market_data)
        
        logger.info("RealTimeSimulator initialized")
    
    def add_strategy(self, name: str, strategy: Any) -> None:
        """Add a trading strategy"""
        self.strategy_engine.add_strategy(name, strategy)
    
    async def start(self) -> None:
        """Start the real-time simulation"""
        logger.info("Starting real-time simulation...")
        self.running = True
        
        # Start data feed
        data_feed_task = asyncio.create_task(self.data_feed.start())
        
        try:
            await data_feed_task
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the simulation"""
        logger.info("Stopping real-time simulation...")
        self.running = False
        self.data_feed.stop()

# Simple strategy for demonstration
class SimpleRealTimeStrategy:
    """Simple strategy that demonstrates real-time signal generation"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_history = deque(maxlen=20)
        self.signal_count = 0
    
    def should_trade(self, market_data: MarketDataPoint) -> bool:
        """Simple mean reversion logic"""
        if market_data.metadata.get('symbol') != self.symbol:
            return False
        
        self.price_history.append(market_data.price)
        
        if len(self.price_history) < 10:
            return False
        
        # Simple mean reversion: trade when price deviates from recent average
        recent_avg = sum(list(self.price_history)[-10:]) / 10
        current_price = market_data.price
        
        deviation = abs(current_price - recent_avg) / recent_avg
        
        if deviation > 0.002:  # 0.2% deviation threshold
            self.signal_count += 1
            return True
        
        return False

async def main():
    """Main demonstration function"""
    print("[REALTIME] Real-Time HFT Simulator Prototype")
    print("=" * 50)
    
    # Configuration
    config = RealTimeConfig(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        update_frequency_ms=100,  # 10 updates per second for demo
        enable_strategies=True
    )
    
    # Create simulator
    simulator = RealTimeSimulator(config)
    
    # Add simple strategies
    for symbol in config.symbols:
        strategy = SimpleRealTimeStrategy(symbol)
        simulator.add_strategy(f"MeanReversion_{symbol}", strategy)
    
    print(f"[CONFIG] Configured for symbols: {config.symbols}")
    print(f"[SPEED] Update frequency: {config.update_frequency_ms}ms")
    print(f"[STRATEGY] Strategies enabled: {config.enable_strategies}")
    print("\n[START] Starting real-time simulation for 5 seconds...")
    
    try:
        # Run for 5 seconds for demonstration
        start_task = asyncio.create_task(simulator.start())
        await asyncio.sleep(5)  # Run for 5 seconds
        await simulator.stop()
        print("\n[STOP] Demo completed after 5 seconds")
    except KeyboardInterrupt:
        print("\n[STOP] Simulation stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
    
    print("[SUCCESS] Real-time simulation completed")

if __name__ == "__main__":
    asyncio.run(main())