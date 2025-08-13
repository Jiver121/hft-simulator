"""
Execution Simulator for HFT Simulator

This module provides the main execution simulator that orchestrates the entire
trading simulation process, including data replay, strategy execution, and
performance tracking.

Educational Notes:
- The execution simulator is the main orchestrator of the HFT simulation
- It coordinates data ingestion, order book updates, strategy decisions, and trade execution
- Real-time simulation allows testing strategies under realistic market conditions
- Backtesting capabilities enable historical strategy validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Iterator, Callable, Union
from datetime import datetime, timedelta
import time
from pathlib import Path
import warnings
import sys

from src.utils.logger import get_logger, log_performance, log_memory_usage
from src.utils.helpers import Timer, format_price, format_volume
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.data.ingestion import DataIngestion
from src.data.preprocessor import DataPreprocessor
from src.engine.order_book import OrderBook
from src.engine.order_types import Order, Trade, OrderUpdate
from src.engine.market_data import BookSnapshot, MarketData
from .matching_engine import MatchingEngine, MultiSymbolMatchingEngine
from .fill_models import FillModel, RealisticFillModel


class BacktestResult:
    """
    Container for backtest results and performance metrics
    
    This class stores all the results from a backtest run including
    trades, performance metrics, and detailed analytics.
    """
    
    def __init__(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
        self.symbol = symbol
        self.start_time = start_time
        self.end_time = end_time
        self.duration = end_time - start_time
        
        # Trading results
        self.trades: List[Trade] = []
        self.orders: List[Order] = []
        self.snapshots: List[BookSnapshot] = []
        
        # Performance metrics
        self.total_pnl = 0.0
        self.total_volume = 0
        self.total_trades = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Execution statistics
        self.fill_rate = 0.0
        self.avg_slippage = 0.0
        self.avg_latency_us = 0.0
        
        # Market statistics
        self.market_returns = 0.0
        self.strategy_returns = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        
        # Additional metrics
        self.metadata: Dict[str, Any] = {}
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the results"""
        self.trades.append(trade)
        self.total_trades = len(self.trades)  # Ensure consistency
        self.total_volume += trade.volume
    
    def add_order(self, order: Order) -> None:
        """Add an order to the results"""
        self.orders.append(order)
    
    def add_snapshot(self, snapshot: BookSnapshot) -> None:
        """Add a market snapshot to the results"""
        self.snapshots.append(snapshot)
    
    def calculate_metrics(self) -> None:
        """Calculate performance metrics from trades and orders"""
        if not self.trades:
            return
        
        # Initialize position tracking and P&L calculation
        position = 0  # Current position (positive = long, negative = short)
        position_queue = []  # Queue of buy trades for FIFO P&L calculation
        cumulative_pnl = 0.0
        pnl_series = []
        trade_pnls = []  # Individual trade P&L for win rate calculation
        
        for trade in self.trades:
            trade_pnl = 0.0
            
            # Determine if this trade is a buy or sell from our perspective
            # We need to check if we are the buy_order_id or sell_order_id owner
            # For simplicity, we'll use the aggressor side to determine direction
            if trade.is_buy_aggressor():
                # This is a buy trade - adds to our position
                position += trade.volume
                # Store buy trade info for later P&L calculation
                position_queue.append({
                    'volume': trade.volume,
                    'price': trade.price,
                    'timestamp': trade.timestamp
                })
            else:
                # This is a sell trade - reduces our position
                sell_volume = trade.volume
                sell_price = trade.price
                
                # Calculate realized P&L by matching against oldest buys (FIFO)
                while sell_volume > 0 and position_queue:
                    buy_trade = position_queue[0]
                    
                    # Determine how much of this buy trade to close
                    close_volume = min(sell_volume, buy_trade['volume'])
                    
                    # Calculate P&L: (sell_price - buy_price) * volume
                    pnl_for_this_close = (sell_price - buy_trade['price']) * close_volume
                    trade_pnl += pnl_for_this_close
                    
                    # Update buy trade remaining volume
                    buy_trade['volume'] -= close_volume
                    sell_volume -= close_volume
                    position -= close_volume
                    
                    # Remove buy trade if fully consumed
                    if buy_trade['volume'] <= 0:
                        position_queue.pop(0)
                
                # If we still have sell volume but no buys to match against,
                # we're going short (not typical in backtests but handle it)
                if sell_volume > 0:
                    position -= sell_volume
                    # For short positions, we could track them separately
                    # For now, we'll just note the position change
            
            # Only record trade P&L if it was a position-reducing trade (sell)
            if not trade.is_buy_aggressor() and trade_pnl != 0:
                trade_pnls.append(trade_pnl)
            
            cumulative_pnl += trade_pnl
            pnl_series.append(cumulative_pnl)
        
        self.total_pnl = cumulative_pnl
        
        # Calculate win rate based on profitable position-closing trades
        if trade_pnls:
            winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
            self.win_rate = winning_trades / len(trade_pnls)
        else:
            self.win_rate = 0.0
        
        # Calculate max drawdown
        if pnl_series:
            peak = pnl_series[0]
            max_dd = 0.0
            
            for pnl in pnl_series:
                if pnl > peak:
                    peak = pnl
                drawdown = peak - pnl
                if drawdown > max_dd:
                    max_dd = drawdown
            
            self.max_drawdown = max_dd
        
        # Calculate fill rate
        filled_orders = sum(1 for order in self.orders if order.status == OrderStatus.FILLED)
        self.fill_rate = filled_orders / len(self.orders) if self.orders else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'symbol': self.symbol,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.duration.total_seconds(),
            'total_pnl': self.total_pnl,
            'total_volume': self.total_volume,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'fill_rate': self.fill_rate,
            'avg_slippage': self.avg_slippage,
            'avg_latency_us': self.avg_latency_us,
            'market_returns': self.market_returns,
            'strategy_returns': self.strategy_returns,
            'alpha': self.alpha,
            'beta': self.beta,
            'metadata': self.metadata
        }


class ExecutionSimulator:
    """
    Main execution simulator for HFT strategies
    
    This class orchestrates the entire simulation process:
    - Loading and preprocessing market data
    - Replaying market events in chronological order
    - Executing strategy logic at each time step
    - Processing orders through the matching engine
    - Tracking performance and generating reports
    
    Key Features:
    - Real-time and historical data simulation
    - Multiple strategy support
    - Realistic execution modeling
    - Comprehensive performance tracking
    - Memory-efficient processing for large datasets
    
    Example Usage:
        >>> simulator = ExecutionSimulator("AAPL")
        >>> strategy = MyTradingStrategy()
        >>> result = simulator.run_backtest(
        ...     data_file="aapl_hft_data.csv",
        ...     strategy=strategy,
        ...     start_date="2023-01-01",
        ...     end_date="2023-01-02"
        ... )
        >>> print(f"Total P&L: ${result.total_pnl:.2f}")
    """
    
    def __init__(self, 
                 symbol: str,
                 fill_model: Optional[FillModel] = None,
                 tick_size: float = 0.01,
                 initial_cash: float = 100000.0):
        """
        Initialize the execution simulator
        
        Args:
            symbol: Trading symbol
            fill_model: Model for realistic order execution
            tick_size: Minimum price increment
            initial_cash: Starting cash for simulation
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.initial_cash = initial_cash
        
        self.logger = get_logger(f"{__name__}.{symbol}")
        
        # Core components
        self.matching_engine = MatchingEngine(
            symbol=symbol,
            fill_model=fill_model or RealisticFillModel(),
            tick_size=tick_size
        )
        
        self.data_ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor(tick_size=tick_size)
        
        # Simulation state
        self.current_time: Optional[pd.Timestamp] = None
        self.current_cash = initial_cash
        self.current_position = 0
        self.current_snapshot: Optional[BookSnapshot] = None
        
        # Strategy management
        self.strategies: List[Any] = []  # Will be properly typed when strategy classes are implemented
        self.strategy_callbacks: Dict[str, Callable] = {}
        
        # Event tracking
        self.events: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Progress tracking variables
        self.orders_generated_count = 0
        self.trades_executed_count = 0
        self.ticks_processed_count = 0
        self.last_progress_update_time = 0
        self.last_progress_update_ticks = 0
        self.orders_filled_count = 0  # Track filled orders separately
        self.orders_cancelled_count = 0  # Track cancelled orders
        self.total_volume_traded = 0  # Track total traded volume
        self.current_pnl = 0.0  # Track real-time P&L
        
        # Configuration
        self.config = {
            'max_position_size': 1000,
            'max_order_size': 100,
            'risk_limit': 10000.0,
            'enable_logging': True,
            'save_snapshots': True,
            'progress_update_frequency': 100,  # Update every N ticks
            'progress_time_frequency': 1.0,   # Update every N seconds
            'show_progress_bar': True,        # Show progress bar
            'verbose_progress': False,        # Verbose progress output
            'show_order_trade_counters': True,  # Show order/trade counters in progress
            'show_pnl_in_progress': True,    # Show real-time P&L in progress
            'compact_progress_mode': False,   # Use compact progress display
        }
        
        self.logger.info(f"ExecutionSimulator initialized for {symbol}")
    
    @log_performance
    @log_memory_usage
    def run_backtest(self, 
                     data_source: Union[str, Path, pd.DataFrame],
                     strategy: Any,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     **kwargs) -> BacktestResult:
        """
        Run a complete backtest simulation
        
        Args:
            data_source: Path to data file or DataFrame with market data
            strategy: Trading strategy instance
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            **kwargs: Additional configuration options
            
        Returns:
            BacktestResult with comprehensive results
            
        Educational Notes:
        - Backtesting simulates strategy performance on historical data
        - Results help evaluate strategy viability before live trading
        - Important to avoid lookahead bias and overfitting
        - Transaction costs and market impact should be included
        """
        self.logger.info(f"Starting backtest for {self.symbol}")
        
        # Load and prepare data
        if isinstance(data_source, pd.DataFrame):
            data = data_source.copy()
        else:
            data = self.data_ingestion.load_csv(data_source)
        
        # Filter by date range if specified
        if start_date or end_date:
            data = self._filter_by_date_range(data, start_date, end_date)
        
        # Check if data is empty using proper length check to avoid ambiguous Series comparison
        if len(data) == 0:
            raise ValueError("No data available for specified date range")
        
        # Preprocess data
        data = self.preprocessor.process_tick_data(data)
        
        # Initialize backtest
        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()
        result = BacktestResult(self.symbol, start_time, end_time)
        
        self.logger.info(f"Backtesting from {start_time} to {end_time} ({len(data):,} records)")
        
        # Reset simulation state
        self._reset_simulation_state()
        
        # Register strategy
        self.strategies = [strategy]
        
        # Initialize progress tracking for this backtest
        self._init_progress_tracking(len(data))
        
        # Process data chronologically
        with Timer() as timer:
            processed_count = 0
            
            for _, row in data.iterrows():
                # Update current time
                self.current_time = row['timestamp']
                
                # Process market data update (this is our "tick" processing)
                self._process_tick(row, strategy, result)
                
                processed_count += 1
                self.ticks_processed_count += 1
                
                # Update progress display
                self._update_progress_display(processed_count, len(data))
                
                # Update performance tracking
                if processed_count % 1000 == 0:  # Every 1000 records
                    self._update_performance_tracking(result)
                
                # Memory management for large datasets
                if processed_count % 10000 == 0:
                    self.logger.debug(f"Processed {processed_count:,} records")
        
        # Finalize results
        result.calculate_metrics()
        
        # Final progress update
        self._finalize_progress_display()
        
        # Log summary
        self.logger.info(
            f"Backtest completed in {timer.elapsed():.1f}ms: "
            f"P&L=${result.total_pnl:.2f}, "
            f"Trades={result.total_trades}, "
            f"Fill Rate={result.fill_rate:.1%}"
        )
        
        return result
    
    def run_live_simulation(self, 
                           data_stream: Iterator[Dict[str, Any]],
                           strategy: Any,
                           duration_seconds: Optional[int] = None) -> BacktestResult:
        """
        Run live simulation with streaming data
        
        Args:
            data_stream: Iterator providing real-time market data
            strategy: Trading strategy instance
            duration_seconds: Maximum simulation duration
            
        Returns:
            BacktestResult with simulation results
        """
        self.logger.info("Starting live simulation")
        
        start_time = pd.Timestamp.now()
        result = BacktestResult(self.symbol, start_time, start_time)
        
        # Reset simulation state
        self._reset_simulation_state()
        
        # Register strategy
        self.strategies = [strategy]
        
        simulation_start = time.time()
        
        try:
            for market_update in data_stream:
                # Check duration limit
                if duration_seconds and (time.time() - simulation_start) > duration_seconds:
                    break
                
                # Update current time
                self.current_time = pd.Timestamp.now()
                
                # Process market update
                self._process_market_update(market_update, result)
                
                # Execute strategy logic
                self._execute_strategy_logic(strategy, result)
                
                # Real-time performance tracking
                self._update_performance_tracking(result)
        
        except KeyboardInterrupt:
            self.logger.info("Live simulation interrupted by user")
        
        # Finalize results
        result.end_time = pd.Timestamp.now()
        result.duration = result.end_time - result.start_time
        result.calculate_metrics()
        
        return result
    
    def _process_market_update(self, update: Union[Dict[str, Any], pd.Series], 
                              result: BacktestResult) -> None:
        """Process a single market data update"""
        
        # Convert to dictionary if pandas Series
        if isinstance(update, pd.Series):
            update = update.to_dict()
        
        # Create BookSnapshot with valid bid/ask/mid prices from CSV data
        snapshot = self._create_book_snapshot_from_data(update, result)
        
        # Verify snapshot has non-None values before using it
        if snapshot and self._validate_snapshot(snapshot):
            self.current_snapshot = snapshot
            
            # CRITICAL FIX: Populate order book with resting limit orders from CSV bid/ask data
            # This creates liquidity that our market orders can match against
            self._seed_order_book_from_snapshot(snapshot)
            
            # Save snapshot if configured
            if self.config.get('save_snapshots', True):
                result.add_snapshot(self.current_snapshot)
        else:
            # Log warning if snapshot creation failed
            self.logger.debug(f"Failed to create valid snapshot from update: {update}")
    
    def _create_book_snapshot_from_data(self, update: Dict[str, Any], result: BacktestResult) -> Optional[BookSnapshot]:
        """Create a BookSnapshot from CSV data with fallback logic for missing bid/ask"""
        
        try:
            # Extract price information from update
            price = None
            bid_price = None
            ask_price = None
            mid_price = None
            volume = update.get('volume', 100)
            
            # Try to get bid and ask prices directly
            if 'bid' in update and update['bid'] is not None:
                bid_price = float(update['bid'])
            elif 'bid_price' in update and update['bid_price'] is not None:
                bid_price = float(update['bid_price'])
            
            if 'ask' in update and update['ask'] is not None:
                ask_price = float(update['ask'])
            elif 'ask_price' in update and update['ask_price'] is not None:
                ask_price = float(update['ask_price'])
            
            # Try to get mid price
            if 'mid_price' in update and update['mid_price'] is not None:
                mid_price = float(update['mid_price'])
            elif 'mid' in update and update['mid'] is not None:
                mid_price = float(update['mid'])
            
            # Try to get general price
            if 'price' in update and update['price'] is not None:
                price = float(update['price'])
            
            # Fallback logic if bid/ask columns are missing
            if bid_price is None or ask_price is None or pd.isna(bid_price) or pd.isna(ask_price):
                # Use price with small spread as fallback
                base_price = mid_price or price
                if base_price is not None and not pd.isna(base_price) and base_price > 0:
                    # Create realistic spread (0.01% to 0.05% of price)
                    spread = max(0.01, base_price * 0.0002)  # 2 basis points spread
                    
                    if bid_price is None or pd.isna(bid_price):
                        bid_price = base_price - spread / 2
                    if ask_price is None or pd.isna(ask_price):
                        ask_price = base_price + spread / 2
                else:
                    # Last resort: use default prices around $100
                    self.logger.warning("No price data found in update, using default prices")
                    bid_price = 99.99
                    ask_price = 100.01
            
            # Ensure we have valid bid and ask prices
            if bid_price is None or ask_price is None:
                return None
            
            # Ensure bid < ask (fix crossed markets)
            if bid_price >= ask_price:
                mid = (bid_price + ask_price) / 2
                spread = max(0.01, mid * 0.0001)  # Minimum 1 basis point
                bid_price = mid - spread / 2
                ask_price = mid + spread / 2
            
            # Convert volume to integer
            bid_volume = int(volume) if volume is not None else 100
            ask_volume = int(volume) if volume is not None else 100
            
            # Create BookSnapshot using the create_from_best_quotes method
            snapshot = BookSnapshot.create_from_best_quotes(
                symbol=self.symbol,
                timestamp=self.current_time or pd.Timestamp.now(),
                best_bid=bid_price,
                best_bid_volume=bid_volume,
                best_ask=ask_price,
                best_ask_volume=ask_volume,
                sequence_number=len(result.snapshots)
            )
            
            # Set additional trade information if available
            if price is not None:
                snapshot.last_trade_price = price
                snapshot.last_trade_volume = bid_volume
                snapshot.last_trade_timestamp = self.current_time
            
            return snapshot
            
        except Exception as e:
            self.logger.debug(f"Error creating BookSnapshot: {str(e)}")
            return None
    
    def _validate_snapshot(self, snapshot: BookSnapshot) -> bool:
        """Verify snapshot has valid non-None values before passing to strategy"""
        
        # Check that snapshot has basic required fields
        if snapshot is None:
            return False
        
        # Verify bid/ask prices are valid
        if snapshot.best_bid is None or snapshot.best_ask is None:
            self.logger.debug("Snapshot missing bid or ask price")
            return False
        
        # Check for NaN values
        if pd.isna(snapshot.best_bid) or pd.isna(snapshot.best_ask):
            self.logger.debug(f"Snapshot has NaN prices: bid={snapshot.best_bid}, ask={snapshot.best_ask}")
            return False
        
        # Verify bid/ask prices are positive
        if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
            self.logger.debug(f"Invalid prices: bid={snapshot.best_bid}, ask={snapshot.best_ask}")
            return False
        
        # Verify market is not crossed (bid should be < ask)
        if snapshot.best_bid >= snapshot.best_ask:
            self.logger.debug(f"Crossed market: bid={snapshot.best_bid}, ask={snapshot.best_ask}")
            return False
        
        # Verify volumes are positive
        if snapshot.best_bid_volume is None or snapshot.best_ask_volume is None:
            self.logger.debug("Snapshot missing bid or ask volume")
            return False
        
        if snapshot.best_bid_volume <= 0 or snapshot.best_ask_volume <= 0:
            self.logger.debug(f"Invalid volumes: bid_vol={snapshot.best_bid_volume}, ask_vol={snapshot.best_ask_volume}")
            return False
        
        # Verify mid price calculation works
        if snapshot.mid_price is None:
            self.logger.debug("Snapshot mid_price calculation failed")
            return False
        
        return True
    
    def _seed_order_book_from_snapshot(self, snapshot: BookSnapshot) -> None:
        """CRITICAL FIX: Seed order book with resting limit orders from CSV bid/ask data
        
        This method creates the necessary liquidity in the order book so that
        our strategy's market orders have something to match against.
        
        Args:
            snapshot: BookSnapshot containing bid/ask levels from CSV data
        """
        try:
            # Generate unique order IDs for market data orders  
            timestamp = self.current_time or pd.Timestamp.now()
            sequence = len(self.matching_engine.get_trade_history())
            
            # Create resting bid order (buy-side liquidity)
            if snapshot.best_bid and snapshot.best_bid_volume:
                bid_order_id = f"market_bid_{sequence}_{timestamp.strftime('%H%M%S%f')[:-3]}"
                bid_order = Order(
                    order_id=bid_order_id,
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=snapshot.best_bid,
                    volume=snapshot.best_bid_volume,
                    timestamp=timestamp,
                    source='market_data_seeding',
                    metadata={'type': 'market_liquidity_provider'}
                )
                
                # Add directly to order book (bypass matching engine to avoid recursive processing)
                self.matching_engine.get_order_book()._add_order_to_book(bid_order)
                
                self.logger.debug(f"[ORDER_BOOK_SEEDING] Added BID liquidity: Price={snapshot.best_bid:.4f}, "
                                f"Volume={snapshot.best_bid_volume}, OrderID={bid_order_id}")
            
            # Create resting ask order (sell-side liquidity)
            if snapshot.best_ask and snapshot.best_ask_volume:
                ask_order_id = f"market_ask_{sequence}_{timestamp.strftime('%H%M%S%f')[:-3]}"
                ask_order = Order(
                    order_id=ask_order_id,
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=snapshot.best_ask,
                    volume=snapshot.best_ask_volume,
                    timestamp=timestamp,
                    source='market_data_seeding',
                    metadata={'type': 'market_liquidity_provider'}
                )
                
                # Add directly to order book (bypass matching engine to avoid recursive processing)
                self.matching_engine.get_order_book()._add_order_to_book(ask_order)
                
                self.logger.debug(f"[ORDER_BOOK_SEEDING] Added ASK liquidity: Price={snapshot.best_ask:.4f}, "
                                f"Volume={snapshot.best_ask_volume}, OrderID={ask_order_id}")
            
            # Log order book state after seeding
            order_book = self.matching_engine.get_order_book()
            bid_levels = len(order_book.bids)
            ask_levels = len(order_book.asks)
            self.logger.debug(f"[ORDER_BOOK_SEEDED] Order book now has {bid_levels} bid levels and {ask_levels} ask levels")
            
        except Exception as e:
            self.logger.error(f"[ORDER_BOOK_SEEDING_ERROR] Failed to seed order book from snapshot: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _execute_strategy_logic(self, strategy: Any, result: BacktestResult) -> None:
        """Execute strategy logic for current market state"""
        
        if not self.current_snapshot:
            return
        
        try:
            # Call strategy's decision method
            if hasattr(strategy, 'on_market_update'):
                strategy_result = strategy.on_market_update(self.current_snapshot, self.current_time)
                
                # Handle different return types for backward compatibility
                if strategy_result is None:
                    return
                    
                # If it's a StrategyResult object, extract orders
                if hasattr(strategy_result, 'orders'):
                    orders = strategy_result.orders
                    # Log strategy decision reasoning
                    if hasattr(strategy_result, 'decision_reason') and strategy_result.decision_reason:
                        self.logger.debug(f"Strategy decision: {strategy_result.decision_reason}")
                elif isinstance(strategy_result, list):
                    # Backward compatibility: simple list of orders
                    orders = strategy_result
                else:
                    self.logger.warning(f"Unexpected strategy result type: {type(strategy_result)}")
                    return
                
                # Process any orders generated by strategy
                if orders:
                    for order in orders:
                        self._process_strategy_order(order, result)
                        
                # Handle strategy cancellations and modifications if present
                if hasattr(strategy_result, 'cancellations') and strategy_result.cancellations:
                    for order_id in strategy_result.cancellations:
                        self.logger.debug(f"Strategy requested cancellation: {order_id}")
                        # TODO: Implement order cancellation
                        
                if hasattr(strategy_result, 'modifications') and strategy_result.modifications:
                    for order_id, new_price, new_volume in strategy_result.modifications:
                        self.logger.debug(f"Strategy requested modification: {order_id}")
                        # TODO: Implement order modification
        
        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _process_strategy_order(self, order: Order, result: BacktestResult) -> None:
        """Process an order generated by a strategy"""
        
        # Log when orders are submitted for processing
        price_str = f"{order.price:.4f}" if order.price is not None else "MARKET"
        self.logger.debug(f"[SIMULATOR_ORDER_ENTRY] Strategy order entering ExecutionSimulator: "
                        f"ID={order.order_id}, Side={order.side.value}, Volume={order.volume}, "
                        f"Price={price_str}, Type={order.order_type.value}")
        
        # Validate order against risk limits
        if not self._validate_strategy_order(order):
            self.logger.debug(f"[SIMULATOR_ORDER_REJECTED] Order {order.order_id} failed validation")
            return
        
        # Log matching engine delegation
        self.logger.debug(f"[SIMULATOR_DELEGATION] Delegating order {order.order_id} to MatchingEngine")
        
        # Store the current order ID for trade processing
        self._current_processing_order_id = order.order_id
        
        # Process through matching engine
        trades, order_update = self.matching_engine.process_order(order)
        
        # Log matching engine response with trade propagation details
        self.logger.debug(f"[SIMULATOR_ENGINE_RESPONSE] MatchingEngine returned: "
                         f"Trades={len(trades) if trades else 0}, "
                         f"OrderUpdate={order_update.update_type if order_update else 'None'}")
        
        # Check if trades are being returned
        if trades is None:
            self.logger.warning(f"[SIMULATOR_TRADES_NULL] MatchingEngine returned None trades for order {order.order_id}")
            trades = []  # Ensure we have an empty list instead of None
        elif not isinstance(trades, list):
            self.logger.warning(f"[SIMULATOR_TRADES_INVALID] MatchingEngine returned trades of type {type(trades)} instead of list")
            trades = []  # Convert to empty list for safety
        
        # Log trade propagation analysis
        if trades:
            total_trade_volume = sum(t.volume for t in trades)
            total_trade_value = sum(t.price * t.volume for t in trades)
            avg_trade_price = total_trade_value / total_trade_volume if total_trade_volume > 0 else 0
            
            self.logger.debug(f"[SIMULATOR_TRADE_ANALYSIS] Order {order.order_id} generated {len(trades)} trades: "
                             f"TotalVolume={total_trade_volume}, AvgPrice={avg_trade_price:.4f}, "
                             f"TotalValue=${total_trade_value:.2f}")
            
            # Log individual trade propagation
            for i, trade in enumerate(trades):
                self.logger.debug(f"[SIMULATOR_TRADE_PROPAGATION] Trade {i+1}/{len(trades)}: "
                                f"ID={trade.trade_id}, Price={trade.price:.4f}, Volume={trade.volume}, "
                                f"BuyOrder={trade.buy_order_id}, SellOrder={trade.sell_order_id}, "
                                f"Aggressor={trade.aggressor_side.value}")
        else:
            self.logger.debug(f"[SIMULATOR_NO_TRADES] Order {order.order_id} did not generate any trades")
        
        # Process and propagate trades to simulation state
        trades_propagated = 0
        pre_position = self.current_position
        pre_cash = self.current_cash
        
        for trade in trades:
            try:
                self._update_position_and_cash(trade)
                result.add_trade(trade)
                trades_propagated += 1
                
                self.logger.debug(f"[SIMULATOR_TRADE_PROPAGATED] Trade {trade.trade_id} propagated to simulation state")
            except Exception as e:
                self.logger.error(f"[SIMULATOR_PROPAGATION_ERROR] Failed to propagate trade {trade.trade_id}: {str(e)}")
        
        # Log position and cash changes
        if trades_propagated > 0:
            position_change = self.current_position - pre_position
            cash_change = self.current_cash - pre_cash
            self.logger.debug(f"[SIMULATOR_STATE_UPDATE] Position: {pre_position} → {self.current_position} ({position_change:+d}), "
                             f"Cash: ${pre_cash:.2f} → ${self.current_cash:.2f} (${cash_change:+.2f})")
        
        # Add order to results for tracking
        result.add_order(order)
        
        # Log final propagation summary
        self.logger.debug(f"[SIMULATOR_PROPAGATION_COMPLETE] Order {order.order_id}: "
                         f"TradesGenerated={len(trades)}, TradesPropagated={trades_propagated}, "
                         f"BacktestTotalTrades={len(result.trades)}")
        
        # Clear processing state
        self._current_processing_order_id = None
        
        # Log execution outcome
        if trades:
            total_volume = sum(t.volume for t in trades)
            self.logger.info(f"[SIMULATOR_EXECUTION_SUCCESS] Strategy order {order.order_id} executed: "
                            f"{len(trades)} trades, Volume={total_volume}, Status={order.status}")
        else:
            self.logger.debug(f"[SIMULATOR_EXECUTION_COMPLETE] Strategy order {order.order_id} processed without trades")
    
    def _validate_strategy_order(self, order: Order) -> bool:
        """Validate strategy order against risk limits"""
        
        # Check position limits
        if abs(self.current_position) >= self.config['max_position_size']:
            self.logger.warning(f"Order rejected: position limit exceeded ({self.current_position})")
            return False
        
        # Check order size limits
        if order.volume > self.config['max_order_size']:
            self.logger.warning(f"Order rejected: order size too large ({order.volume})")
            return False
        
        # Check cash requirements (simplified)
        if order.is_buy() and self.current_snapshot and self.current_snapshot.best_ask:
            required_cash = order.volume * self.current_snapshot.best_ask
            if required_cash > self.current_cash:
                self.logger.warning(f"Order rejected: insufficient cash ({required_cash} > {self.current_cash})")
                return False
        
        return True
    
    def _update_position_and_cash(self, trade: Trade) -> None:
        """Update current position and cash based on trade
        
        Note: The trade direction is determined by checking which order ID
        belongs to our strategy. If our order was the buyer, we bought shares.
        If our order was the seller, we sold shares.
        """
        # We need to determine if we were the buyer or seller in this trade
        # by checking which order ID matches the one we just processed
        
        # Get the most recent order from our orders (the one that generated this trade)
        if hasattr(self, '_current_processing_order_id'):
            our_order_id = self._current_processing_order_id
        else:
            # Fallback: try to infer from trade IDs
            our_order_id = None
        
        # Determine if we were the buyer or seller
        if trade.buy_order_id == our_order_id:
            # We were the buyer - we bought shares
            self.current_position += trade.volume
            self.current_cash -= trade.volume * trade.price
            self.logger.debug(f"[POSITION_UPDATE] BUY: +{trade.volume} shares, -{trade.volume * trade.price:.2f} cash")
        elif trade.sell_order_id == our_order_id:
            # We were the seller - we sold shares
            self.current_position -= trade.volume
            self.current_cash += trade.volume * trade.price
            self.logger.debug(f"[POSITION_UPDATE] SELL: -{trade.volume} shares, +{trade.volume * trade.price:.2f} cash")
        else:
            # Fallback: use aggressor side to determine direction
            # This assumes that our order was the aggressor
            if trade.is_buy_aggressor():
                self.current_position += trade.volume
                self.current_cash -= trade.volume * trade.price
                self.logger.debug(f"[POSITION_UPDATE] BUY (aggressor): +{trade.volume} shares, -{trade.volume * trade.price:.2f} cash")
            else:
                self.current_position -= trade.volume
                self.current_cash += trade.volume * trade.price
                self.logger.debug(f"[POSITION_UPDATE] SELL (aggressor): -{trade.volume} shares, +{trade.volume * trade.price:.2f} cash")
    
    def _update_performance_tracking(self, result: BacktestResult) -> None:
        """Update performance tracking metrics"""
        
        if not self.current_snapshot or not self.current_snapshot.mid_price:
            return
        
        # Calculate current portfolio value
        position_value = self.current_position * self.current_snapshot.mid_price
        total_value = self.current_cash + position_value
        
        # Track performance
        performance_record = {
            'timestamp': self.current_time,
            'cash': self.current_cash,
            'position': self.current_position,
            'position_value': position_value,
            'total_value': total_value,
            'unrealized_pnl': total_value - self.initial_cash,
            'mid_price': self.current_snapshot.mid_price,
            'spread': self.current_snapshot.spread,
        }
        
        self.performance_history.append(performance_record)
    
    def _filter_by_date_range(self, data: pd.DataFrame, 
                             start_date: Optional[str], 
                             end_date: Optional[str]) -> pd.DataFrame:
        """Filter data by date range"""
        
        if 'timestamp' not in data.columns:
            return data
        
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        if start_date:
            start_ts = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_ts]
        
        if end_date:
            end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # Include full end date
            data = data[data['timestamp'] < end_ts]
        
        return data
    
    def _process_tick(self, row: pd.Series, strategy: Any, result: BacktestResult) -> None:
        """Process a single tick of market data with progress tracking
        
        This method combines market data processing and strategy execution into
        a single tick processing unit, making it easier to track progress.
        
        Args:
            row: Market data row
            strategy: Trading strategy instance
            result: BacktestResult to update
        """
        # Process market data update
        self._process_market_update(row, result)
        
        # Execute strategy logic and track orders/trades
        orders_before = len(result.orders)
        trades_before = len(result.trades)
        
        self._execute_strategy_logic(strategy, result)
        
        # Update counters based on what was generated
        orders_generated = len(result.orders) - orders_before
        trades_executed = len(result.trades) - trades_before
        
        self.orders_generated_count += orders_generated
        self.trades_executed_count += trades_executed
        
        # Update volume tracking if we have new trades
        if trades_executed > 0:
            for trade in result.trades[-trades_executed:]:
                self.total_volume_traded += trade.volume
        
        # Update real-time P&L estimate
        if self.current_snapshot and self.current_snapshot.mid_price:
            position_value = self.current_position * self.current_snapshot.mid_price
            total_value = self.current_cash + position_value
            self.current_pnl = total_value - self.initial_cash
    
    def _init_progress_tracking(self, total_ticks: int) -> None:
        """Initialize progress tracking for the backtest
        
        Args:
            total_ticks: Total number of ticks to process
        """
        self.total_ticks = total_ticks
        self.orders_generated_count = 0
        self.trades_executed_count = 0
        self.ticks_processed_count = 0
        self.last_progress_update_time = time.time()
        self.last_progress_update_ticks = 0
        self.start_time = time.time()
        
        # Print initial progress header
        if self.config.get('show_progress_bar', True):
            print(f"\nStarting backtest with {total_ticks:,} ticks...")
            print("-" * 80)
    
    def _update_progress_display(self, processed_count: int, total_count: int) -> None:
        """Update progress display based on configuration
        
        Args:
            processed_count: Number of ticks processed
            total_count: Total number of ticks
        """
        current_time = time.time()
        
        # Check if we should update based on tick frequency
        tick_update_needed = (processed_count - self.last_progress_update_ticks >= 
                             self.config.get('progress_update_frequency', 100))
        
        # Check if we should update based on time frequency
        time_update_needed = (current_time - self.last_progress_update_time >= 
                             self.config.get('progress_time_frequency', 1.0))
        
        # Update if either condition is met, or if this is the last tick
        if tick_update_needed or time_update_needed or processed_count == total_count:
            self._display_progress(processed_count, total_count, current_time)
            self.last_progress_update_time = current_time
            self.last_progress_update_ticks = processed_count
    
    def _display_progress(self, processed_count: int, total_count: int, current_time: float) -> None:
        """Display progress information
        
        Args:
            processed_count: Number of ticks processed
            total_count: Total number of ticks
            current_time: Current timestamp
        """
        if not self.config.get('show_progress_bar', True):
            return
        
        # Calculate progress metrics
        progress_pct = (processed_count / total_count) * 100 if total_count > 0 else 0
        elapsed_time = current_time - self.start_time
        
        # Calculate processing rate
        ticks_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
        
        # Estimate remaining time
        if ticks_per_second > 0 and processed_count < total_count:
            remaining_ticks = total_count - processed_count
            eta_seconds = remaining_ticks / ticks_per_second
            eta_str = f" ETA: {eta_seconds:.0f}s"
        else:
            eta_str = ""
        
        # Create progress bar
        bar_width = 40
        filled_width = int(bar_width * progress_pct / 100)
        bar = '█' * filled_width + '░' * (bar_width - filled_width)
        
        # Format the progress message with enhanced metrics
        if self.config.get('compact_progress_mode', False):
            # Ultra-compact mode for minimal output
            progress_msg = (
                f"\r[{bar}] {progress_pct:5.1f}% | "
                f"O:{self.orders_generated_count} T:{self.trades_executed_count}{eta_str}"
            )
        elif self.config.get('verbose_progress', False):
            # Verbose mode - show detailed information including P&L and volume
            position_str = f"Pos:{self.current_position}" if self.current_position != 0 else "Flat"
            pnl_str = f"P&L:${self.current_pnl:+.0f}" if self.config.get('show_pnl_in_progress', True) else ""
            volume_str = f"Vol:{self.total_volume_traded:,}" if self.total_volume_traded > 0 else ""
            
            progress_msg = (
                f"\r[{bar}] {progress_pct:5.1f}% | "
                f"Ticks: {processed_count:,}/{total_count:,} | "
                f"Orders: {self.orders_generated_count} | "
                f"Trades: {self.trades_executed_count} | "
                f"{position_str} | {pnl_str} {volume_str} | "
                f"Rate: {ticks_per_second:.0f}/s{eta_str}"
            ).replace('  ', ' ')  # Clean up double spaces
        else:
            # Standard mode - balanced information display
            extra_info = ""
            if self.config.get('show_pnl_in_progress', True) and abs(self.current_pnl) > 1:
                extra_info += f" | P&L: ${self.current_pnl:+.0f}"
            if self.total_volume_traded > 0:
                extra_info += f" | Vol: {self.total_volume_traded:,}"
                
            progress_msg = (
                f"\r[{bar}] {progress_pct:5.1f}% | "
                f"Orders: {self.orders_generated_count}, "
                f"Trades: {self.trades_executed_count}{extra_info}{eta_str}"
            )
        
        # Print progress (overwrite previous line)
        sys.stdout.write(progress_msg)
        sys.stdout.flush()
    
    def _finalize_progress_display(self) -> None:
        """Finalize and clean up progress display"""
        if not self.config.get('show_progress_bar', True):
            return
        
        # Move to next line and print summary
        print()
        print("-" * 80)
        
        elapsed_time = time.time() - self.start_time
        avg_rate = self.ticks_processed_count / elapsed_time if elapsed_time > 0 else 0
        
        summary_msg = (
            f"Backtest completed: {self.ticks_processed_count:,} ticks processed | "
            f"Orders Generated: {self.orders_generated_count} | "
            f"Trades Executed: {self.trades_executed_count} | "
            f"Avg Rate: {avg_rate:.0f} ticks/s | "
            f"Total Time: {elapsed_time:.1f}s"
        )
        
        print(summary_msg)
        print("-" * 80)
        print()
    
    def _reset_simulation_state(self) -> None:
        """Reset simulation state for new run"""
        self.current_time = None
        self.current_cash = self.initial_cash
        self.current_position = 0
        self.current_snapshot = None
        self.events.clear()
        self.performance_history.clear()
        self.matching_engine.reset_session()
        
        # Reset progress tracking counters
        self.orders_generated_count = 0
        self.trades_executed_count = 0
        self.ticks_processed_count = 0
        self.last_progress_update_time = 0
        self.last_progress_update_ticks = 0
        
        # Clear any order processing state
        self._current_processing_order_id = None
    
    def get_performance_dataframe(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        if not self.performance_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            'current_time': self.current_time,
            'current_cash': self.current_cash,
            'current_position': self.current_position,
            'current_snapshot': self.current_snapshot.to_dict() if self.current_snapshot else None,
            'matching_engine_stats': self.matching_engine.get_statistics(),
        }
    
    def set_config(self, **kwargs) -> None:
        """Update simulation configuration"""
        self.config.update(kwargs)
        self.logger.info(f"Configuration updated: {kwargs}")
    
    def add_strategy_callback(self, event_type: str, callback: Callable) -> None:
        """Add callback for strategy events"""
        self.strategy_callbacks[event_type] = callback
    
    def load_csv_data(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load CSV data with proper error handling and validation
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for DataIngestion
            
        Returns:
            Loaded DataFrame with market data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data is invalid or empty
        """
        try:
            self.logger.info(f"Loading CSV data from {filepath}")
            
            # Ensure file exists
            file_path = Path(filepath)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            # Load data using data ingestion module
            data = self.data_ingestion.load_csv(file_path, **kwargs)
            
            # Basic validation
            if data.empty:
                raise ValueError(f"CSV file is empty: {filepath}")
            
            # Check for required columns (flexible approach)
            required_cols = ['timestamp', 'price', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if len(missing_cols) == len(required_cols):
                # If all required columns are missing, try to infer them
                self.logger.warning(f"Missing standard columns: {missing_cols}")
                data = self._infer_column_mappings(data)
            
            self.logger.info(f"Loaded {len(data):,} records with columns: {list(data.columns)}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV data from {filepath}: {str(e)}")
            raise
    
    def _infer_column_mappings(self, data: pd.DataFrame) -> pd.DataFrame:
        """Try to infer column mappings from CSV data
        
        Args:
            data: Raw DataFrame from CSV
            
        Returns:
            DataFrame with inferred column names
        """
        data = data.copy()
        
        # Common column name variations
        column_mappings = {
            'timestamp': ['time', 'datetime', 'ts', 'date_time', 'date'],
            'price': ['px', 'price_level', 'mid_price', 'close', 'last'],
            'volume': ['vol', 'size', 'qty', 'quantity', 'amount'],
            'side': ['direction', 'type', 'order_side', 'buy_sell']
        }
        
        # Try to map columns
        for standard_name, variations in column_mappings.items():
            if standard_name in data.columns:
                continue  # Already exists
            
            for variation in variations:
                # Try exact match and case-insensitive match
                for col in data.columns:
                    if col.lower() == variation.lower():
                        data = data.rename(columns={col: standard_name})
                        self.logger.info(f"Mapped column '{col}' to '{standard_name}'")
                        break
                if standard_name in data.columns:
                    break
        
        # If still missing timestamp, try to create a synthetic one
        if 'timestamp' not in data.columns:
            self.logger.warning("Creating synthetic timestamp column")
            data['timestamp'] = pd.date_range('2023-01-01 09:30:00', periods=len(data), freq='1ms')
        
        # If still missing price, create synthetic prices around $100
        if 'price' not in data.columns:
            self.logger.warning("Creating synthetic price column")
            np.random.seed(42)  # For reproducible synthetic data
            data['price'] = 100 + np.random.normal(0, 1, len(data)).cumsum() * 0.01
        
        # If still missing volume, create synthetic volumes
        if 'volume' not in data.columns:
            self.logger.warning("Creating synthetic volume column")
            np.random.seed(42)
            data['volume'] = np.random.randint(1, 100, len(data))
        
        return data
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data integrity and return report
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for required columns
        required_columns = ['timestamp', 'price', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_report['is_valid'] = False
            validation_report['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types and ranges
        if 'price' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['price']):
                validation_report['is_valid'] = False
                validation_report['issues'].append("Price column is not numeric")
            elif (data['price'] <= 0).any():
                validation_report['warnings'].append("Price column contains non-positive values")
        
        if 'volume' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['volume']):
                validation_report['is_valid'] = False
                validation_report['issues'].append("Volume column is not numeric")
            elif (data['volume'] <= 0).any():
                validation_report['warnings'].append("Volume column contains non-positive values")
        
        # Check timestamp ordering
        if 'timestamp' in data.columns:
            try:
                timestamps = pd.to_datetime(data['timestamp'])
                if not timestamps.is_monotonic_increasing:
                    validation_report['warnings'].append("Timestamps are not in chronological order")
            except Exception as e:
                validation_report['is_valid'] = False
                validation_report['issues'].append(f"Cannot parse timestamps: {str(e)}")
        
        # Generate statistics
        validation_report['stats'] = {
            'total_rows': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.astype(str).to_dict()
        }
        
        if 'timestamp' in data.columns:
            try:
                timestamps = pd.to_datetime(data['timestamp'])
                validation_report['stats']['time_range'] = {
                    'start': timestamps.min(),
                    'end': timestamps.max(),
                    'duration': timestamps.max() - timestamps.min()
                }
            except:
                pass
        
        return validation_report
    
    def __str__(self) -> str:
        """String representation of the simulator"""
        return (f"ExecutionSimulator({self.symbol}): "
                f"Cash=${self.current_cash:.2f}, "
                f"Position={self.current_position}, "
                f"Strategies={len(self.strategies)}")
    
    def __repr__(self) -> str:
        return self.__str__()


# Utility functions for simulation analysis
def compare_backtest_results(results: List[BacktestResult]) -> pd.DataFrame:
    """
    Compare multiple backtest results
    
    Args:
        results: List of BacktestResult objects
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for i, result in enumerate(results):
        comparison_data.append({
            'run_id': i,
            'symbol': result.symbol,
            'total_pnl': result.total_pnl,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown,
            'fill_rate': result.fill_rate,
            'sharpe_ratio': result.sharpe_ratio,
            'duration_hours': result.duration.total_seconds() / 3600,
        })
    
    return pd.DataFrame(comparison_data)


def analyze_execution_quality(result: BacktestResult) -> Dict[str, Any]:
    """
    Analyze execution quality from backtest results
    
    Args:
        result: BacktestResult to analyze
        
    Returns:
        Dictionary with execution quality metrics
    """
    if not result.trades:
        return {'error': 'No trades to analyze'}
    
    # Calculate execution metrics
    trade_sizes = [trade.volume for trade in result.trades]
    trade_prices = [trade.price for trade in result.trades]
    
    analysis = {
        'execution_summary': {
            'total_trades': len(result.trades),
            'total_volume': sum(trade_sizes),
            'avg_trade_size': np.mean(trade_sizes),
            'median_trade_size': np.median(trade_sizes),
            'price_range': {
                'min': min(trade_prices),
                'max': max(trade_prices),
                'avg': np.mean(trade_prices)
            }
        },
        'timing_analysis': {
            'first_trade': result.trades[0].timestamp,
            'last_trade': result.trades[-1].timestamp,
            'trading_duration': result.trades[-1].timestamp - result.trades[0].timestamp,
        },
        'performance_metrics': {
            'total_pnl': result.total_pnl,
            'win_rate': result.win_rate,
            'fill_rate': result.fill_rate,
            'max_drawdown': result.max_drawdown,
        }
    }
    
    return analysis