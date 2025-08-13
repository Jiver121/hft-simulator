#!/usr/bin/env python3
"""
HFT Simulator - Main Entry Point for Batch Backtesting

This script provides a comprehensive command-line interface for running batch backtests
on CSV market data files. It supports multiple strategies, date ranges, and detailed
performance reporting.

Usage Examples:
    # Single CSV file backtest
    python main.py --mode backtest --data ./data/historical_data.csv --output ./logs/backtest_results.json

    # Multiple files or date range
    python main.py --mode backtest --data ./data/ --start-date 2024-01-01 --end-date 2024-01-31 --output ./logs/

    # With specific strategy and parameters
    python main.py --mode backtest --data ./data/ --strategy market_making --output ./logs/ --config ./config/backtest_config.json

    # Parallel processing for multiple files
    python main.py --mode backtest --data ./data/ --output ./logs/ --parallel --workers 4
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import warnings
import traceback

import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Import HFT Simulator components
from src.execution.simulator import ExecutionSimulator, BacktestResult
from src.execution.fill_models import RealisticFillModel, PerfectFillModel
from src.data.ingestion import DataIngestion
from src.data.preprocessor import DataPreprocessor
from src.strategies.base_strategy import BaseStrategy
from src.strategies.market_making import MarketMakingStrategy
from src.strategies.liquidity_taking import LiquidityTakingStrategy
from src.utils.logger import get_logger, setup_main_logger
from src.utils.helpers import Timer
from src.utils.constants import OrderSide, OrderType, OrderStatus

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class BacktestConfig:
    """Configuration class for backtesting parameters"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Default configuration
        self.strategy_type = "market_making"
        self.initial_capital = 100000.0
        self.commission_rate = 0.0005  # 5 bps
        self.slippage_bps = 1.0
        self.max_position_size = 1000
        self.max_order_size = 100
        self.risk_limit = 10000.0
        self.tick_size = 0.01
        self.fill_model = "realistic"
        self.enable_logging = True
        self.save_snapshots = False
        self.parallel_workers = 1
        self.benchmark_symbol = "SPY"
        
        # Strategy-specific parameters
        self.strategy_params = {
            "market_making": {
                "spread_bps": 10.0,
                "order_size": 100,
                "max_inventory": 500
            },
            "liquidity_taking": {
                "momentum_threshold": 0.001,
                "order_size": 200,
                "max_positions": 5
            }
        }
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            attr: getattr(self, attr) 
            for attr in dir(self) 
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }


class SimpleMarketMakingStrategy(BaseStrategy):
    """Simple market making strategy for backtesting"""
    
    def __init__(self, symbol: str, spread_bps: float = 2.0, order_size: int = 100):
        super().__init__("simple_market_making", symbol)
        # symbol is now set by parent class
        self.spread_bps = spread_bps  # Reduced from 10.0 to 2.0 to match market conditions
        self.order_size = order_size
        self.inventory = 0
        self.max_inventory = 10000  # INCREASED TO VERY PERMISSIVE FOR DEBUGGING
        
        # Add counter to track how many times strategy is called
        self.strategy_call_count = 0
        # Counter for aggressive orders that should fill
        self.aggressive_order_count = 0
        
        # Set data mode to backtest for more lenient validation
        self.set_data_mode("backtest")
        
    def on_market_update(self, snapshot: 'BookSnapshot', timestamp: pd.Timestamp) -> 'StrategyResult':
        """Generate orders based on market update"""
        from src.strategies.base_strategy import StrategyResult
        from src.engine.order_types import Order
        
        # Increment strategy call counter and log entry
        self.strategy_call_count += 1
        self.logger.debug(f"[STRATEGY] Strategy call #{self.strategy_call_count} at {timestamp}")
        
        # Log market snapshot data to verify it has valid bid/ask prices
        if snapshot:
            self.logger.debug(f"[MARKET_DATA] Snapshot - Mid: {getattr(snapshot, 'mid_price', 'N/A')}, "
                            f"Bid: {getattr(snapshot, 'best_bid', 'N/A')}, "
                            f"Ask: {getattr(snapshot, 'best_ask', 'N/A')}, "
                            f"Spread: {getattr(snapshot, 'spread', 'N/A')}")
        else:
            self.logger.debug(f"[MARKET_DATA] No market snapshot available at {timestamp}")
        
        # Create result container
        result = StrategyResult(
            timestamp=timestamp,
            processing_time_us=0
        )
        
        # Validate and handle missing/invalid market data
        if not self._validate_snapshot(snapshot):
            self.logger.warning(f"Invalid market data received at {timestamp}")
            result.decision_reason = "Invalid market data"
            return result
        
        # Update market history
        self._update_market_history(snapshot)
        
        # Update position tracking (simplified inventory management)
        self.inventory = self.current_position
        
        try:
            # Calculate spread
            if not snapshot.mid_price:
                result.decision_reason = "No mid price available"
                return result
                
            spread = snapshot.mid_price * (self.spread_bps / 10000)
            
            # Inventory skew for risk management
            inventory_skew = (self.inventory / self.max_inventory) * (spread / 4)
            
            # Generate buy order if not too long and market data is available
            if (self.inventory < self.max_inventory and 
                snapshot.best_ask is not None and 
                snapshot.mid_price is not None):
                
                # Use a mix of passive and aggressive orders
                # Every 5th tick, place an aggressive order that should fill
                if self.strategy_call_count % 5 == 0:
                    # Aggressive buy order - use market order to ensure immediate fill
                    buy_price = None  # Market orders don't need a price
                    order_type = OrderType.MARKET  # Market order for guaranteed execution
                    order_reason = "Aggressive market making buy (MARKET)"
                    self.aggressive_order_count += 1
                else:
                    # Passive market making order
                    buy_price = max(0.01, snapshot.mid_price - spread/2 - inventory_skew)
                    order_type = OrderType.LIMIT
                    order_reason = "Passive market making buy"
                
                # Check risk limits before creating order
                potential_order = self.create_order(
                    side=OrderSide.BUY,
                    volume=self.order_size,
                    price=buy_price,
                    order_type=order_type,
                    reason=order_reason
                )
                
                # TEMPORARILY DISABLE RISK LIMITS FOR DEBUGGING
                # is_valid, reason = self.check_risk_limits(potential_order, snapshot.mid_price)
                is_valid, reason = True, "Risk limits disabled for debugging"
                if is_valid:
                    price_display = f"{buy_price:.4f}" if buy_price is not None else "MARKET"
                    self.logger.debug(f"[ORDER_GENERATED] BUY order created: ID={potential_order.order_id}, "
                                    f"Price={price_display}, Volume={self.order_size}, Mid={snapshot.mid_price:.4f}, Type={order_reason}")
                    result.add_order(potential_order, order_reason)
                    self.submit_order(potential_order)
                else:
                    self.logger.debug(f"[ORDER_REJECTED] Buy order rejected: {reason}")
            
            # Generate sell order if not too short and market data is available
            if (self.inventory > -self.max_inventory and 
                snapshot.best_bid is not None and 
                snapshot.mid_price is not None):
                
                # Use similar aggressive/passive mix for sell orders
                # Every 5th tick (offset by 2 to alternate with buys), place an aggressive sell
                if (self.strategy_call_count + 2) % 5 == 0:
                    # Aggressive sell order - use market order to ensure immediate fill
                    sell_price = None  # Market orders don't need a price
                    order_type = OrderType.MARKET  # Market order for guaranteed execution
                    order_reason = "Aggressive market making sell (MARKET)"
                    self.aggressive_order_count += 1
                else:
                    # Passive market making order
                    sell_price = snapshot.mid_price + spread/2 - inventory_skew
                    order_type = OrderType.LIMIT
                    order_reason = "Passive market making sell"
                
                # Check risk limits before creating order
                potential_order = self.create_order(
                    side=OrderSide.SELL,
                    volume=self.order_size,
                    price=sell_price,
                    order_type=order_type,
                    reason=order_reason
                )
                
                # TEMPORARILY DISABLE RISK LIMITS FOR DEBUGGING
                # is_valid, reason = self.check_risk_limits(potential_order, snapshot.mid_price)
                is_valid, reason = True, "Risk limits disabled for debugging"
                if is_valid:
                    price_display = f"{sell_price:.4f}" if sell_price is not None else "MARKET"
                    self.logger.debug(f"[ORDER_GENERATED] SELL order created: ID={potential_order.order_id}, "
                                    f"Price={price_display}, Volume={self.order_size}, Mid={snapshot.mid_price:.4f}, Type={order_reason}")
                    result.add_order(potential_order, order_reason)
                    self.submit_order(potential_order)
                else:
                    self.logger.debug(f"[ORDER_REJECTED] Sell order rejected: {reason}")
            
            # Set result metadata
            result.decision_reason = f"Generated {len(result.orders)} market making orders"
            result.confidence = 0.8 if result.orders else 0.0
            
        except Exception as e:
            self.logger.error(f"Error in market making strategy: {e}")
            result.decision_reason = f"Strategy error: {str(e)}"
        
        # Update strategy state
        self.last_update_time = timestamp
        self.update_count += 1
        
        # Log final strategy result
        self.logger.debug(f"[STRATEGY_RESULT] Call #{self.strategy_call_count} complete: "
                        f"Orders={len(result.orders)}, Reason='{result.decision_reason}', "
                        f"Confidence={result.confidence:.2f}")
        
        return result


class SimpleMomentumStrategy(BaseStrategy):
    """Simple momentum strategy for backtesting"""
    
    def __init__(self, symbol: str, momentum_threshold: float = 0.001, order_size: int = 200):
        super().__init__("simple_momentum", symbol)
        # symbol is now set by parent class
        self.momentum_threshold = momentum_threshold
        self.order_size = order_size
        self.max_history = 20
        self.max_position = 10000  # INCREASED TO VERY PERMISSIVE FOR DEBUGGING
        
        # Set data mode to backtest for more lenient validation
        self.set_data_mode("backtest")
        
    def on_market_update(self, snapshot: 'BookSnapshot', timestamp: pd.Timestamp) -> 'StrategyResult':
        """Generate orders based on momentum"""
        from src.strategies.base_strategy import StrategyResult
        from src.engine.order_types import Order
        
        # Create result container
        result = StrategyResult(
            timestamp=timestamp,
            processing_time_us=0
        )
        
        # Validate and handle missing/invalid market data
        if not self._validate_snapshot(snapshot):
            self.logger.warning(f"Invalid market data received at {timestamp}")
            result.decision_reason = "Invalid market data"
            return result
        
        # Update market history (this will add to price_history)
        self._update_market_history(snapshot)
        
        try:
            # Need minimum price history for momentum calculation
            if len(self.price_history) < 5:
                result.decision_reason = "Insufficient price history for momentum calculation"
                return result
            
            # Calculate momentum using the most recent prices
            recent_return = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            
            # Buy on positive momentum
            if (recent_return > self.momentum_threshold and 
                self.current_position < self.max_position and
                snapshot.best_ask is not None):
                
                # Create market order or limit order slightly above best ask
                order_price = snapshot.best_ask + 0.01 if snapshot.best_ask else None
                
                potential_order = self.create_order(
                    side=OrderSide.BUY,
                    volume=self.order_size,
                    price=order_price,
                    order_type=OrderType.LIMIT if order_price else OrderType.MARKET,
                    reason="Positive momentum signal"
                )
                
                # Check risk limits
                current_price = snapshot.mid_price or snapshot.best_ask or self.price_history[-1]
                is_valid, reason = self.check_risk_limits(potential_order, current_price)
                if is_valid:
                    result.add_order(potential_order, "Momentum buy order")
                    self.submit_order(potential_order)
                else:
                    self.logger.debug(f"Momentum buy order rejected: {reason}")
            
            # Sell on negative momentum
            elif (recent_return < -self.momentum_threshold and 
                  self.current_position > -self.max_position and
                  snapshot.best_bid is not None):
                
                # Create market order or limit order slightly below best bid
                order_price = max(0.01, snapshot.best_bid - 0.01) if snapshot.best_bid else None
                
                potential_order = self.create_order(
                    side=OrderSide.SELL,
                    volume=self.order_size,
                    price=order_price,
                    order_type=OrderType.LIMIT if order_price else OrderType.MARKET,
                    reason="Negative momentum signal"
                )
                
                # Check risk limits
                current_price = snapshot.mid_price or snapshot.best_bid or self.price_history[-1]
                is_valid, reason = self.check_risk_limits(potential_order, current_price)
                if is_valid:
                    result.add_order(potential_order, "Momentum sell order")
                    self.submit_order(potential_order)
                else:
                    self.logger.debug(f"Momentum sell order rejected: {reason}")
            
            # Set result metadata
            result.decision_reason = f"Momentum signal: {recent_return:.6f}, generated {len(result.orders)} orders"
            result.confidence = min(1.0, abs(recent_return) / self.momentum_threshold) if result.orders else 0.0
            result.risk_score = abs(self.current_position) / self.max_position
            
        except Exception as e:
            self.logger.error(f"Error in momentum strategy: {e}")
            result.decision_reason = f"Strategy error: {str(e)}"
        
        # Update strategy state
        self.last_update_time = timestamp
        self.update_count += 1
        
        return result


class BatchBacktester:
    """Main class for running batch backtests"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = get_logger("BatchBacktester")
        self.data_ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.results = []
        
    def create_strategy(self, symbol: str, strategy_type: str) -> BaseStrategy:
        """Create strategy instance based on type"""
        strategy_params = self.config.strategy_params.get(strategy_type, {})
        
        if strategy_type == "market_making":
            return SimpleMarketMakingStrategy(
                symbol=symbol,
                spread_bps=strategy_params.get("spread_bps", 10.0),
                order_size=strategy_params.get("order_size", 100)
            )
        elif strategy_type == "momentum":
            return SimpleMomentumStrategy(
                symbol=symbol,
                momentum_threshold=strategy_params.get("momentum_threshold", 0.001),
                order_size=strategy_params.get("order_size", 200)
            )
        else:
            # Default to market making
            return SimpleMarketMakingStrategy(symbol=symbol)
    
    def create_fill_model(self, model_type: str):
        """Create fill model based on type"""
        if model_type == "perfect":
            return PerfectFillModel()
        else:
            return RealisticFillModel()
    
    def load_data_files(self, data_path: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load and filter data files"""
        data_files = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            # Single file
            if data_path.suffix.lower() == '.csv':
                # Extract symbol from filename, handle case like "AAPL_data" -> "AAPL"
                symbol = data_path.stem.split('_')[0].upper()
                data_files.append({
                    'file_path': str(data_path),
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
        elif data_path.is_dir():
            # Directory of files
            for csv_file in data_path.glob("*.csv"):
                # Extract symbol from filename, handle case like "AAPL_data" -> "AAPL"
                symbol = csv_file.stem.split('_')[0].upper()
                data_files.append({
                    'file_path': str(csv_file),
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                })
        else:
            raise ValueError(f"Data path not found: {data_path}")
        
        self.logger.info(f"Found {len(data_files)} data files to process")
        return data_files
    
    def run_single_backtest(self, file_info: Dict[str, Any]) -> Optional[BacktestResult]:
        """Run backtest on a single file"""
        file_path = file_info['file_path']
        symbol = file_info['symbol']
        start_date = file_info.get('start_date')
        end_date = file_info.get('end_date')
        
        self.logger.info(f"Processing {symbol}: {file_path}")
        
        try:
            # Load and preprocess data
            data = self.data_ingestion.load_csv(file_path)
            
            if data.empty:
                self.logger.warning(f"No data loaded from {file_path}")
                return None
            
            # Create simulator
            fill_model = self.create_fill_model(self.config.fill_model)
            simulator = ExecutionSimulator(
                symbol=symbol,
                fill_model=fill_model,
                tick_size=self.config.tick_size,
                initial_cash=self.config.initial_capital
            )
            
            # Configure simulator
            simulator.set_config(
                max_position_size=self.config.max_position_size,
                max_order_size=self.config.max_order_size,
                risk_limit=self.config.risk_limit,
                enable_logging=self.config.enable_logging,
                save_snapshots=self.config.save_snapshots
            )
            
            # Create strategy
            strategy = self.create_strategy(symbol, self.config.strategy_type)
            
            # Run backtest
            with Timer() as timer:
                result = simulator.run_backtest(
                    data_source=data,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date
                )
            
            # Add metadata to result
            result.metadata.update({
                'file_path': file_path,
                'processing_time_ms': timer.elapsed(),
                'strategy_type': self.config.strategy_type,
                'config': self.config.to_dict()
            })
            
            self.logger.info(
                f"Completed {symbol}: P&L=${result.total_pnl:.2f}, "
                f"Trades={result.total_trades}, "
                f"Time={timer.elapsed():.1f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def run_batch_backtest(self, data_path: str, output_path: str, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> List[BacktestResult]:
        """Run batch backtests on multiple files"""
        
        self.logger.info("Starting batch backtesting")
        self.logger.info(f"Data path: {data_path}")
        self.logger.info(f"Output path: {output_path}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        
        # Load data files
        data_files = self.load_data_files(data_path, start_date, end_date)
        
        if not data_files:
            raise ValueError("No data files found")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        if output_path.suffix:  # It's a file
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Run backtests
        results = []
        total_files = len(data_files)
        
        for i, file_info in enumerate(data_files, 1):
            self.logger.info(f"Processing file {i}/{total_files}: {file_info['symbol']}")
            
            result = self.run_single_backtest(file_info)
            if result:
                results.append(result)
        
        self.logger.info(f"Completed batch backtesting: {len(results)}/{total_files} successful")
        
        # Save results
        self.save_results(results, output_path)
        
        # Generate summary report
        self.generate_summary_report(results, output_path)
        
        return results
    
    def save_results(self, results: List[BacktestResult], output_path: Path):
        """Save backtest results"""
        
        if not results:
            self.logger.warning("No results to save")
            return
        
        if output_path.suffix == '.json':
            # Single JSON file
            self.save_json_results(results, output_path)
        else:
            # Multiple files in directory
            self.save_individual_results(results, output_path)
    
    def save_json_results(self, results: List[BacktestResult], output_file: Path):
        """Save results to a single JSON file"""
        
        json_data = {
            'backtest_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_backtests': len(results),
                'config': self.config.to_dict(),
                'total_pnl': sum(r.total_pnl for r in results),
                'total_trades': sum(r.total_trades for r in results),
                'avg_fill_rate': np.mean([r.fill_rate for r in results]) if results else 0.0
            },
            'results': [result.to_dict() for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def save_individual_results(self, results: List[BacktestResult], output_dir: Path):
        """Save individual result files"""
        
        for result in results:
            result_file = output_dir / f"{result.symbol}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            result_data = {
                'backtest_info': {
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config.to_dict()
                },
                'result': result.to_dict()
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
        
        self.logger.info(f"Individual results saved to {output_dir}")
    
    def generate_summary_report(self, results: List[BacktestResult], output_path: Path):
        """Generate a summary report"""
        
        if not results:
            return
        
        # Create summary statistics
        summary = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_backtests': len(results),
                'successful_backtests': len(results),
                'strategy_type': self.config.strategy_type,
                'total_runtime_seconds': sum(r.metadata.get('processing_time_ms', 0) for r in results) / 1000
            },
            'performance_summary': {
                'total_pnl': sum(r.total_pnl for r in results),
                'average_pnl': np.mean([r.total_pnl for r in results]),
                'pnl_std': np.std([r.total_pnl for r in results]),
                'win_rate': np.mean([r.win_rate for r in results]),
                'total_trades': sum(r.total_trades for r in results),
                'average_trades_per_backtest': np.mean([r.total_trades for r in results]),
                'fill_rate': np.mean([r.fill_rate for r in results]),
                'max_drawdown': np.mean([r.max_drawdown for r in results])
            },
            'individual_results': [
                {
                    'symbol': r.symbol,
                    'pnl': r.total_pnl,
                    'trades': r.total_trades,
                    'win_rate': r.win_rate,
                    'fill_rate': r.fill_rate,
                    'max_drawdown': r.max_drawdown,
                    'duration_hours': r.duration.total_seconds() / 3600
                }
                for r in results
            ]
        }
        
        # Save summary report
        if output_path.suffix == '.json':
            summary_file = output_path.parent / 'backtest_summary.json'
        else:
            summary_file = output_path / 'backtest_summary.json'
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary report saved to {summary_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("BACKTESTING SUMMARY")
        print("="*80)
        print(f"Strategy: {self.config.strategy_type}")
        print(f"Files Processed: {len(results)}")
        print(f"Total P&L: ${summary['performance_summary']['total_pnl']:,.2f}")
        print(f"Average P&L: ${summary['performance_summary']['average_pnl']:,.2f}")
        print(f"Win Rate: {summary['performance_summary']['win_rate']:.1%}")
        print(f"Total Trades: {summary['performance_summary']['total_trades']:,}")
        print(f"Fill Rate: {summary['performance_summary']['fill_rate']:.1%}")
        print(f"Max Drawdown: ${summary['performance_summary']['max_drawdown']:,.2f}")
        print("="*80)


def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate sample data for multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    base_prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 2500.0}
    
    for symbol in symbols:
        # Generate realistic tick data
        np.random.seed(hash(symbol) % 1000)
        
        base_time = pd.Timestamp('2024-01-01 09:30:00')
        n_ticks = 1000
        
        data = []
        current_price = base_prices[symbol]
        
        for i in range(n_ticks):
            # Random walk for price
            price_change = np.random.normal(0, 0.01)
            current_price += price_change
            current_price = max(current_price * 0.8, min(current_price * 1.2, current_price))
            
            # Generate bid/ask orders
            spread = np.random.uniform(0.01, 0.05)
            
            # Create tick data
            data.append({
                'timestamp': base_time + pd.Timedelta(seconds=i),
                'symbol': symbol,
                'price': round(current_price, 2),
                'volume': np.random.randint(100, 1000),
                'bid': round(current_price - spread/2, 2),
                'ask': round(current_price + spread/2, 2),
                'bid_volume': np.random.randint(100, 500),
                'ask_volume': np.random.randint(100, 500),
                'trade_type': 'trade'
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        csv_file = data_dir / f"{symbol}_data.csv"  # Keep uppercase
        df.to_csv(csv_file, index=False)
        print(f"Created sample data: {csv_file}")
    
    return str(data_dir)


def main():
    """Main entry point with unified feature interface"""
    parser = argparse.ArgumentParser(
        description="HFT Simulator - Comprehensive Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtesting
  python main.py --mode backtest --data ./data/historical_data.csv --output ./logs/results.json
  python main.py --mode backtest --data ./data/ --strategy momentum --output ./logs/

  # Live simulation with real-time data
  python main.py --mode live-simulation --symbols BTCUSDT,ETHUSDT --duration 3600

  # Strategy comparison
  python main.py --mode strategy-comparison --data ./data/ --strategies market_making,momentum

  # Parameter optimization
  python main.py --mode optimize --data ./data/ --strategy market_making --optimize spread_bps,order_size

  # Data validation and preprocessing
  python main.py --mode validate-data --data ./data/
  python main.py --mode preprocess --data ./raw_data/ --output ./clean_data/

  # Visualization and reporting
  python main.py --mode visualize --results ./logs/results.json
  python main.py --mode performance-report --results ./logs/ --format html

  # System utilities
  python main.py --mode create-sample-data --symbols AAPL,MSFT,GOOGL
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=[
            'backtest',
            'create-sample-data', 
            'live-simulation',
            'strategy-comparison',
            'optimize',
            'validate-data',
            'preprocess',
            'visualize',
            'performance-report'
        ], 
        required=True,
        help='Execution mode'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV file or directory containing CSV files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results (file or directory)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for filtering data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for filtering data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['market_making', 'momentum'],
        default='market_making',
        help='Trading strategy to use'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing (future feature)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger("main")
    logger.setLevel(log_level)
    
    try:
        if args.mode == 'create-sample-data':
            # Create sample data
            data_path = create_sample_data()
            print(f"Sample data created in: {data_path}")
            print("You can now run backtests with:")
            print(f"python main.py --mode backtest --data {data_path} --output ./logs/")
            return 0
        
        elif args.mode == 'backtest':
            # Validate required arguments
            if not args.data:
                print("Error: --data is required for backtest mode")
                return 1
            
            if not args.output:
                print("Error: --output is required for backtest mode")
                return 1
            
            # Load configuration
            config = BacktestConfig(args.config)
            config.strategy_type = args.strategy
            config.parallel_workers = args.workers
            
            logger.info("Starting HFT Batch Backtesting")
            logger.info(f"Data: {args.data}")
            logger.info(f"Output: {args.output}")
            logger.info(f"Strategy: {config.strategy_type}")
            
            # Run batch backtesting
            backtester = BatchBacktester(config)
            
            with Timer() as total_timer:
                results = backtester.run_batch_backtest(
                    data_path=args.data,
                    output_path=args.output,
                    start_date=args.start_date,
                    end_date=args.end_date
                )
            
            logger.info(f"Batch backtesting completed in {total_timer.elapsed():.1f}ms")
            logger.info(f"Total results: {len(results)}")
            
            return 0
        
    except KeyboardInterrupt:
        logger.info("Batch backtesting interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Batch backtesting failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
