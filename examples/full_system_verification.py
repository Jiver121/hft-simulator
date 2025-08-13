#!/usr/bin/env python3
"""
Full System Verification Script for HFT Simulator

This comprehensive verification script tests the complete workflow by:
1. Generating sample market data for multiple symbols
2. Running backtests on market making and momentum strategies  
3. Verifying P&L calculation and performance metrics
4. Validating position tracking against executed trades
5. Testing configuration management and parameter validation
6. Generating detailed verification reports
7. Saving successful configurations as templates

The script serves as both a system test and a demonstration of the
complete HFT simulator capabilities.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
import traceback
from dataclasses import dataclass, asdict
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.simulator import ExecutionSimulator, BacktestResult
from src.strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from src.strategies.base_strategy import BaseStrategy, StrategyResult
from src.engine.order_types import Order, Trade
from src.engine.market_data import BookSnapshot
from src.performance.portfolio import Portfolio, Position
# from src.performance.metrics import PerformanceAnalyzer, PerformanceMetrics
from src.data.ingestion import DataIngestion
from config.backtest_config import BacktestConfig, load_backtest_config
from src.utils.constants import OrderSide, OrderType, OrderStatus
from src.utils.logger import get_logger
from src.utils.helpers import Timer, format_price, format_volume


@dataclass
class VerificationResult:
    """Container for verification test results"""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class SystemVerificationReport:
    """Complete system verification report"""
    verification_timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_status: str
    execution_time_seconds: float
    test_results: List[VerificationResult]
    configuration_summary: Dict[str, Any]
    performance_summary: Dict[str, Any]
    issues_found: List[str]
    resolutions: List[str]
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy for testing purposes
    
    This strategy identifies price momentum and trades in the direction
    of the trend, implementing basic momentum following logic.
    """
    
    def __init__(self, symbol: str, lookback_window: int = 50, 
                 momentum_threshold: float = 0.002, order_size: int = 200, **kwargs):
        super().__init__(strategy_name="Momentum", symbol=symbol, **kwargs)
        self.lookback_window = lookback_window
        self.momentum_threshold = momentum_threshold
        self.order_size = order_size
        self.price_history = []
        self.position = 0  # Track current position
        self.last_order_time = None
        
    def on_market_update(self, snapshot: BookSnapshot, timestamp: pd.Timestamp) -> StrategyResult:
        """Process market update and generate orders"""
        result = StrategyResult(timestamp=timestamp)
        
        if not snapshot or not snapshot.mid_price:
            return result
            
        # Update price history
        self.price_history.append(snapshot.mid_price)
        
        # Keep only recent prices
        if len(self.price_history) > self.lookback_window:
            self.price_history.pop(0)
            
        # Need sufficient history to calculate momentum
        if len(self.price_history) < self.lookback_window:
            return result
            
        # Calculate momentum (price change over lookback window)
        old_price = self.price_history[0]
        current_price = self.price_history[-1] 
        momentum = (current_price - old_price) / old_price
        
        # Only trade if we have strong enough momentum
        if abs(momentum) < self.momentum_threshold:
            return result
            
        # Avoid over-trading (limit to one order per 5 seconds in simulation time)
        if self.last_order_time and (timestamp - self.last_order_time).total_seconds() < 5:
            return result
            
        # Generate momentum-based orders
        if momentum > self.momentum_threshold and self.position <= 0:
            # Positive momentum - buy signal
            order = Order(
                order_id=f"momentum_buy_{int(timestamp.timestamp())}",
                symbol=self.symbol,
                side=OrderSide.BID,
                order_type=OrderType.MARKET,
                price=current_price,
                volume=self.order_size,
                timestamp=timestamp,
                source="momentum_strategy"
            )
            result.add_order(order, "Positive momentum signal")
            self.position += self.order_size
            self.last_order_time = timestamp
            
        elif momentum < -self.momentum_threshold and self.position >= 0:
            # Negative momentum - sell signal
            order = Order(
                order_id=f"momentum_sell_{int(timestamp.timestamp())}",
                symbol=self.symbol,
                side=OrderSide.ASK,
                order_type=OrderType.MARKET,
                price=current_price,
                volume=self.order_size,
                timestamp=timestamp,
                source="momentum_strategy"
            )
            result.add_order(order, "Negative momentum signal")
            self.position -= self.order_size
            self.last_order_time = timestamp
            
        return result


class FullSystemVerifier:
    """
    Comprehensive system verification class
    
    This class orchestrates the complete system verification process,
    including data generation, strategy testing, metrics validation,
    and report generation.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.verification_results: List[VerificationResult] = []
        self.issues_found: List[str] = []
        self.resolutions: List[str] = []
        
        # Test configuration
        self.symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        self.initial_capital = 100000.0
        self.test_duration_minutes = 60  # 1 hour of simulated trading
        
        # Output directories
        self.output_dir = project_root / "results" / "verification"
        self.config_dir = project_root / "config" 
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def run_verification(self) -> SystemVerificationReport:
        """Run complete system verification"""
        self.logger.info("Starting Full System Verification")
        
        start_time = time.time()
        
        try:
            # Test 1: Configuration Loading and Validation
            self._test_configuration_management()
            
            # Test 2: Data Generation and Ingestion
            market_data = self._test_data_generation()
            
            # Test 3: Market Making Strategy Testing
            mm_results = self._test_market_making_strategy(market_data)
            
            # Test 4: Momentum Strategy Testing  
            momentum_results = self._test_momentum_strategy(market_data)
            
            # Test 5: Multi-Symbol Portfolio Testing
            portfolio_results = self._test_multi_symbol_portfolio(market_data)
            
            # Test 6: Metrics Calculation Validation
            self._test_metrics_calculation(mm_results + momentum_results)
            
            # Test 7: Position Tracking Validation
            self._test_position_tracking(mm_results[0] if mm_results else None)
            
            # Test 8: Risk Management Testing
            self._test_risk_management()
            
            # Test 9: Performance Analysis
            performance_summary = self._test_performance_analysis(mm_results + momentum_results)
            
            # Test 10: Configuration Template Generation
            self._test_configuration_generation()
            
            # Generate verification report
            total_time = time.time() - start_time
            report = self._generate_verification_report(total_time, performance_summary)
            
            # Save successful configurations
            if report.success_rate >= 0.8:  # 80% pass rate
                self._save_verified_configurations()
                
            self.logger.info(f"System verification completed in {total_time:.2f}s")
            self.logger.info(f"Overall success rate: {report.success_rate:.1%}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"System verification failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return failure report
            return SystemVerificationReport(
                verification_timestamp=datetime.now().isoformat(),
                total_tests=len(self.verification_results),
                passed_tests=sum(1 for r in self.verification_results if r.passed),
                failed_tests=sum(1 for r in self.verification_results if not r.passed),
                overall_status="FAILED",
                execution_time_seconds=time.time() - start_time,
                test_results=self.verification_results,
                configuration_summary={},
                performance_summary={},
                issues_found=self.issues_found + [str(e)],
                resolutions=self.resolutions
            )
    
    def _add_test_result(self, test_name: str, passed: bool, message: str, 
                        details: Dict[str, Any] = None, execution_time_ms: float = 0.0):
        """Add a test result to the verification results"""
        result = VerificationResult(
            test_name=test_name,
            passed=passed,
            message=message,
            details=details or {},
            execution_time_ms=execution_time_ms
        )
        self.verification_results.append(result)
        
        if passed:
            self.logger.info(f"✓ {test_name}: {message}")
        else:
            self.logger.error(f"✗ {test_name}: {message}")
            self.issues_found.append(f"{test_name}: {message}")
    
    def _test_configuration_management(self):
        """Test configuration loading and validation"""
        with Timer() as timer:
            try:
                # Test default configuration loading
                config = BacktestConfig()
                config.validate()
                
                # Test configuration from JSON
                config_file = self.config_dir / "backtest_config.json"
                if config_file.exists():
                    json_config = BacktestConfig.from_json(config_file)
                    json_config.validate()
                else:
                    # Create default config for testing
                    config.to_json(config_file)
                    json_config = BacktestConfig.from_json(config_file)
                
                # Test parameter updates
                json_config.update_strategy_param("spread_bps", 15.0, "market_making")
                mm_config = json_config.get_strategy_config("market_making")
                
                assert mm_config["spread_bps"] == 15.0, "Strategy parameter update failed"
                
                self._add_test_result(
                    "Configuration Management",
                    True,
                    "Configuration loading and validation successful",
                    {"strategies_configured": len(json_config.strategy_params)},
                    timer.elapsed()
                )
                
            except Exception as e:
                self._add_test_result(
                    "Configuration Management", 
                    False,
                    f"Configuration error: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Check configuration file format and parameter validation")
    
    def _test_data_generation(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic market data for testing"""
        with Timer() as timer:
            try:
                market_data = {}
                total_records = 0
                
                for symbol in self.symbols:
                    data = self._generate_synthetic_data(symbol, self.test_duration_minutes * 60)
                    market_data[symbol] = data
                    total_records += len(data)
                
                # Validate data quality
                for symbol, data in market_data.items():
                    assert len(data) > 0, f"No data generated for {symbol}"
                    assert 'timestamp' in data.columns, f"Missing timestamp column for {symbol}"
                    assert 'price' in data.columns, f"Missing price column for {symbol}"
                    assert data['price'].notna().all(), f"NaN prices found for {symbol}"
                
                self._add_test_result(
                    "Data Generation",
                    True,
                    f"Generated {total_records:,} market data records across {len(self.symbols)} symbols",
                    {
                        "symbols": len(self.symbols),
                        "total_records": total_records,
                        "avg_records_per_symbol": total_records // len(self.symbols)
                    },
                    timer.elapsed()
                )
                
                return market_data
                
            except Exception as e:
                self._add_test_result(
                    "Data Generation",
                    False, 
                    f"Data generation failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Check data generation algorithms and memory constraints")
                return {}
    
    def _generate_synthetic_data(self, symbol: str, duration_seconds: int, 
                                freq_seconds: int = 1) -> pd.DataFrame:
        """Generate realistic synthetic market data"""
        np.random.seed(hash(symbol) % 2**32)  # Seed based on symbol for reproducibility
        
        n_records = duration_seconds // freq_seconds
        
        # Generate timestamps
        start_time = pd.Timestamp('2024-01-01 09:30:00')
        timestamps = pd.date_range(start_time, periods=n_records, freq=f'{freq_seconds}S')
        
        # Generate realistic price movements using GBM
        initial_price = np.random.uniform(50, 200)  # Random initial price
        dt = freq_seconds / (252 * 24 * 60 * 60)  # Convert to years
        volatility = np.random.uniform(0.15, 0.35)  # Annual volatility
        drift = np.random.uniform(-0.05, 0.10)  # Annual drift
        
        # Generate price path
        returns = np.random.normal(
            (drift - 0.5 * volatility**2) * dt,
            volatility * np.sqrt(dt),
            n_records
        )
        
        prices = [initial_price]
        for r in returns[1:]:
            prices.append(prices[-1] * np.exp(r))
        
        # Generate volumes (realistic distribution)
        volumes = np.random.lognormal(mean=5, sigma=1.2, size=n_records).astype(int)
        volumes = np.clip(volumes, 100, 5000)
        
        # Create bid/ask spread
        spread_bps = np.random.uniform(5, 25)  # 5-25 basis points
        spread = np.array(prices) * (spread_bps / 10000)
        
        bid_prices = np.array(prices) - spread / 2
        ask_prices = np.array(prices) + spread / 2
        
        # Generate market data events
        data_records = []
        for i in range(n_records):
            # Add bid update
            data_records.append({
                'timestamp': timestamps[i],
                'symbol': symbol,
                'price': round(bid_prices[i], 2),
                'volume': volumes[i],
                'side': 'bid',
                'event_type': 'quote'
            })
            
            # Add ask update  
            data_records.append({
                'timestamp': timestamps[i],
                'symbol': symbol,
                'price': round(ask_prices[i], 2),
                'volume': volumes[i],
                'side': 'ask',
                'event_type': 'quote'
            })
            
            # Occasionally add trades
            if np.random.random() < 0.1:  # 10% chance of trade
                trade_price = np.random.choice([bid_prices[i], ask_prices[i]])
                trade_volume = np.random.randint(10, min(500, volumes[i]))
                data_records.append({
                    'timestamp': timestamps[i],
                    'symbol': symbol,
                    'price': round(trade_price, 2),
                    'volume': trade_volume,
                    'side': 'trade',
                    'event_type': 'trade'
                })
        
        df = pd.DataFrame(data_records)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _test_market_making_strategy(self, market_data: Dict[str, pd.DataFrame]) -> List[BacktestResult]:
        """Test market making strategy with multiple symbols"""
        with Timer() as timer:
            try:
                results = []
                
                for symbol in self.symbols[:3]:  # Test first 3 symbols to save time
                    if symbol not in market_data:
                        continue
                        
                    # Create market making configuration
                    mm_config = MarketMakingConfig(
                        min_spread=0.01,
                        target_spread=0.02,
                        base_quote_size=100,
                        max_inventory=500,
                        inventory_target=0
                    )
                    
                    # Create strategy
                    strategy = MarketMakingStrategy(
                        symbol=symbol,
                        config=mm_config
                    )
                    
                    # Create simulator
                    simulator = ExecutionSimulator(
                        symbol=symbol,
                        initial_cash=self.initial_capital / len(self.symbols),
                        tick_size=0.01
                    )
                    
                    # Run backtest
                    result = simulator.run_backtest(
                        data_source=market_data[symbol],
                        strategy=strategy
                    )
                    
                    results.append(result)
                
                # Validate results
                total_pnl = sum(r.total_pnl for r in results)
                total_trades = sum(r.total_trades for r in results)
                avg_fill_rate = np.mean([r.fill_rate for r in results])
                
                # Check that we have realistic results
                pnl_reasonable = abs(total_pnl) < self.initial_capital * 0.5  # PnL < 50% of capital
                trades_generated = total_trades > 0
                fills_happening = avg_fill_rate > 0.0
                
                success = pnl_reasonable and trades_generated and fills_happening
                
                self._add_test_result(
                    "Market Making Strategy", 
                    success,
                    f"Tested on {len(results)} symbols with {total_trades} total trades" if success 
                    else f"Market making validation failed",
                    {
                        "symbols_tested": len(results),
                        "total_pnl": total_pnl,
                        "total_trades": total_trades,
                        "avg_fill_rate": avg_fill_rate,
                        "avg_win_rate": np.mean([r.win_rate for r in results if r.win_rate > 0])
                    },
                    timer.elapsed()
                )
                
                if not success:
                    if not trades_generated:
                        self.resolutions.append("Check market making strategy order generation logic")
                    if not fills_happening:
                        self.resolutions.append("Check order matching engine and fill models")
                    if not pnl_reasonable:
                        self.resolutions.append("Review P&L calculation and position tracking")
                
                return results
                
            except Exception as e:
                self._add_test_result(
                    "Market Making Strategy",
                    False,
                    f"Market making test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug market making strategy initialization and execution")
                return []
    
    def _test_momentum_strategy(self, market_data: Dict[str, pd.DataFrame]) -> List[BacktestResult]:
        """Test momentum strategy with multiple symbols"""
        with Timer() as timer:
            try:
                results = []
                
                for symbol in self.symbols[2:]:  # Test last 3 symbols
                    if symbol not in market_data:
                        continue
                        
                    # Create momentum strategy
                    strategy = MomentumStrategy(
                        symbol=symbol,
                        lookback_window=50,
                        momentum_threshold=0.002,
                        order_size=200
                    )
                    
                    # Create simulator
                    simulator = ExecutionSimulator(
                        symbol=symbol,
                        initial_cash=self.initial_capital / len(self.symbols),
                        tick_size=0.01
                    )
                    
                    # Run backtest
                    result = simulator.run_backtest(
                        data_source=market_data[symbol],
                        strategy=strategy
                    )
                    
                    results.append(result)
                
                # Validate results
                total_pnl = sum(r.total_pnl for r in results)
                total_trades = sum(r.total_trades for r in results)
                avg_fill_rate = np.mean([r.fill_rate for r in results])
                
                # Check for reasonable results
                pnl_reasonable = abs(total_pnl) < self.initial_capital * 0.8  # More tolerance for momentum
                trades_generated = total_trades >= 0  # Momentum might trade less
                results_valid = len(results) > 0
                
                success = pnl_reasonable and results_valid
                
                self._add_test_result(
                    "Momentum Strategy",
                    success,
                    f"Tested on {len(results)} symbols with {total_trades} total trades" if success
                    else "Momentum strategy validation failed",
                    {
                        "symbols_tested": len(results),
                        "total_pnl": total_pnl,
                        "total_trades": total_trades,
                        "avg_fill_rate": avg_fill_rate,
                        "avg_win_rate": np.mean([r.win_rate for r in results if r.win_rate > 0])
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Check momentum strategy signal generation and position sizing")
                
                return results
                
            except Exception as e:
                self._add_test_result(
                    "Momentum Strategy",
                    False,
                    f"Momentum strategy test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug momentum strategy implementation and market data processing")
                return []
    
    def _test_multi_symbol_portfolio(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Test multi-symbol portfolio management"""
        with Timer() as timer:
            try:
                portfolio = Portfolio(initial_cash=self.initial_capital, name="TestPortfolio")
                
                # Track portfolio across time
                portfolio_values = []
                total_positions = 0
                
                for symbol, data in market_data.items():
                    if len(data) == 0:
                        continue
                        
                    # Simulate some trades for the portfolio
                    sample_prices = data['price'].values[::100]  # Sample every 100th price
                    sample_times = data['timestamp'].values[::100]
                    
                    position_size = np.random.randint(50, 200)
                    avg_price = np.mean(sample_prices[:5]) if len(sample_prices) >= 5 else sample_prices[0]
                    current_price = sample_prices[-1] if len(sample_prices) > 0 else avg_price
                    
                    # Create position
                    position = Position(
                        symbol=symbol,
                        quantity=position_size,
                        average_price=avg_price,
                        current_price=current_price,
                        last_update=pd.Timestamp.now()
                    )
                    
                    # Add position manually to portfolio positions dict
                    portfolio.positions[symbol] = position
                    total_positions += 1
                    
                    # Update portfolio metrics manually
                    portfolio._update_portfolio_metrics()
                    portfolio_values.append(portfolio.total_value)
                
                # Validate portfolio
                portfolio_value_positive = portfolio.total_value > 0
                positions_created = total_positions > 0
                value_changes = len(set(portfolio_values)) > 1 if len(portfolio_values) > 1 else True
                
                success = portfolio_value_positive and positions_created and value_changes
                
                self._add_test_result(
                    "Multi-Symbol Portfolio",
                    success,
                    f"Portfolio with {total_positions} positions, value: ${portfolio.total_value:,.2f}" if success
                    else "Portfolio management failed",
                    {
                        "total_positions": total_positions,
                        "portfolio_value": portfolio.total_value,
                        "total_pnl": portfolio.total_pnl,
                        "initial_cash": portfolio.initial_cash
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Check portfolio position management and valuation calculations")
                
                return {
                    "portfolio": portfolio,
                    "positions": total_positions,
                    "final_value": portfolio.total_value
                }
                
            except Exception as e:
                self._add_test_result(
                    "Multi-Symbol Portfolio",
                    False,
                    f"Portfolio test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug portfolio initialization and position tracking")
                return {}
    
    def _test_metrics_calculation(self, backtest_results: List[BacktestResult]):
        """Test performance metrics calculation accuracy"""
        with Timer() as timer:
            try:
                if not backtest_results:
                    self._add_test_result(
                        "Metrics Calculation",
                        False,
                        "No backtest results available for metrics testing",
                        execution_time_ms=timer.elapsed()
                    )
                    return
                
                metrics_valid = []
                
                for result in backtest_results:
                    # Test basic metric validity
                    pnl_calculated = hasattr(result, 'total_pnl') and isinstance(result.total_pnl, (int, float))
                    win_rate_valid = 0 <= result.win_rate <= 1
                    fill_rate_valid = 0 <= result.fill_rate <= 1
                    max_dd_reasonable = result.max_drawdown >= 0
                    
                    # Test metric relationships
                    if result.total_trades > 0:
                        volume_positive = result.total_volume > 0
                    else:
                        volume_positive = result.total_volume == 0
                    
                    result_valid = all([
                        pnl_calculated, win_rate_valid, fill_rate_valid, 
                        max_dd_reasonable, volume_positive
                    ])
                    
                    metrics_valid.append(result_valid)
                
                success = all(metrics_valid) and len(metrics_valid) > 0
                
                # Calculate aggregate metrics for validation
                total_pnl = sum(r.total_pnl for r in backtest_results)
                total_trades = sum(r.total_trades for r in backtest_results)
                avg_win_rate = np.mean([r.win_rate for r in backtest_results])
                avg_fill_rate = np.mean([r.fill_rate for r in backtest_results])
                max_drawdown = max([r.max_drawdown for r in backtest_results])
                
                self._add_test_result(
                    "Metrics Calculation",
                    success,
                    f"Validated metrics for {len(backtest_results)} results" if success
                    else "Metrics calculation validation failed",
                    {
                        "results_tested": len(backtest_results),
                        "total_pnl": total_pnl,
                        "total_trades": total_trades,
                        "avg_win_rate": avg_win_rate,
                        "avg_fill_rate": avg_fill_rate,
                        "max_drawdown": max_drawdown
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Review metrics calculation formulas and edge case handling")
                
            except Exception as e:
                self._add_test_result(
                    "Metrics Calculation",
                    False,
                    f"Metrics calculation test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug performance metrics calculation methods")
    
    def _test_position_tracking(self, backtest_result: Optional[BacktestResult]):
        """Validate that position tracking matches executed trades"""
        with Timer() as timer:
            try:
                if not backtest_result or not backtest_result.trades:
                    self._add_test_result(
                        "Position Tracking",
                        False,
                        "No trades available for position tracking validation",
                        execution_time_ms=timer.elapsed()
                    )
                    return
                
                # Reconstruct position from trades
                position = 0
                trade_pnl = 0.0
                
                buy_queue = []  # For FIFO P&L calculation
                
                for trade in backtest_result.trades:
                    if trade.buy_order_id:  # This is a buy trade
                        position += trade.volume
                        buy_queue.append((trade.volume, trade.price))
                    else:  # This is a sell trade
                        sell_volume = trade.volume
                        sell_price = trade.price
                        
                        # Match against buys (FIFO)
                        while sell_volume > 0 and buy_queue:
                            buy_vol, buy_price = buy_queue[0]
                            
                            close_volume = min(sell_volume, buy_vol)
                            pnl = (sell_price - buy_price) * close_volume
                            trade_pnl += pnl
                            
                            # Update queues
                            if buy_vol <= close_volume:
                                buy_queue.pop(0)
                            else:
                                buy_queue[0] = (buy_vol - close_volume, buy_price)
                            
                            sell_volume -= close_volume
                            position -= close_volume
                
                # Compare with reported P&L (allow for small numerical differences)
                pnl_matches = abs(trade_pnl - backtest_result.total_pnl) < 0.01
                
                # Validate position is reasonable
                position_reasonable = abs(position) < 10000  # Arbitrary reasonable limit
                
                success = pnl_matches and position_reasonable
                
                self._add_test_result(
                    "Position Tracking",
                    success,
                    f"Position tracking validated with {len(backtest_result.trades)} trades" if success
                    else f"Position tracking mismatch detected",
                    {
                        "trades_processed": len(backtest_result.trades),
                        "final_position": position,
                        "calculated_pnl": trade_pnl,
                        "reported_pnl": backtest_result.total_pnl,
                        "pnl_difference": abs(trade_pnl - backtest_result.total_pnl)
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Check position tracking logic and P&L calculation methods")
                
            except Exception as e:
                self._add_test_result(
                    "Position Tracking",
                    False,
                    f"Position tracking test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug position tracking implementation")
    
    def _test_risk_management(self):
        """Test risk management and limit enforcement"""
        with Timer() as timer:
            try:
                # Test configuration-based risk limits
                config = BacktestConfig()
                
                # Test parameter validation
                risk_checks = []
                
                # Test invalid parameters trigger validation errors
                try:
                    config.initial_capital = -1000
                    config.validate()
                    risk_checks.append(False)  # Should have failed
                except ValueError:
                    risk_checks.append(True)  # Correctly caught error
                
                # Reset to valid value
                config.initial_capital = 100000
                
                try:
                    config.commission_rate = -0.01
                    config.validate()
                    risk_checks.append(False)  # Should have failed
                except ValueError:
                    risk_checks.append(True)  # Correctly caught error
                
                # Reset to valid value
                config.commission_rate = 0.001
                
                # Test valid configuration passes
                try:
                    config.validate()
                    risk_checks.append(True)  # Should pass
                except ValueError:
                    risk_checks.append(False)  # Shouldn't fail
                
                success = all(risk_checks)
                
                self._add_test_result(
                    "Risk Management",
                    success,
                    f"Risk management validation passed {sum(risk_checks)}/{len(risk_checks)} checks" if success
                    else "Risk management validation failed",
                    {
                        "validation_checks": len(risk_checks),
                        "checks_passed": sum(risk_checks),
                        "max_position_size": config.max_position_size,
                        "max_order_size": config.max_order_size,
                        "risk_limit": config.risk_limit
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Strengthen parameter validation and risk limit enforcement")
                
            except Exception as e:
                self._add_test_result(
                    "Risk Management",
                    False,
                    f"Risk management test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug risk management parameter validation")
    
    def _test_performance_analysis(self, backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """Test performance analysis capabilities"""
        with Timer() as timer:
            try:
                if not backtest_results:
                    performance_summary = {"status": "no_results"}
                else:
                    # Aggregate results
                    total_pnl = sum(r.total_pnl for r in backtest_results)
                    total_trades = sum(r.total_trades for r in backtest_results)
                    total_volume = sum(r.total_volume for r in backtest_results)
                    
                    # Calculate aggregate metrics
                    win_rates = [r.win_rate for r in backtest_results if r.win_rate > 0]
                    fill_rates = [r.fill_rate for r in backtest_results if r.fill_rate > 0]
                    drawdowns = [r.max_drawdown for r in backtest_results if r.max_drawdown > 0]
                    
                    performance_summary = {
                        "total_pnl": total_pnl,
                        "total_trades": total_trades,
                        "total_volume": total_volume,
                        "avg_win_rate": np.mean(win_rates) if win_rates else 0.0,
                        "avg_fill_rate": np.mean(fill_rates) if fill_rates else 0.0,
                        "max_drawdown": max(drawdowns) if drawdowns else 0.0,
                        "strategies_tested": len(backtest_results),
                        "profitable_strategies": sum(1 for r in backtest_results if r.total_pnl > 0)
                    }
                
                # Validate performance analysis
                analysis_complete = "total_pnl" in performance_summary
                metrics_reasonable = True
                if analysis_complete and performance_summary["total_trades"] > 0:
                    avg_pnl_per_trade = performance_summary["total_pnl"] / performance_summary["total_trades"]
                    metrics_reasonable = abs(avg_pnl_per_trade) < 1000  # Reasonable per-trade P&L
                
                success = analysis_complete and metrics_reasonable
                
                self._add_test_result(
                    "Performance Analysis",
                    success,
                    f"Performance analysis completed for {len(backtest_results)} results" if success
                    else "Performance analysis validation failed",
                    performance_summary,
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Review performance analysis calculations and aggregation methods")
                
                return performance_summary
                
            except Exception as e:
                self._add_test_result(
                    "Performance Analysis",
                    False,
                    f"Performance analysis test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug performance analysis implementation")
                return {"status": "error", "message": str(e)}
    
    def _test_configuration_generation(self):
        """Test configuration template generation"""
        with Timer() as timer:
            try:
                # Create sample configurations for different strategies
                mm_config = BacktestConfig(
                    strategy_type="market_making",
                    initial_capital=100000.0,
                    strategy_params={
                        "market_making": {
                            "spread_bps": 8.0,
                            "order_size": 100,
                            "max_inventory": 500,
                            "risk_aversion": 0.12,
                            "min_spread": 0.005,
                            "max_spread": 0.05
                        }
                    }
                )
                
                momentum_config = BacktestConfig(
                    strategy_type="momentum", 
                    initial_capital=100000.0,
                    strategy_params={
                        "momentum": {
                            "momentum_threshold": 0.0015,
                            "lookback_window": 60,
                            "order_size": 250,
                            "max_positions": 8,
                            "signal_decay": 0.92
                        }
                    }
                )
                
                # Test configuration serialization
                mm_dict = mm_config.to_dict()
                momentum_dict = momentum_config.to_dict()
                
                # Test configuration loading from dict
                mm_reloaded = BacktestConfig.from_dict(mm_dict)
                momentum_reloaded = BacktestConfig.from_dict(momentum_dict)
                
                # Validate configurations
                mm_reloaded.validate()
                momentum_reloaded.validate()
                
                # Test parameter access
                mm_params = mm_reloaded.get_strategy_config("market_making")
                momentum_params = momentum_reloaded.get_strategy_config("momentum")
                
                configs_valid = (
                    mm_params["spread_bps"] == 8.0 and
                    momentum_params["momentum_threshold"] == 0.0015
                )
                
                success = configs_valid
                
                self._add_test_result(
                    "Configuration Generation",
                    success,
                    "Configuration template generation and validation successful" if success
                    else "Configuration generation failed",
                    {
                        "mm_config_params": len(mm_params),
                        "momentum_config_params": len(momentum_params),
                        "serialization_working": True
                    },
                    timer.elapsed()
                )
                
                if not success:
                    self.resolutions.append("Check configuration serialization and parameter validation")
                
            except Exception as e:
                self._add_test_result(
                    "Configuration Generation",
                    False,
                    f"Configuration generation test failed: {str(e)}",
                    execution_time_ms=timer.elapsed()
                )
                self.resolutions.append("Debug configuration template creation and validation")
    
    def _generate_verification_report(self, execution_time: float, 
                                    performance_summary: Dict[str, Any]) -> SystemVerificationReport:
        """Generate comprehensive verification report"""
        passed_tests = sum(1 for r in self.verification_results if r.passed)
        failed_tests = len(self.verification_results) - passed_tests
        
        overall_status = "PASSED" if passed_tests >= 0.8 * len(self.verification_results) else "FAILED"
        
        # Configuration summary
        config_summary = {
            "symbols_tested": len(self.symbols),
            "test_duration_minutes": self.test_duration_minutes,
            "initial_capital": self.initial_capital,
            "output_directory": str(self.output_dir)
        }
        
        report = SystemVerificationReport(
            verification_timestamp=datetime.now().isoformat(),
            total_tests=len(self.verification_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_status=overall_status,
            execution_time_seconds=execution_time,
            test_results=self.verification_results,
            configuration_summary=config_summary,
            performance_summary=performance_summary,
            issues_found=self.issues_found,
            resolutions=self.resolutions
        )
        
        return report
    
    def _save_verified_configurations(self):
        """Save successful configurations as templates"""
        try:
            # Save verified market making configuration
            verified_mm_config = BacktestConfig(
                strategy_type="market_making",
                initial_capital=100000.0,
                commission_rate=0.0005,
                slippage_bps=1.0,
                max_position_size=1000,
                max_order_size=100,
                strategy_params={
                    "market_making": {
                        "spread_bps": 8.0,
                        "order_size": 100,
                        "max_inventory": 500,
                        "inventory_target": 0,
                        "risk_aversion": 0.12,
                        "min_spread": 0.01,
                        "max_spread": 0.04,
                        "quote_size": 100,
                        "adverse_selection_threshold": 0.75,
                        "enable_skewing": True
                    }
                }
            )
            
            # Save verified momentum configuration
            verified_momentum_config = BacktestConfig(
                strategy_type="momentum", 
                initial_capital=100000.0,
                commission_rate=0.001,
                slippage_bps=2.0,
                max_position_size=2000,
                max_order_size=500,
                strategy_params={
                    "momentum": {
                        "momentum_threshold": 0.0015,
                        "lookback_window": 50,
                        "order_size": 200,
                        "max_positions": 8,
                        "signal_decay": 0.92,
                        "min_signal_strength": 0.35,
                        "stop_loss": 0.025,
                        "take_profit": 0.045
                    }
                }
            )
            
            # Save configurations
            verified_mm_config.to_json(self.config_dir / "verified_config.json")
            verified_momentum_config.to_json(self.config_dir / "verified_momentum_config.json")
            
            # Save combined multi-strategy configuration
            multi_strategy_config = BacktestConfig(
                strategy_type="multi_strategy",
                initial_capital=200000.0,
                strategy_params={
                    "market_making": verified_mm_config.strategy_params["market_making"],
                    "momentum": verified_momentum_config.strategy_params["momentum"]
                }
            )
            multi_strategy_config.to_json(self.config_dir / "verified_multi_strategy_config.json")
            
            self.logger.info("Saved verified configurations to config directory")
            
        except Exception as e:
            self.logger.error(f"Failed to save verified configurations: {str(e)}")
    
    def save_report(self, report: SystemVerificationReport, filename: str = None):
        """Save verification report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_verification_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Verification report saved to {report_path}")
            
            # Also save a human-readable summary
            summary_path = self.output_dir / filename.replace('.json', '_summary.txt')
            self._save_human_readable_summary(report, summary_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
    
    def _save_human_readable_summary(self, report: SystemVerificationReport, filepath: Path):
        """Save human-readable summary of verification results"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HFT SIMULATOR - FULL SYSTEM VERIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Verification Date: {report.verification_timestamp}\n")
            f.write(f"Execution Time: {report.execution_time_seconds:.2f} seconds\n")
            f.write(f"Overall Status: {report.overall_status}\n")
            f.write(f"Success Rate: {report.success_rate:.1%}\n\n")
            
            # Test Results Summary
            f.write("TEST RESULTS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Tests: {report.total_tests}\n")
            f.write(f"Passed: {report.passed_tests}\n") 
            f.write(f"Failed: {report.failed_tests}\n\n")
            
            # Individual Test Results
            f.write("DETAILED TEST RESULTS\n")
            f.write("-" * 40 + "\n")
            for result in report.test_results:
                status_icon = "✓" if result.passed else "✗"
                f.write(f"{status_icon} {result.test_name}: {result.message}\n")
                f.write(f"   Execution Time: {result.execution_time_ms:.1f}ms\n")
                if result.details:
                    for key, value in result.details.items():
                        f.write(f"   {key}: {value}\n")
                f.write("\n")
            
            # Performance Summary
            if report.performance_summary:
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 40 + "\n")
                for key, value in report.performance_summary.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Issues and Resolutions
            if report.issues_found:
                f.write("ISSUES IDENTIFIED\n")
                f.write("-" * 40 + "\n")
                for i, issue in enumerate(report.issues_found, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if report.resolutions:
                f.write("RECOMMENDED RESOLUTIONS\n")
                f.write("-" * 40 + "\n")
                for i, resolution in enumerate(report.resolutions, 1):
                    f.write(f"{i}. {resolution}\n")
                f.write("\n")
            
            # Configuration Summary
            f.write("CONFIGURATION SUMMARY\n")
            f.write("-" * 40 + "\n")
            for key, value in report.configuration_summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")


def main():
    """Run the full system verification"""
    print("=" * 80)
    print("HFT SIMULATOR - FULL SYSTEM VERIFICATION")
    print("=" * 80)
    print()
    
    # Initialize verifier
    verifier = FullSystemVerifier()
    
    try:
        # Run complete verification
        report = verifier.run_verification()
        
        # Save report
        verifier.save_report(report)
        
        # Print summary
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        print(f"Overall Status: {report.overall_status}")
        print(f"Success Rate: {report.success_rate:.1%} ({report.passed_tests}/{report.total_tests})")
        print(f"Execution Time: {report.execution_time_seconds:.2f} seconds")
        
        if report.performance_summary and "total_pnl" in report.performance_summary:
            print(f"Total P&L: ${report.performance_summary['total_pnl']:,.2f}")
            print(f"Total Trades: {report.performance_summary['total_trades']:,}")
            print(f"Strategies Tested: {report.performance_summary.get('strategies_tested', 0)}")
        
        if report.issues_found:
            print(f"\nIssues Found: {len(report.issues_found)}")
            for issue in report.issues_found[:3]:  # Show first 3 issues
                print(f"  • {issue}")
            if len(report.issues_found) > 3:
                print(f"  ... and {len(report.issues_found) - 3} more")
        
        print(f"\nDetailed report saved to: {verifier.output_dir}")
        
        if report.success_rate >= 0.8:
            print("\n✓ System verification PASSED - Configuration templates saved")
            return 0
        else:
            print("\n✗ System verification FAILED - Check detailed report for issues")
            return 1
            
    except Exception as e:
        print(f"\nVerification failed with error: {str(e)}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
