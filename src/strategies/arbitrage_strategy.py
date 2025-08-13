"""
Multi-Asset Arbitrage Strategy

This module implements various arbitrage strategies including statistical arbitrage,
pairs trading, and cross-venue arbitrage for HFT trading.

Educational Notes:
- Arbitrage exploits price discrepancies between related assets or venues
- Statistical arbitrage uses mean reversion of price relationships
- Pairs trading involves long/short positions in correlated assets
- Cross-venue arbitrage exploits price differences across exchanges
- Risk management is crucial due to execution and model risks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.strategies.base_strategy import BaseStrategy
from src.engine.order_types import Order, MarketDataPoint
from src.utils.constants import OrderSide, OrderType
from src.utils.logger import get_logger, log_performance
from src.performance.metrics import calculate_correlation, calculate_cointegration


class ArbitrageType(Enum):
    """Types of arbitrage strategies"""
    STATISTICAL = "statistical"
    PAIRS_TRADING = "pairs_trading"
    CROSS_VENUE = "cross_venue"
    TRIANGULAR = "triangular"
    CALENDAR = "calendar"


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    type: ArbitrageType
    assets: List[str]
    expected_profit: float
    confidence: float
    entry_signals: Dict[str, Order]
    exit_signals: Dict[str, Order]
    risk_metrics: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any]


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using pairs trading
    
    This strategy identifies pairs of assets with historical correlation
    and trades on temporary divergences from their normal relationship.
    """
    
    def __init__(self, symbol_pair: Tuple[str, str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize pairs trading strategy
        
        Args:
            symbol_pair: Tuple of two symbols to trade
            config: Strategy configuration
        """
        super().__init__(f"{symbol_pair[0]}_{symbol_pair[1]}", config)
        
        self.symbol_a, self.symbol_b = symbol_pair
        self.symbols = {self.symbol_a, self.symbol_b}
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 100)
        self.entry_threshold = config.get('entry_threshold', 2.0)  # Z-score threshold
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.stop_loss_threshold = config.get('stop_loss_threshold', 4.0)
        self.min_correlation = config.get('min_correlation', 0.7)
        self.position_size = config.get('position_size', 1000)
        self.max_position = config.get('max_position', 5000)
        
        # Cointegration parameters
        self.cointegration_window = config.get('cointegration_window', 252)
        self.min_half_life = config.get('min_half_life', 1)  # days
        self.max_half_life = config.get('max_half_life', 30)  # days
        
        # Data storage
        self.price_history = {self.symbol_a: [], self.symbol_b: []}
        self.spread_history = []
        self.z_score_history = []
        
        # Model parameters
        self.hedge_ratio = 1.0
        self.spread_mean = 0.0
        self.spread_std = 1.0
        self.last_update = None
        
        # Position tracking
        self.positions = {self.symbol_a: 0, self.symbol_b: 0}
        self.entry_prices = {self.symbol_a: None, self.symbol_b: None}
        self.current_spread = None
        self.current_z_score = None
        
        # Performance tracking
        self.opportunities_identified = 0
        self.trades_executed = 0
        self.profitable_trades = 0
        
        self.logger.info(f"PairsTradingStrategy initialized for {self.symbol_a}/{self.symbol_b}")
    
    def add_market_data(self, symbol: str, market_data: MarketDataPoint):
        """
        Add market data for one of the symbols in the pair
        
        Args:
            symbol: Symbol identifier
            market_data: Market data point
        """
        if symbol not in self.symbols:
            return
        
        # Store price data
        self.price_history[symbol].append({
            'timestamp': market_data.timestamp,
            'price': market_data.price,
            'volume': market_data.volume
        })
        
        # Maintain history size
        if len(self.price_history[symbol]) > self.lookback_period * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.lookback_period:]
        
        # Update model if we have data for both symbols
        if self._has_sufficient_data():
            self._update_model()
    
    @log_performance
    def generate_signals(self, market_data: MarketDataPoint) -> List[Order]:
        """
        Generate arbitrage signals based on pairs relationship
        
        Args:
            market_data: Current market data (should specify symbol)
            
        Returns:
            List of orders for arbitrage execution
        """
        # This method should be called after add_market_data
        if not self._has_sufficient_data() or self.current_z_score is None:
            return []
        
        orders = []
        
        # Check for entry signals
        if abs(self.current_z_score) > self.entry_threshold and self._no_open_positions():
            orders.extend(self._generate_entry_orders())
        
        # Check for exit signals
        elif self._has_open_positions():
            if (abs(self.current_z_score) < self.exit_threshold or 
                abs(self.current_z_score) > self.stop_loss_threshold):
                orders.extend(self._generate_exit_orders())
        
        return orders
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for both symbols"""
        return (len(self.price_history[self.symbol_a]) >= self.lookback_period and
                len(self.price_history[self.symbol_b]) >= self.lookback_period)
    
    def _update_model(self):
        """Update the pairs trading model"""
        # Get aligned price data
        df_a = pd.DataFrame(self.price_history[self.symbol_a])
        df_b = pd.DataFrame(self.price_history[self.symbol_b])
        
        # Align timestamps
        df_a.set_index('timestamp', inplace=True)
        df_b.set_index('timestamp', inplace=True)
        
        # Merge on timestamp
        merged = df_a.join(df_b, how='inner', lsuffix='_a', rsuffix='_b')
        
        if len(merged) < self.lookback_period:
            return
        
        prices_a = merged['price_a'].values
        prices_b = merged['price_b'].values
        
        # Calculate hedge ratio using linear regression
        X = prices_b.reshape(-1, 1)
        y = prices_a
        
        reg = LinearRegression().fit(X, y)
        self.hedge_ratio = reg.coef_[0]
        
        # Calculate spread
        spread = prices_a - self.hedge_ratio * prices_b
        self.spread_history = spread.tolist()
        
        # Calculate spread statistics
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)
        
        # Calculate current spread and z-score
        if len(prices_a) > 0 and len(prices_b) > 0:
            current_spread = prices_a[-1] - self.hedge_ratio * prices_b[-1]
            self.current_spread = current_spread
            
            if self.spread_std > 0:
                self.current_z_score = (current_spread - self.spread_mean) / self.spread_std
            else:
                self.current_z_score = 0
        
        # Store z-score history
        if self.current_z_score is not None:
            self.z_score_history.append(self.current_z_score)
            if len(self.z_score_history) > self.lookback_period:
                self.z_score_history = self.z_score_history[-self.lookback_period:]
        
        self.last_update = datetime.now()
    
    def _no_open_positions(self) -> bool:
        """Check if there are no open positions"""
        return all(pos == 0 for pos in self.positions.values())
    
    def _has_open_positions(self) -> bool:
        """Check if there are open positions"""
        return any(pos != 0 for pos in self.positions.values())
    
    def _generate_entry_orders(self) -> List[Order]:
        """Generate entry orders for pairs trade"""
        orders = []
        
        if self.current_z_score > self.entry_threshold:
            # Spread is too high: short A, long B
            # Short symbol A
            orders.append(Order.create_limit_order(
                symbol=self.symbol_a,
                side=OrderSide.ASK,
                volume=self.position_size,
                price=self._get_current_price(self.symbol_a),
                metadata={'strategy': 'pairs_trading', 'action': 'entry_short_a'}
            ))
            
            # Long symbol B
            hedge_volume = int(self.position_size * self.hedge_ratio)
            orders.append(Order.create_limit_order(
                symbol=self.symbol_b,
                side=OrderSide.BID,
                volume=hedge_volume,
                price=self._get_current_price(self.symbol_b),
                metadata={'strategy': 'pairs_trading', 'action': 'entry_long_b'}
            ))
            
        elif self.current_z_score < -self.entry_threshold:
            # Spread is too low: long A, short B
            # Long symbol A
            orders.append(Order.create_limit_order(
                symbol=self.symbol_a,
                side=OrderSide.BID,
                volume=self.position_size,
                price=self._get_current_price(self.symbol_a),
                metadata={'strategy': 'pairs_trading', 'action': 'entry_long_a'}
            ))
            
            # Short symbol B
            hedge_volume = int(self.position_size * self.hedge_ratio)
            orders.append(Order.create_limit_order(
                symbol=self.symbol_b,
                side=OrderSide.ASK,
                volume=hedge_volume,
                price=self._get_current_price(self.symbol_b),
                metadata={'strategy': 'pairs_trading', 'action': 'entry_short_b'}
            ))
        
        if orders:
            self.opportunities_identified += 1
        
        return orders
    
    def _generate_exit_orders(self) -> List[Order]:
        """Generate exit orders to close positions"""
        orders = []
        
        # Close position in symbol A
        if self.positions[self.symbol_a] > 0:
            orders.append(Order.create_limit_order(
                symbol=self.symbol_a,
                side=OrderSide.ASK,
                volume=abs(self.positions[self.symbol_a]),
                price=self._get_current_price(self.symbol_a),
                metadata={'strategy': 'pairs_trading', 'action': 'exit_long_a'}
            ))
        elif self.positions[self.symbol_a] < 0:
            orders.append(Order.create_limit_order(
                symbol=self.symbol_a,
                side=OrderSide.BID,
                volume=abs(self.positions[self.symbol_a]),
                price=self._get_current_price(self.symbol_a),
                metadata={'strategy': 'pairs_trading', 'action': 'exit_short_a'}
            ))
        
        # Close position in symbol B
        if self.positions[self.symbol_b] > 0:
            orders.append(Order.create_limit_order(
                symbol=self.symbol_b,
                side=OrderSide.ASK,
                volume=abs(self.positions[self.symbol_b]),
                price=self._get_current_price(self.symbol_b),
                metadata={'strategy': 'pairs_trading', 'action': 'exit_long_b'}
            ))
        elif self.positions[self.symbol_b] < 0:
            orders.append(Order.create_limit_order(
                symbol=self.symbol_b,
                side=OrderSide.BID,
                volume=abs(self.positions[self.symbol_b]),
                price=self._get_current_price(self.symbol_b),
                metadata={'strategy': 'pairs_trading', 'action': 'exit_short_b'}
            ))
        
        return orders
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        if symbol in self.price_history and self.price_history[symbol]:
            return self.price_history[symbol][-1]['price']
        return 100.0  # Default price
    
    def update_position(self, symbol: str, trade_volume: int, trade_price: float, side: OrderSide):
        """
        Update position after trade execution
        
        Args:
            symbol: Symbol that was traded
            trade_volume: Volume of the trade
            trade_price: Price of the trade
            side: Side of the trade
        """
        if symbol not in self.symbols:
            return
        
        # Update position
        if side == OrderSide.BID:
            self.positions[symbol] += trade_volume
        else:
            self.positions[symbol] -= trade_volume
        
        # Update entry prices
        if self.entry_prices[symbol] is None:
            self.entry_prices[symbol] = trade_price
        
        self.trades_executed += 1
    
    def calculate_pnl(self) -> Dict[str, float]:
        """Calculate current PnL for the pairs trade"""
        pnl = {}
        total_pnl = 0
        
        for symbol in self.symbols:
            if self.positions[symbol] != 0 and self.entry_prices[symbol] is not None:
                current_price = self._get_current_price(symbol)
                position_pnl = self.positions[symbol] * (current_price - self.entry_prices[symbol])
                pnl[symbol] = position_pnl
                total_pnl += position_pnl
            else:
                pnl[symbol] = 0
        
        pnl['total'] = total_pnl
        return pnl
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = super().get_strategy_info()
        
        pairs_info = {
            'symbol_pair': (self.symbol_a, self.symbol_b),
            'hedge_ratio': self.hedge_ratio,
            'current_spread': self.current_spread,
            'current_z_score': self.current_z_score,
            'spread_mean': self.spread_mean,
            'spread_std': self.spread_std,
            'positions': self.positions.copy(),
            'entry_prices': self.entry_prices.copy(),
            'opportunities_identified': self.opportunities_identified,
            'trades_executed': self.trades_executed,
            'current_pnl': self.calculate_pnl(),
            'model_last_update': self.last_update,
            'parameters': {
                'lookback_period': self.lookback_period,
                'entry_threshold': self.entry_threshold,
                'exit_threshold': self.exit_threshold,
                'stop_loss_threshold': self.stop_loss_threshold,
                'min_correlation': self.min_correlation,
            }
        }
        
        base_info.update(pairs_info)
        return base_info


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Multi-asset statistical arbitrage strategy
    
    This strategy identifies and trades on statistical relationships
    between multiple assets using mean reversion principles.
    """
    
    def __init__(self, symbols: List[str], config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical arbitrage strategy
        
        Args:
            symbols: List of symbols to trade
            config: Strategy configuration
        """
        super().__init__("_".join(symbols), config)
        
        self.symbols = set(symbols)
        self.symbol_list = list(symbols)
        
        # Strategy parameters
        self.lookback_period = config.get('lookback_period', 100)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.z_score_entry = config.get('z_score_entry', 2.0)
        self.z_score_exit = config.get('z_score_exit', 0.5)
        self.position_size = config.get('position_size', 1000)
        self.max_positions = config.get('max_positions', 3)
        
        # Data storage
        self.price_data = {symbol: [] for symbol in symbols}
        self.correlation_matrix = None
        self.cointegration_pairs = []
        
        # Active arbitrage opportunities
        self.active_opportunities = []
        self.opportunity_counter = 0
        
        self.logger.info(f"StatisticalArbitrageStrategy initialized for {len(symbols)} symbols")
    
    def add_market_data(self, symbol: str, market_data: MarketDataPoint):
        """Add market data for a symbol"""
        if symbol not in self.symbols:
            return
        
        self.price_data[symbol].append({
            'timestamp': market_data.timestamp,
            'price': market_data.price,
            'volume': market_data.volume
        })
        
        # Maintain data size
        if len(self.price_data[symbol]) > self.lookback_period * 2:
            self.price_data[symbol] = self.price_data[symbol][-self.lookback_period:]
        
        # Update correlations periodically
        if len(self.price_data[symbol]) % 50 == 0:
            self._update_correlations()
    
    def _update_correlations(self):
        """Update correlation matrix and identify cointegrated pairs"""
        # Create price matrix
        price_matrix = {}
        min_length = min(len(data) for data in self.price_data.values() if data)
        
        if min_length < self.lookback_period:
            return
        
        for symbol in self.symbols:
            prices = [point['price'] for point in self.price_data[symbol][-min_length:]]
            price_matrix[symbol] = prices
        
        # Calculate correlation matrix
        df = pd.DataFrame(price_matrix)
        self.correlation_matrix = df.corr()
        
        # Identify highly correlated pairs
        self.cointegration_pairs = []
        for i, symbol_a in enumerate(self.symbol_list):
            for j, symbol_b in enumerate(self.symbol_list[i+1:], i+1):
                correlation = self.correlation_matrix.loc[symbol_a, symbol_b]
                if abs(correlation) > self.correlation_threshold:
                    self.cointegration_pairs.append((symbol_a, symbol_b, correlation))
    
    @log_performance
    def generate_signals(self, market_data: MarketDataPoint) -> List[Order]:
        """Generate statistical arbitrage signals"""
        if not self.cointegration_pairs:
            return []
        
        orders = []
        
        # Check each cointegrated pair for opportunities
        for symbol_a, symbol_b, correlation in self.cointegration_pairs:
            opportunity = self._check_pair_opportunity(symbol_a, symbol_b)
            if opportunity:
                orders.extend(opportunity.entry_signals.values())
                self.active_opportunities.append(opportunity)
        
        # Check exit conditions for active opportunities
        exit_orders = self._check_exit_conditions()
        orders.extend(exit_orders)
        
        return orders
    
    def _check_pair_opportunity(self, symbol_a: str, symbol_b: str) -> Optional[ArbitrageOpportunity]:
        """Check for arbitrage opportunity between a pair of symbols"""
        if (len(self.price_data[symbol_a]) < self.lookback_period or 
            len(self.price_data[symbol_b]) < self.lookback_period):
            return None
        
        # Get recent prices
        prices_a = [p['price'] for p in self.price_data[symbol_a][-self.lookback_period:]]
        prices_b = [p['price'] for p in self.price_data[symbol_b][-self.lookback_period:]]
        
        # Calculate spread
        spread = np.array(prices_a) - np.array(prices_b)
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        if spread_std == 0:
            return None
        
        # Current z-score
        current_spread = prices_a[-1] - prices_b[-1]
        z_score = (current_spread - spread_mean) / spread_std
        
        # Check for entry signal
        if abs(z_score) > self.z_score_entry:
            entry_signals = {}
            
            if z_score > self.z_score_entry:
                # Short A, Long B
                entry_signals[symbol_a] = Order.create_limit_order(
                    symbol=symbol_a,
                    side=OrderSide.ASK,
                    volume=self.position_size,
                    price=prices_a[-1]
                )
                entry_signals[symbol_b] = Order.create_limit_order(
                    symbol=symbol_b,
                    side=OrderSide.BID,
                    volume=self.position_size,
                    price=prices_b[-1]
                )
            else:
                # Long A, Short B
                entry_signals[symbol_a] = Order.create_limit_order(
                    symbol=symbol_a,
                    side=OrderSide.BID,
                    volume=self.position_size,
                    price=prices_a[-1]
                )
                entry_signals[symbol_b] = Order.create_limit_order(
                    symbol=symbol_b,
                    side=OrderSide.ASK,
                    volume=self.position_size,
                    price=prices_b[-1]
                )
            
            # Create opportunity
            opportunity = ArbitrageOpportunity(
                type=ArbitrageType.STATISTICAL,
                assets=[symbol_a, symbol_b],
                expected_profit=abs(z_score) * spread_std * self.position_size,
                confidence=min(abs(z_score) / self.z_score_entry, 1.0),
                entry_signals=entry_signals,
                exit_signals={},
                risk_metrics={'z_score': z_score, 'spread_std': spread_std},
                timestamp=datetime.now(),
                metadata={'spread_mean': spread_mean, 'current_spread': current_spread}
            )
            
            self.opportunity_counter += 1
            return opportunity
        
        return None
    
    def _check_exit_conditions(self) -> List[Order]:
        """Check exit conditions for active opportunities"""
        exit_orders = []
        opportunities_to_remove = []
        
        for opportunity in self.active_opportunities:
            if len(opportunity.assets) != 2:
                continue
            
            symbol_a, symbol_b = opportunity.assets
            
            # Get current prices
            current_price_a = self.price_data[symbol_a][-1]['price'] if self.price_data[symbol_a] else 0
            current_price_b = self.price_data[symbol_b][-1]['price'] if self.price_data[symbol_b] else 0
            
            # Calculate current spread and z-score
            current_spread = current_price_a - current_price_b
            spread_mean = opportunity.metadata['spread_mean']
            spread_std = opportunity.risk_metrics['spread_std']
            current_z_score = (current_spread - spread_mean) / spread_std
            
            # Check exit condition
            if abs(current_z_score) < self.z_score_exit:
                # Generate exit orders (reverse of entry)
                for symbol, entry_order in opportunity.entry_signals.items():
                    exit_side = OrderSide.ASK if entry_order.side == OrderSide.BID else OrderSide.BID
                    exit_order = Order.create_limit_order(
                        symbol=symbol,
                        side=exit_side,
                        volume=entry_order.volume,
                        price=current_price_a if symbol == symbol_a else current_price_b,
                        metadata={'strategy': 'statistical_arbitrage', 'action': 'exit'}
                    )
                    exit_orders.append(exit_order)
                
                opportunities_to_remove.append(opportunity)
        
        # Remove closed opportunities
        for opportunity in opportunities_to_remove:
            self.active_opportunities.remove(opportunity)
        
        return exit_orders
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = super().get_strategy_info()
        
        stat_arb_info = {
            'symbols': list(self.symbols),
            'correlation_matrix': self.correlation_matrix.to_dict() if self.correlation_matrix is not None else None,
            'cointegration_pairs': self.cointegration_pairs,
            'active_opportunities': len(self.active_opportunities),
            'total_opportunities': self.opportunity_counter,
            'parameters': {
                'lookback_period': self.lookback_period,
                'correlation_threshold': self.correlation_threshold,
                'z_score_entry': self.z_score_entry,
                'z_score_exit': self.z_score_exit,
            }
        }
        
        base_info.update(stat_arb_info)
        return base_info


# Utility functions for arbitrage strategy development
def identify_arbitrage_opportunities(price_data: Dict[str, pd.DataFrame],
                                   strategy_type: ArbitrageType = ArbitrageType.STATISTICAL,
                                   **kwargs) -> List[ArbitrageOpportunity]:
    """
    Identify arbitrage opportunities in price data
    
    Args:
        price_data: Dictionary of symbol -> price DataFrame
        strategy_type: Type of arbitrage to look for
        **kwargs: Additional parameters
        
    Returns:
        List of identified opportunities
    """
    opportunities = []
    
    if strategy_type == ArbitrageType.STATISTICAL:
        # Statistical arbitrage between pairs
        symbols = list(price_data.keys())
        correlation_threshold = kwargs.get('correlation_threshold', 0.8)
        z_score_threshold = kwargs.get('z_score_threshold', 2.0)
        
        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i+1:]:
                df_a = price_data[symbol_a]
                df_b = price_data[symbol_b]
                
                # Align data
                merged = df_a.join(df_b, how='inner', lsuffix='_a', rsuffix='_b')
                if len(merged) < 50:
                    continue
                
                # Calculate correlation
                correlation = merged['price_a'].corr(merged['price_b'])
                
                if abs(correlation) > correlation_threshold:
                    # Calculate spread statistics
                    spread = merged['price_a'] - merged['price_b']
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    
                    # Check current z-score
                    current_z_score = (spread.iloc[-1] - spread_mean) / spread_std
                    
                    if abs(current_z_score) > z_score_threshold:
                        opportunity = ArbitrageOpportunity(
                            type=ArbitrageType.STATISTICAL,
                            assets=[symbol_a, symbol_b],
                            expected_profit=abs(current_z_score) * spread_std,
                            confidence=min(abs(current_z_score) / z_score_threshold, 1.0),
                            entry_signals={},
                            exit_signals={},
                            risk_metrics={'z_score': current_z_score, 'correlation': correlation},
                            timestamp=merged.index[-1],
                            metadata={'spread_mean': spread_mean, 'spread_std': spread_std}
                        )
                        opportunities.append(opportunity)
    
    return opportunities


def backtest_arbitrage_strategy(strategy: BaseStrategy,
                               price_data: Dict[str, pd.DataFrame],
                               initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Backtest arbitrage strategy
    
    Args:
        strategy: Arbitrage strategy instance
        price_data: Historical price data
        initial_capital: Starting capital
        
    Returns:
        Backtest results
    """
    results = {
        'trades': [],
        'pnl_series': [],
        'opportunities': [],
        'performance_metrics': {}
    }
    
    # Simulate trading
    capital = initial_capital
    positions = {symbol: 0 for symbol in price_data.keys()}
    
    # Get common time index
    common_index = None
    for df in price_data.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    for timestamp in common_index:
        # Add market data to strategy
        for symbol, df in price_data.items():
            if timestamp in df.index:
                market_point = MarketDataPoint(
                    timestamp=timestamp,
                    price=df.loc[timestamp, 'price'],
                    volume=df.loc[timestamp, 'volume'] if 'volume' in df.columns else 1000
                )
                
                if hasattr(strategy, 'add_market_data'):
                    strategy.add_market_data(symbol, market_point)
        
        # Generate signals
        orders = strategy.generate_signals(market_point)
        
        # Execute orders (simplified)
        for order in orders:
            if order.symbol in positions:
                if order.side == OrderSide.BID:
                    positions[order.symbol] += order.volume
                    capital -= order.volume * order.price
                else:
                    positions[order.symbol] -= order.volume
                    capital += order.volume * order.price
                
                results['trades'].append({
                    'timestamp': timestamp,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'volume': order.volume,
                    'price': order.price,
                    'position': positions[order.symbol]
                })
        
        # Calculate current portfolio value
        portfolio_value = capital
        for symbol, position in positions.items():
            if symbol in price_data and timestamp in price_data[symbol].index:
                current_price = price_data[symbol].loc[timestamp, 'price']
                portfolio_value += position * current_price
        
        results['pnl_series'].append({
            'timestamp': timestamp,
            'pnl': portfolio_value - initial_capital,
            'capital': capital,
            'positions': positions.copy()
        })
    
    # Calculate performance metrics
    if results['pnl_series']:
        pnl_df = pd.DataFrame(results['pnl_series'])
        returns = pnl_df['pnl'].pct_change().dropna()
        
        results['performance_metrics'] = {
            'total_return': (pnl_df['pnl'].iloc[-1] / initial_capital) * 100,
            'num_trades': len(results['trades']),
            'num_opportunities': len(results['opportunities']),
            'win_rate': 0.0,  # Would need to calculate based on individual trade PnL
            'max_drawdown': 0.0,  # Would need to implement drawdown calculation
        }
    
    return results