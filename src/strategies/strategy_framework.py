"""
Strategy Framework and Extensibility System

This module provides a comprehensive framework for developing, testing, and
deploying custom trading strategies in the HFT simulator.

Educational Notes:
- Strategy frameworks enable rapid development and testing of new ideas
- Plugin architecture allows for modular strategy components
- Configuration management enables parameter optimization
- Event-driven architecture supports real-time strategy execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Type, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings

from src.strategies.base_strategy import BaseStrategy
from src.engine.order_types import Order, MarketDataPoint
from src.utils.constants import OrderSide, OrderType
from src.utils.logger import get_logger, log_performance
from src.performance.metrics import PerformanceMetrics


class StrategyType(Enum):
    """Types of trading strategies"""
    MARKET_MAKING = "market_making"
    LIQUIDITY_TAKING = "liquidity_taking"
    ARBITRAGE = "arbitrage"
    MACHINE_LEARNING = "machine_learning"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    CUSTOM = "custom"


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    performance_targets: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Represents a trading signal"""
    signal_type: SignalType
    symbol: str
    confidence: float
    price: Optional[float] = None
    volume: Optional[int] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyComponent(ABC):
    """
    Abstract base class for strategy components
    
    Strategy components are modular pieces that can be combined
    to create complex trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, market_data: MarketDataPoint, context: Dict[str, Any]) -> Any:
        """Process market data and return component output"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get component information"""
        pass


class SignalGenerator(StrategyComponent):
    """Base class for signal generation components"""
    
    @abstractmethod
    def generate_signals(self, market_data: MarketDataPoint, 
                        context: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals"""
        pass
    
    def process(self, market_data: MarketDataPoint, context: Dict[str, Any]) -> List[TradingSignal]:
        return self.generate_signals(market_data, context)


class RiskManager(StrategyComponent):
    """Base class for risk management components"""
    
    @abstractmethod
    def check_risk(self, signals: List[TradingSignal], 
                   context: Dict[str, Any]) -> List[TradingSignal]:
        """Filter signals based on risk criteria"""
        pass
    
    def process(self, signals: List[TradingSignal], context: Dict[str, Any]) -> List[TradingSignal]:
        return self.check_risk(signals, context)


class OrderManager(StrategyComponent):
    """Base class for order management components"""
    
    @abstractmethod
    def create_orders(self, signals: List[TradingSignal], 
                     context: Dict[str, Any]) -> List[Order]:
        """Convert signals to orders"""
        pass
    
    def process(self, signals: List[TradingSignal], context: Dict[str, Any]) -> List[Order]:
        return self.create_orders(signals, context)


class MovingAverageSignalGenerator(SignalGenerator):
    """Simple moving average signal generator"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.short_window = config.get('short_window', 10)
        self.long_window = config.get('long_window', 30)
        self.price_history = []
    
    def generate_signals(self, market_data: MarketDataPoint, 
                        context: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals based on moving average crossover"""
        self.price_history.append(market_data.price)
        
        # Maintain history size
        max_history = self.long_window * 2
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
        
        if len(self.price_history) < self.long_window:
            return []
        
        # Calculate moving averages
        short_ma = np.mean(self.price_history[-self.short_window:])
        long_ma = np.mean(self.price_history[-self.long_window:])
        
        # Previous moving averages
        if len(self.price_history) > self.long_window:
            prev_short_ma = np.mean(self.price_history[-self.short_window-1:-1])
            prev_long_ma = np.mean(self.price_history[-self.long_window-1:-1])
        else:
            return []
        
        signals = []
        
        # Crossover signals
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            # Bullish crossover
            signals.append(TradingSignal(
                signal_type=SignalType.BUY,
                symbol=market_data.symbol,
                confidence=0.7,
                price=market_data.price,
                timestamp=market_data.timestamp,
                metadata={'short_ma': short_ma, 'long_ma': long_ma}
            ))
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            # Bearish crossover
            signals.append(TradingSignal(
                signal_type=SignalType.SELL,
                symbol=market_data.symbol,
                confidence=0.7,
                price=market_data.price,
                timestamp=market_data.timestamp,
                metadata={'short_ma': short_ma, 'long_ma': long_ma}
            ))
        
        return signals
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'component_type': 'MovingAverageSignalGenerator',
            'short_window': self.short_window,
            'long_window': self.long_window,
            'history_length': len(self.price_history)
        }


class PositionSizeRiskManager(RiskManager):
    """Risk manager that controls position sizes"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_position_size = config.get('max_position_size', 1000)
        self.max_total_exposure = config.get('max_total_exposure', 10000)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)  # 2% of capital
    
    def check_risk(self, signals: List[TradingSignal], 
                   context: Dict[str, Any]) -> List[TradingSignal]:
        """Filter signals based on position size limits"""
        filtered_signals = []
        current_position = context.get('current_position', 0)
        capital = context.get('capital', 100000)
        
        for signal in signals:
            # Calculate position size based on risk
            risk_amount = capital * self.risk_per_trade
            
            if signal.price and signal.price > 0:
                max_volume = int(risk_amount / signal.price)
                max_volume = min(max_volume, self.max_position_size)
                
                # Check total exposure
                new_position = current_position
                if signal.signal_type == SignalType.BUY:
                    new_position += max_volume
                elif signal.signal_type == SignalType.SELL:
                    new_position -= max_volume
                
                if abs(new_position) <= self.max_total_exposure:
                    # Update signal with calculated volume
                    signal.volume = max_volume
                    filtered_signals.append(signal)
                else:
                    self.logger.warning(f"Signal rejected due to exposure limit: {signal}")
            else:
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'component_type': 'PositionSizeRiskManager',
            'max_position_size': self.max_position_size,
            'max_total_exposure': self.max_total_exposure,
            'risk_per_trade': self.risk_per_trade
        }


class LimitOrderManager(OrderManager):
    """Order manager that creates limit orders"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.price_offset = config.get('price_offset', 0.01)  # Offset from market price
        self.default_volume = config.get('default_volume', 100)
    
    def create_orders(self, signals: List[TradingSignal], 
                     context: Dict[str, Any]) -> List[Order]:
        """Convert signals to limit orders"""
        orders = []
        
        for signal in signals:
            if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                # Determine order side
                side = OrderSide.BID if signal.signal_type == SignalType.BUY else OrderSide.ASK
                
                # Calculate order price with offset
                if signal.price:
                    if signal.signal_type == SignalType.BUY:
                        order_price = signal.price - self.price_offset
                    else:
                        order_price = signal.price + self.price_offset
                else:
                    order_price = 100.0  # Default price
                
                # Use signal volume or default
                volume = signal.volume or self.default_volume
                
                # Create order
                order = Order.create_limit_order(
                    symbol=signal.symbol,
                    side=side,
                    volume=volume,
                    price=order_price,
                    metadata={
                        'strategy_component': 'LimitOrderManager',
                        'signal_confidence': signal.confidence,
                        'signal_metadata': signal.metadata
                    }
                )
                
                orders.append(order)
        
        return orders
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'component_type': 'LimitOrderManager',
            'price_offset': self.price_offset,
            'default_volume': self.default_volume
        }


class ModularStrategy(BaseStrategy):
    """
    Modular strategy that combines multiple components
    
    This strategy allows for flexible composition of different
    signal generators, risk managers, and order managers.
    """
    
    def __init__(self, symbol: str, config: StrategyConfig):
        """
        Initialize modular strategy
        
        Args:
            symbol: Trading symbol
            config: Strategy configuration
        """
        super().__init__(symbol, config.parameters)
        
        self.strategy_config = config
        self.components = {
            'signal_generators': [],
            'risk_managers': [],
            'order_managers': []
        }
        
        # Strategy context
        self.context = {
            'current_position': 0,
            'capital': 100000,
            'trades_today': 0,
            'last_signal_time': None,
            'performance_metrics': {}
        }
        
        # Performance tracking
        self.signals_generated = 0
        self.orders_created = 0
        self.component_performance = {}
        
        self.logger.info(f"ModularStrategy initialized: {config.name}")
    
    def add_component(self, component: StrategyComponent, component_type: str):
        """
        Add a component to the strategy
        
        Args:
            component: Strategy component instance
            component_type: Type of component ('signal_generators', 'risk_managers', 'order_managers')
        """
        if component_type in self.components:
            self.components[component_type].append(component)
            self.logger.info(f"Added {component.__class__.__name__} to {component_type}")
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    @log_performance
    def generate_signals(self, market_data: MarketDataPoint) -> List[Order]:
        """
        Generate trading signals using all components
        
        Args:
            market_data: Current market data
            
        Returns:
            List of orders to execute
        """
        # Update context
        self.context['last_signal_time'] = market_data.timestamp
        
        # Step 1: Generate signals
        all_signals = []
        for generator in self.components['signal_generators']:
            try:
                signals = generator.generate_signals(market_data, self.context)
                all_signals.extend(signals)
                self.signals_generated += len(signals)
            except Exception as e:
                self.logger.error(f"Error in signal generator {generator.__class__.__name__}: {e}")
        
        if not all_signals:
            return []
        
        # Step 2: Apply risk management
        filtered_signals = all_signals
        for risk_manager in self.components['risk_managers']:
            try:
                filtered_signals = risk_manager.check_risk(filtered_signals, self.context)
            except Exception as e:
                self.logger.error(f"Error in risk manager {risk_manager.__class__.__name__}: {e}")
        
        # Step 3: Create orders
        all_orders = []
        for order_manager in self.components['order_managers']:
            try:
                orders = order_manager.create_orders(filtered_signals, self.context)
                all_orders.extend(orders)
                self.orders_created += len(orders)
            except Exception as e:
                self.logger.error(f"Error in order manager {order_manager.__class__.__name__}: {e}")
        
        return all_orders
    
    def update_context(self, key: str, value: Any):
        """Update strategy context"""
        self.context[key] = value
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        base_info = super().get_strategy_info()
        
        modular_info = {
            'strategy_config': {
                'name': self.strategy_config.name,
                'type': self.strategy_config.strategy_type.value,
                'parameters': self.strategy_config.parameters,
                'risk_limits': self.strategy_config.risk_limits,
            },
            'components': {
                'signal_generators': [comp.get_info() for comp in self.components['signal_generators']],
                'risk_managers': [comp.get_info() for comp in self.components['risk_managers']],
                'order_managers': [comp.get_info() for comp in self.components['order_managers']],
            },
            'context': self.context.copy(),
            'performance': {
                'signals_generated': self.signals_generated,
                'orders_created': self.orders_created,
                'component_performance': self.component_performance,
            }
        }
        
        base_info.update(modular_info)
        return base_info


class StrategyFactory:
    """
    Factory for creating and managing trading strategies
    
    This factory provides methods to create strategies from configurations,
    load strategies from files, and manage strategy lifecycles.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.registered_components = {
            'signal_generators': {},
            'risk_managers': {},
            'order_managers': {}
        }
        
        # Register built-in components
        self._register_builtin_components()
    
    def _register_builtin_components(self):
        """Register built-in strategy components"""
        self.register_component('MovingAverageSignalGenerator', MovingAverageSignalGenerator, 'signal_generators')
        self.register_component('PositionSizeRiskManager', PositionSizeRiskManager, 'risk_managers')
        self.register_component('LimitOrderManager', LimitOrderManager, 'order_managers')
    
    def register_component(self, name: str, component_class: Type[StrategyComponent], 
                          component_type: str):
        """
        Register a strategy component
        
        Args:
            name: Component name
            component_class: Component class
            component_type: Type of component
        """
        if component_type in self.registered_components:
            self.registered_components[component_type][name] = component_class
            self.logger.info(f"Registered component: {name} ({component_type})")
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def create_strategy(self, symbol: str, config: StrategyConfig) -> ModularStrategy:
        """
        Create a strategy from configuration
        
        Args:
            symbol: Trading symbol
            config: Strategy configuration
            
        Returns:
            Configured strategy instance
        """
        strategy = ModularStrategy(symbol, config)
        
        # Add components based on configuration
        components_config = config.parameters.get('components', {})
        
        for component_type, component_configs in components_config.items():
            if component_type in self.registered_components:
                for component_config in component_configs:
                    component_name = component_config.get('name')
                    component_params = component_config.get('parameters', {})
                    
                    if component_name in self.registered_components[component_type]:
                        component_class = self.registered_components[component_type][component_name]
                        component = component_class(component_params)
                        strategy.add_component(component, component_type)
                    else:
                        self.logger.warning(f"Unknown component: {component_name}")
        
        return strategy
    
    def load_strategy_from_file(self, symbol: str, config_file: str) -> ModularStrategy:
        """
        Load strategy from configuration file
        
        Args:
            symbol: Trading symbol
            config_file: Path to configuration file
            
        Returns:
            Configured strategy instance
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Load configuration
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Create strategy configuration
        strategy_config = StrategyConfig(
            name=config_data.get('name', 'unnamed_strategy'),
            strategy_type=StrategyType(config_data.get('type', 'custom')),
            parameters=config_data.get('parameters', {}),
            risk_limits=config_data.get('risk_limits', {}),
            performance_targets=config_data.get('performance_targets', {}),
            metadata=config_data.get('metadata', {})
        )
        
        return self.create_strategy(symbol, strategy_config)
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """Get list of available components"""
        return {
            component_type: list(components.keys())
            for component_type, components in self.registered_components.items()
        }


class StrategyTester:
    """
    Comprehensive strategy testing framework
    
    This class provides tools for backtesting, walk-forward analysis,
    and performance evaluation of trading strategies.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def backtest_strategy(self, strategy: BaseStrategy, 
                         market_data: pd.DataFrame,
                         initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Backtest a strategy with historical data
        
        Args:
            strategy: Strategy to test
            market_data: Historical market data
            initial_capital: Starting capital
            
        Returns:
            Backtest results
        """
        results = {
            'trades': [],
            'signals': [],
            'performance_metrics': {},
            'strategy_info': strategy.get_strategy_info()
        }
        
        capital = initial_capital
        position = 0
        
        for _, row in market_data.iterrows():
            # Create market data point
            market_point = MarketDataPoint(
                timestamp=row['timestamp'],
                symbol=strategy.symbol,
                price=row['price'],
                volume=row.get('volume', 1000)
            )
            
            # Generate signals
            orders = strategy.generate_signals(market_point)
            
            # Execute orders (simplified)
            for order in orders:
                if order.side == OrderSide.BID and position < 10000:  # Max position limit
                    position += order.volume
                    capital -= order.volume * order.price
                    
                    results['trades'].append({
                        'timestamp': row['timestamp'],
                        'side': 'buy',
                        'volume': order.volume,
                        'price': order.price,
                        'position': position,
                        'capital': capital
                    })
                    
                elif order.side == OrderSide.ASK and position > -10000:  # Max position limit
                    position -= order.volume
                    capital += order.volume * order.price
                    
                    results['trades'].append({
                        'timestamp': row['timestamp'],
                        'side': 'sell',
                        'volume': order.volume,
                        'price': order.price,
                        'position': position,
                        'capital': capital
                    })
        
        # Calculate performance metrics
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            final_value = capital + position * market_data['price'].iloc[-1]
            
            results['performance_metrics'] = {
                'total_return': ((final_value - initial_capital) / initial_capital) * 100,
                'num_trades': len(results['trades']),
                'final_position': position,
                'final_capital': capital,
                'final_value': final_value
            }
        
        return results
    
    def walk_forward_analysis(self, strategy_factory: StrategyFactory,
                             config: StrategyConfig,
                             market_data: pd.DataFrame,
                             train_window: int = 252,
                             test_window: int = 63) -> Dict[str, Any]:
        """
        Perform walk-forward analysis
        
        Args:
            strategy_factory: Factory for creating strategies
            config: Strategy configuration
            market_data: Historical market data
            train_window: Training window size
            test_window: Testing window size
            
        Returns:
            Walk-forward analysis results
        """
        results = {
            'periods': [],
            'performance_summary': {},
            'strategy_evolution': []
        }
        
        total_periods = (len(market_data) - train_window) // test_window
        
        for period in range(total_periods):
            start_idx = period * test_window
            train_end_idx = start_idx + train_window
            test_end_idx = min(train_end_idx + test_window, len(market_data))
            
            # Training data
            train_data = market_data.iloc[start_idx:train_end_idx]
            
            # Testing data
            test_data = market_data.iloc[train_end_idx:test_end_idx]
            
            if len(test_data) == 0:
                break
            
            # Create strategy for this period
            strategy = strategy_factory.create_strategy(config.name, config)
            
            # Backtest on test data
            period_results = self.backtest_strategy(strategy, test_data)
            
            results['periods'].append({
                'period': period,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'performance': period_results['performance_metrics'],
                'num_trades': len(period_results['trades'])
            })
            
            results['strategy_evolution'].append(strategy.get_strategy_info())
        
        # Calculate summary statistics
        if results['periods']:
            returns = [p['performance'].get('total_return', 0) for p in results['periods']]
            results['performance_summary'] = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'total_periods': len(results['periods'])
            }
        
        return results


# Example strategy configuration
def create_example_strategy_config() -> StrategyConfig:
    """Create an example strategy configuration"""
    return StrategyConfig(
        name="MovingAverageCrossover",
        strategy_type=StrategyType.MOMENTUM,
        parameters={
            'components': {
                'signal_generators': [
                    {
                        'name': 'MovingAverageSignalGenerator',
                        'parameters': {
                            'short_window': 10,
                            'long_window': 30
                        }
                    }
                ],
                'risk_managers': [
                    {
                        'name': 'PositionSizeRiskManager',
                        'parameters': {
                            'max_position_size': 1000,
                            'max_total_exposure': 5000,
                            'risk_per_trade': 0.02
                        }
                    }
                ],
                'order_managers': [
                    {
                        'name': 'LimitOrderManager',
                        'parameters': {
                            'price_offset': 0.01,
                            'default_volume': 100
                        }
                    }
                ]
            }
        },
        risk_limits={
            'max_drawdown': 0.1,
            'max_daily_loss': 0.05
        },
        performance_targets={
            'min_sharpe_ratio': 1.0,
            'min_win_rate': 0.55
        }
    )


# Utility functions
def save_strategy_config(config: StrategyConfig, filepath: str):
    """Save strategy configuration to file"""
    config_dict = {
        'name': config.name,
        'type': config.strategy_type.value,
        'parameters': config.parameters,
        'risk_limits': config.risk_limits,
        'performance_targets': config.performance_targets,
        'metadata': config.metadata
    }
    
    path = Path(filepath)
    if path.suffix.lower() in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif path.suffix.lower() == '.json':
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")