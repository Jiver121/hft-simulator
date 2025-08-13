"""
Configuration module for HFT Simulator

This module provides centralized configuration management for all components
of the HFT Order Book Simulator.
"""

from .settings import (
    config,
    get_config,
    load_config,
    ConfigManager,
    DataConfig,
    OrderBookConfig,
    ExecutionConfig,
    StrategyConfig,
    MarketMakingConfig,
    LiquidityTakingConfig,
    PerformanceConfig,
    VisualizationConfig,
    LoggingConfig,
    PROJECT_ROOT,
    DATA_DIR,
    RESULTS_DIR,
    LOGS_DIR,
)

# Import new configuration classes
from .backtest_config import BacktestConfig, load_backtest_config
from .strategy_config import (
    BaseStrategyConfig,
    MarketMakingStrategyConfig,
    MomentumStrategyConfig,
    LiquidityTakingStrategyConfig,
    StrategyConfigManager,
    load_strategy_config,
    strategy_config_manager
)

__all__ = [
    'config',
    'get_config',
    'load_config',
    'ConfigManager',
    'DataConfig',
    'OrderBookConfig',
    'ExecutionConfig',
    'StrategyConfig',
    'MarketMakingConfig',
    'LiquidityTakingConfig',
    'PerformanceConfig',
    'VisualizationConfig',
    'LoggingConfig',
    'PROJECT_ROOT',
    'DATA_DIR',
    'RESULTS_DIR',
    'LOGS_DIR',
    # New configuration classes
    'BacktestConfig',
    'load_backtest_config',
    'BaseStrategyConfig',
    'MarketMakingStrategyConfig',
    'MomentumStrategyConfig',
    'LiquidityTakingStrategyConfig',
    'StrategyConfigManager',
    'load_strategy_config',
    'strategy_config_manager'
]
