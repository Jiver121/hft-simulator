"""
Backtesting Configuration Module

This module provides the BacktestConfig class for loading and managing
backtest configuration parameters from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration class for backtesting parameters.
    
    This class loads configuration from JSON files and provides
    easy access to all backtesting parameters including strategy
    configurations, risk limits, and execution settings.
    """
    
    # Core backtesting parameters
    strategy_type: str = "market_making"
    initial_capital: float = 100000.0
    commission_rate: float = 0.0005
    slippage_bps: float = 1.0
    max_position_size: int = 1000
    max_order_size: int = 100
    risk_limit: float = 10000.0
    tick_size: float = 0.01
    fill_model: str = "realistic"
    enable_logging: bool = True
    save_snapshots: bool = False
    parallel_workers: int = 1
    benchmark_symbol: str = "SPY"
    
    # Strategy-specific parameters
    strategy_params: Dict[str, Dict[str, Any]] = None
    
    # Risk management parameters
    max_daily_loss: Optional[float] = None
    max_drawdown: Optional[float] = None
    position_limit_pct: Optional[float] = None
    
    # Execution parameters
    latency_model: str = "fixed"
    min_latency_ms: float = 1.0
    max_latency_ms: float = 5.0
    market_impact_model: str = "linear"
    
    # Data parameters
    data_frequency: str = "tick"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbols: Optional[list] = None
    
    def __post_init__(self):
        """Initialize default strategy parameters if not provided."""
        if self.strategy_params is None:
            self.strategy_params = {
                "market_making": {
                    "spread_bps": 10.0,
                    "order_size": 100,
                    "max_inventory": 500,
                    "inventory_target": 0,
                    "risk_aversion": 0.1,
                    "min_spread": 0.01,
                    "max_spread": 0.05,
                    "quote_size": 100
                },
                "momentum": {
                    "momentum_threshold": 0.001,
                    "order_size": 200,
                    "max_positions": 5,
                    "lookback_window": 100,
                    "signal_decay": 0.95,
                    "min_signal_strength": 0.3
                },
                "liquidity_taking": {
                    "signal_threshold": 0.002,
                    "max_position": 1000,
                    "participation_rate": 0.1,
                    "aggression_level": 0.5,
                    "min_liquidity": 500
                }
            }
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'BacktestConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Returns:
            BacktestConfig instance with loaded parameters
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded configuration from {json_path}")
            return cls(**data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON configuration file {json_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration from {json_path}: {e}")
            raise
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            BacktestConfig instance
        """
        return cls(**config_dict)
    
    def to_json(self, json_path: Union[str, Path], indent: int = 2) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            json_path: Path where to save the JSON file
            indent: JSON indentation level
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=indent)
        
        logger.info(f"Saved configuration to {json_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def get_strategy_config(self, strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_type: Strategy type to get config for. If None, uses self.strategy_type
            
        Returns:
            Strategy configuration dictionary
            
        Raises:
            KeyError: If strategy type is not found in strategy_params
        """
        strategy = strategy_type or self.strategy_type
        
        if strategy not in self.strategy_params:
            raise KeyError(f"Strategy '{strategy}' not found in configuration")
        
        return self.strategy_params[strategy]
    
    def update_strategy_param(self, param_name: str, value: Any, 
                            strategy_type: Optional[str] = None) -> None:
        """
        Update a specific strategy parameter.
        
        Args:
            param_name: Parameter name to update
            value: New parameter value
            strategy_type: Strategy type to update. If None, uses self.strategy_type
        """
        strategy = strategy_type or self.strategy_type
        
        if strategy not in self.strategy_params:
            self.strategy_params[strategy] = {}
        
        self.strategy_params[strategy][param_name] = value
        logger.info(f"Updated {strategy}.{param_name} = {value}")
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate basic parameters
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        
        if self.commission_rate < 0:
            raise ValueError("commission_rate must be non-negative")
        
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        
        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        
        if self.max_order_size <= 0:
            raise ValueError("max_order_size must be positive")
        
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")
        
        # Validate strategy type
        if self.strategy_type not in self.strategy_params:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")
        
        # Validate fill model
        valid_fill_models = ["realistic", "immediate", "pessimistic", "optimistic"]
        if self.fill_model not in valid_fill_models:
            raise ValueError(f"fill_model must be one of: {valid_fill_models}")
        
        # Validate latency model
        valid_latency_models = ["fixed", "uniform", "normal", "exponential"]
        if self.latency_model not in valid_latency_models:
            raise ValueError(f"latency_model must be one of: {valid_latency_models}")
        
        # Validate market impact model
        valid_impact_models = ["none", "linear", "square_root", "log"]
        if self.market_impact_model not in valid_impact_models:
            raise ValueError(f"market_impact_model must be one of: {valid_impact_models}")
        
        logger.info("Configuration validation passed")
        return True
    
    def create_default_configs(self, config_dir: Union[str, Path] = "config") -> None:
        """
        Create default configuration files for different strategies.
        
        Args:
            config_dir: Directory where to create configuration files
        """
        config_dir = Path(config_dir)
        config_dir.mkdir(exist_ok=True)
        
        # Create default backtest config
        self.to_json(config_dir / "backtest_config.json")
        
        # Create market making specific config
        mm_config = BacktestConfig(
            strategy_type="market_making",
            initial_capital=100000.0,
            commission_rate=0.0005,
            slippage_bps=1.0,
            max_position_size=1000,
            max_order_size=100,
            strategy_params={
                "market_making": {
                    "spread_bps": 5.0,
                    "order_size": 100,
                    "max_inventory": 500,
                    "inventory_target": 0,
                    "risk_aversion": 0.1,
                    "min_spread": 0.01,
                    "max_spread": 0.03,
                    "quote_size": 100,
                    "quote_refresh_rate": 100
                }
            }
        )
        mm_config.to_json(config_dir / "market_making_config.json")
        
        # Create momentum strategy config
        momentum_config = BacktestConfig(
            strategy_type="momentum",
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_bps=2.0,
            max_position_size=2000,
            max_order_size=500,
            strategy_params={
                "momentum": {
                    "momentum_threshold": 0.002,
                    "order_size": 300,
                    "max_positions": 10,
                    "lookback_window": 50,
                    "signal_decay": 0.9,
                    "min_signal_strength": 0.4,
                    "stop_loss": 0.02,
                    "take_profit": 0.05
                }
            }
        )
        momentum_config.to_json(config_dir / "momentum_config.json")
        
        logger.info(f"Created default configuration files in {config_dir}")


def load_backtest_config(config_path: Optional[Union[str, Path]] = None) -> BacktestConfig:
    """
    Load backtest configuration from file or create default.
    
    Args:
        config_path: Path to configuration file. If None, uses default location.
        
    Returns:
        BacktestConfig instance
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("config/backtest_config.json"),
            Path("backtest_config.json"),
            Path("../config/backtest_config.json")
        ]
        
        for path in default_paths:
            if path.exists():
                config_path = path
                break
    
    if config_path is None or not Path(config_path).exists():
        logger.warning("No configuration file found, using default configuration")
        return BacktestConfig()
    
    return BacktestConfig.from_json(config_path)
