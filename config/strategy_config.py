"""
Strategy Configuration Module

This module provides configuration classes for different trading strategies
including market making, momentum, and liquidity taking strategies.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseStrategyConfig:
    """Base class for all strategy configurations."""
    
    name: str = "base_strategy"
    enabled: bool = True
    description: str = "Base strategy configuration"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, json_path: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent)
        
        logger.info(f"Saved {self.name} configuration to {json_path}")


@dataclass
class MarketMakingStrategyConfig(BaseStrategyConfig):
    """Configuration for market making strategies."""
    
    name: str = "market_making"
    description: str = "Market making strategy configuration"
    
    # Spread management
    target_spread_bps: float = 10.0
    min_spread_bps: float = 5.0
    max_spread_bps: float = 50.0
    spread_adjustment_factor: float = 0.1
    
    # Order size and inventory
    base_order_size: int = 100
    max_order_size: int = 1000
    max_inventory_abs: int = 1000
    inventory_target: int = 0
    inventory_penalty_factor: float = 0.001
    
    # Risk management
    risk_aversion: float = 0.1
    max_position_size: int = 2000
    position_limit_factor: float = 0.8
    
    # Quote management
    quote_refresh_rate_ms: int = 100
    max_quote_levels: int = 3
    quote_size_increment: int = 50
    
    # Market conditions
    adverse_selection_threshold: float = 0.7
    market_impact_threshold: float = 0.002
    volatility_adjustment: bool = True
    
    # Performance settings
    enable_skewing: bool = True
    skew_factor: float = 0.5
    enable_lean: bool = True
    lean_factor: float = 0.3


@dataclass
class MomentumStrategyConfig(BaseStrategyConfig):
    """Configuration for momentum trading strategies."""
    
    name: str = "momentum"
    description: str = "Momentum strategy configuration"
    
    # Signal parameters
    momentum_threshold: float = 0.002
    signal_decay_factor: float = 0.95
    min_signal_strength: float = 0.3
    max_signal_age_seconds: int = 300
    
    # Lookback and calculation
    lookback_window: int = 100
    price_change_window: int = 20
    volume_confirmation: bool = True
    min_volume_multiple: float = 1.5
    
    # Position sizing
    base_order_size: int = 200
    max_order_size: int = 1000
    max_total_positions: int = 5
    position_size_factor: float = 0.5
    
    # Risk controls
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_holding_time_seconds: int = 3600
    
    # Entry and exit
    entry_aggression: float = 0.7
    exit_aggression: float = 0.9
    partial_exit_threshold: float = 0.5
    
    # Filters
    trend_confirmation: bool = True
    volatility_filter: bool = True
    max_volatility_threshold: float = 0.05
    min_liquidity: int = 1000


@dataclass
class LiquidityTakingStrategyConfig(BaseStrategyConfig):
    """Configuration for liquidity taking strategies."""
    
    name: str = "liquidity_taking"
    description: str = "Liquidity taking strategy configuration"
    
    # Signal thresholds
    signal_threshold: float = 0.001
    min_signal_strength: float = 0.4
    signal_confirmation_window: int = 5
    
    # Execution parameters
    aggression_level: float = 0.6
    max_market_impact: float = 0.005
    participation_rate: float = 0.1
    slice_size: int = 100
    
    # Timing controls
    execution_delay_ms: int = 50
    max_execution_time_ms: int = 5000
    time_between_orders_ms: int = 100
    
    # Liquidity requirements
    min_liquidity_threshold: int = 500
    liquidity_check_levels: int = 3
    book_depth_requirement: int = 1000
    
    # Position management
    max_position_size: int = 1500
    target_fill_rate: float = 0.8
    adaptive_sizing: bool = True
    
    # Risk controls
    adverse_price_move_threshold: float = 0.003
    max_slippage_bps: float = 5.0
    enable_cancel_replace: bool = True


class StrategyConfigManager:
    """Manager class for loading and managing strategy configurations."""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, BaseStrategyConfig] = {}
    
    def load_config(self, strategy_name: str, 
                   config_path: Optional[Union[str, Path]] = None) -> BaseStrategyConfig:
        """
        Load strategy configuration from file.
        
        Args:
            strategy_name: Name of the strategy
            config_path: Optional path to config file
            
        Returns:
            Strategy configuration instance
        """
        if config_path is None:
            config_path = self.config_dir / f"{strategy_name}_config.json"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._create_default_config(strategy_name)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            config = self._create_config_from_data(strategy_name, data)
            self.configs[strategy_name] = config
            
            logger.info(f"Loaded {strategy_name} configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading {strategy_name} config: {e}")
            return self._create_default_config(strategy_name)
    
    def save_config(self, strategy_name: str, config: BaseStrategyConfig,
                   config_path: Optional[Union[str, Path]] = None) -> None:
        """Save strategy configuration to file."""
        if config_path is None:
            config_path = self.config_dir / f"{strategy_name}_config.json"
        
        config.to_json(config_path)
        self.configs[strategy_name] = config
    
    def get_config(self, strategy_name: str) -> BaseStrategyConfig:
        """Get cached configuration or load if not cached."""
        if strategy_name not in self.configs:
            return self.load_config(strategy_name)
        return self.configs[strategy_name]
    
    def create_default_configs(self) -> None:
        """Create all default strategy configuration files."""
        self.config_dir.mkdir(exist_ok=True)
        
        # Market making config
        mm_config = MarketMakingStrategyConfig()
        mm_config.to_json(self.config_dir / "market_making_config.json")
        
        # Momentum config  
        momentum_config = MomentumStrategyConfig()
        momentum_config.to_json(self.config_dir / "momentum_config.json")
        
        # Liquidity taking config
        lt_config = LiquidityTakingStrategyConfig()
        lt_config.to_json(self.config_dir / "liquidity_taking_config.json")
        
        logger.info(f"Created default strategy configurations in {self.config_dir}")
    
    def _create_default_config(self, strategy_name: str) -> BaseStrategyConfig:
        """Create default configuration for a strategy."""
        config_classes = {
            "market_making": MarketMakingStrategyConfig,
            "momentum": MomentumStrategyConfig,
            "liquidity_taking": LiquidityTakingStrategyConfig
        }
        
        config_class = config_classes.get(strategy_name, BaseStrategyConfig)
        return config_class()
    
    def _create_config_from_data(self, strategy_name: str, data: Dict[str, Any]) -> BaseStrategyConfig:
        """Create configuration instance from loaded data."""
        config_classes = {
            "market_making": MarketMakingStrategyConfig,
            "momentum": MomentumStrategyConfig,
            "liquidity_taking": LiquidityTakingStrategyConfig
        }
        
        config_class = config_classes.get(strategy_name, BaseStrategyConfig)
        return config_class(**data)


# Create global strategy config manager instance
strategy_config_manager = StrategyConfigManager()


def load_strategy_config(strategy_name: str, 
                        config_path: Optional[Union[str, Path]] = None) -> BaseStrategyConfig:
    """
    Convenience function to load strategy configuration.
    
    Args:
        strategy_name: Name of the strategy
        config_path: Optional path to config file
        
    Returns:
        Strategy configuration instance
    """
    return strategy_config_manager.load_config(strategy_name, config_path)
