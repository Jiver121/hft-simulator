"""
Global configuration settings for HFT Simulator

This module contains all configuration parameters for the HFT Order Book Simulator.
Settings are organized by component and can be overridden via environment variables
or configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for directory in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data ingestion and processing"""
    
    # Data directories
    raw_data_dir: Path = DATA_DIR / "raw"
    processed_data_dir: Path = DATA_DIR / "processed"
    sample_data_dir: Path = DATA_DIR / "sample"
    
    # Processing parameters
    chunk_size: int = 100_000  # Rows per chunk for memory efficiency
    max_memory_usage: float = 0.8  # Maximum memory usage (80% of available)
    
    # Data validation
    validate_data: bool = True
    drop_invalid_rows: bool = True
    
    # Supported file formats
    supported_formats: list = field(default_factory=lambda: ['.csv', '.parquet', '.h5'])
    
    # Default column mappings for Kaggle HFT datasets
    column_mapping: Dict[str, str] = field(default_factory=lambda: {
        'timestamp': 'timestamp',
        'price': 'price',
        'volume': 'volume',
        'side': 'side',  # 'bid' or 'ask'
        'order_type': 'order_type',  # 'limit', 'market', 'cancel'
        'order_id': 'order_id',
        'symbol': 'symbol'
    })


@dataclass
class OrderBookConfig:
    """Configuration for order book engine"""
    
    # Order book parameters
    max_price_levels: int = 100  # Maximum price levels to maintain
    tick_size: float = 0.01  # Minimum price increment
    
    # Performance settings
    use_numba: bool = True  # Enable Numba JIT compilation
    cache_book_states: bool = True  # Cache order book snapshots
    
    # Validation
    validate_orders: bool = True
    check_price_bounds: bool = True
    
    # Logging
    log_book_updates: bool = False  # Detailed logging (performance impact)


@dataclass
class ExecutionConfig:
    """Configuration for execution simulator"""
    
    # Execution parameters
    market_order_fill_probability: float = 1.0  # Market orders always fill
    limit_order_fill_probability: float = 0.95  # Limit order fill rate
    
    # Latency simulation (microseconds)
    min_latency: int = 10
    max_latency: int = 100
    average_latency: int = 50
    
    # Slippage modeling
    enable_slippage: bool = True
    slippage_factor: float = 0.0001  # 1 basis point
    
    # Partial fills
    enable_partial_fills: bool = True
    min_fill_size: int = 1
    
    # Order rejection
    rejection_rate: float = 0.01  # 1% order rejection rate


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""
    
    # Risk management
    max_position_size: int = 1000
    max_daily_loss: float = 10000.0
    max_drawdown: float = 0.05  # 5%
    
    # Position sizing
    default_order_size: int = 100
    min_order_size: int = 1
    max_order_size: int = 1000
    
    # Timing
    strategy_frequency: str = "1ms"  # Strategy execution frequency
    
    # Logging
    log_strategy_decisions: bool = True


@dataclass
class MarketMakingConfig(StrategyConfig):
    """Configuration for market making strategy"""
    
    # Spread management
    min_spread: float = 0.01  # Minimum bid-ask spread
    target_spread: float = 0.02  # Target spread
    max_spread: float = 0.05  # Maximum spread before stopping
    
    # Inventory management
    max_inventory: int = 500  # Maximum inventory position
    inventory_target: int = 0  # Target inventory (neutral)
    inventory_penalty: float = 0.001  # Penalty per unit of inventory
    
    # Quote management
    quote_size: int = 100  # Default quote size
    max_quote_levels: int = 3  # Number of price levels to quote
    quote_refresh_rate: str = "100ms"  # How often to refresh quotes
    
    # Risk controls
    adverse_selection_threshold: float = 0.1  # Stop if adverse selection > 10%
    max_quote_imbalance: float = 0.3  # Maximum order book imbalance


@dataclass
class LiquidityTakingConfig(StrategyConfig):
    """Configuration for liquidity taking strategy"""
    
    # Signal parameters
    signal_threshold: float = 0.001  # Minimum signal strength
    signal_decay: float = 0.95  # Signal decay factor
    
    # Execution parameters
    aggression_level: float = 0.5  # 0 = passive, 1 = aggressive
    max_market_impact: float = 0.002  # Maximum acceptable market impact
    
    # Timing
    execution_delay: str = "10ms"  # Delay before execution
    max_execution_time: str = "1s"  # Maximum time to complete order
    
    # Size management
    participation_rate: float = 0.1  # Maximum % of volume to take
    min_liquidity_threshold: int = 1000  # Minimum liquidity required


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking"""
    
    # Metrics calculation
    calculate_real_time: bool = True
    update_frequency: str = "1s"
    
    # Risk metrics
    var_confidence: float = 0.95  # VaR confidence level
    var_window: int = 252  # Trading days for VaR calculation
    
    # Benchmark
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Reporting
    generate_daily_reports: bool = True
    save_trade_log: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for visualization and reporting"""
    
    # Plot settings
    figure_size: tuple = (12, 8)
    dpi: int = 100
    style: str = "seaborn-v0_8"
    
    # Interactive plots
    enable_interactive: bool = True
    auto_refresh: bool = True
    refresh_rate: str = "1s"
    
    # Export settings
    export_format: str = "png"
    export_dpi: int = 300
    
    # Dashboard
    enable_dashboard: bool = True
    dashboard_port: int = 8050


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    
    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Log files
    log_dir: Path = LOGS_DIR
    log_file: str = "hft_simulator.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    
    # Performance logging
    log_performance: bool = True
    log_memory_usage: bool = True
    
    # Strategy logging
    log_orders: bool = True
    log_fills: bool = True
    log_pnl: bool = True


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        
        # Default configurations
        self.data = DataConfig()
        self.order_book = OrderBookConfig()
        self.execution = ExecutionConfig()
        self.strategy = StrategyConfig()
        self.market_making = MarketMakingConfig()
        self.liquidity_taking = LiquidityTakingConfig()
        self.performance = PerformanceConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()
        
        # Load from file if specified
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                self._update_from_dict(file_config)
        
        # Override with environment variables
        self._load_from_env()
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Example environment variable mappings
        env_mappings = {
            'HFT_CHUNK_SIZE': ('data', 'chunk_size', int),
            'HFT_MAX_POSITION': ('strategy', 'max_position_size', int),
            'HFT_LOG_LEVEL': ('logging', 'console_level', str),
            'HFT_ENABLE_NUMBA': ('order_book', 'use_numba', bool),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, section)
                if type_func == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = type_func(value)
                setattr(config_obj, key, value)
    
    def save_config(self, filename: str):
        """Save current configuration to file"""
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['config_file', 'save_config']:
                attr = getattr(self, attr_name)
                if hasattr(attr, '__dict__'):
                    config_dict[attr_name] = attr.__dict__
        
        with open(filename, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        
        # Add validation logic here
        errors = []
        
        # Validate data config
        if self.data.chunk_size <= 0:
            errors.append("Data chunk_size must be positive")
        
        # Validate order book config
        if self.order_book.tick_size <= 0:
            errors.append("Order book tick_size must be positive")
        
        # Validate strategy config
        if self.strategy.max_position_size <= 0:
            errors.append("Strategy max_position_size must be positive")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True


# Global configuration instance
config = ConfigManager()

# Convenience functions
def get_config() -> ConfigManager:
    """Get the global configuration instance"""
    return config

def load_config(config_file: str) -> ConfigManager:
    """Load configuration from file"""
    return ConfigManager(config_file)