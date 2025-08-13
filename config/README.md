# HFT Simulator Configuration System

This directory contains the configuration system for the HFT Simulator backtesting framework. The configuration system provides a flexible and comprehensive way to manage all parameters for backtesting and trading strategies.

## 📁 Configuration Files

### Core Configuration Files

- **`backtest_config.json`** - Main backtesting configuration with default parameters
- **`strategy_config.json`** - Comprehensive strategy configurations for all strategy types
- **`strategy_configs.yaml`** - Legacy YAML configuration file (maintained for compatibility)

### Strategy Template Files

- **`market_making_template.json`** - Conservative market making strategy template
- **`momentum_template.json`** - Momentum trading strategy template

### Python Configuration Modules

- **`backtest_config.py`** - `BacktestConfig` class for loading backtest configurations
- **`strategy_config.py`** - Strategy-specific configuration classes
- **`settings.py`** - Legacy configuration settings (maintained for compatibility)
- **`__init__.py`** - Configuration module exports

## 🚀 Quick Start

### Loading Backtest Configuration

```python
from config import BacktestConfig, load_backtest_config

# Method 1: Load from default location
config = load_backtest_config()

# Method 2: Load from specific file
config = BacktestConfig.from_json("config/backtest_config.json")

# Method 3: Create with custom parameters
config = BacktestConfig(
    strategy_type="momentum",
    initial_capital=200000.0,
    commission_rate=0.001
)

# Validate configuration
config.validate()

# Get strategy-specific parameters
mm_params = config.get_strategy_config("market_making")
```

### Loading Strategy Configuration

```python
from config import (
    MarketMakingStrategyConfig,
    MomentumStrategyConfig, 
    load_strategy_config
)

# Load strategy config
mm_config = load_strategy_config("market_making")

# Create default config
momentum_config = MomentumStrategyConfig()

# Convert to dictionary
config_dict = momentum_config.to_dict()
```

## 📊 Configuration Structure

### Backtest Configuration Parameters

#### Core Parameters
- `strategy_type`: Strategy to use ("market_making", "momentum", "liquidity_taking")
- `initial_capital`: Starting capital amount
- `commission_rate`: Trading commission rate (fraction)
- `slippage_bps`: Average slippage in basis points
- `max_position_size`: Maximum position size
- `max_order_size`: Maximum single order size
- `risk_limit`: Risk limit amount
- `tick_size`: Minimum price increment

#### Risk Management Parameters
- `max_daily_loss`: Maximum daily loss limit
- `max_drawdown`: Maximum portfolio drawdown (fraction)
- `position_limit_pct`: Position limit as percentage of capital

#### Execution Parameters
- `latency_model`: Latency simulation model ("fixed", "uniform", "normal")
- `min_latency_ms`: Minimum latency in milliseconds
- `max_latency_ms`: Maximum latency in milliseconds
- `market_impact_model`: Market impact model ("linear", "square_root", "log")

#### Data Parameters
- `data_frequency`: Data frequency ("tick", "1s", "1m")
- `start_date`: Backtest start date
- `end_date`: Backtest end date
- `symbols`: List of symbols to trade

### Strategy-Specific Parameters

#### Market Making Strategy
- **Spread Management**: Target spread, min/max spreads, dynamic adjustment
- **Order Management**: Order sizes, refresh rates, quote levels
- **Inventory Management**: Inventory limits, targets, penalty factors
- **Risk Controls**: Risk aversion, position limits, adverse selection thresholds

#### Momentum Strategy
- **Signal Detection**: Momentum thresholds, signal decay, trend confirmation
- **Lookback Parameters**: Windows for price changes and volume analysis
- **Position Sizing**: Base sizes, scaling factors, adaptive sizing
- **Risk Management**: Stop loss, take profit, trailing stops

#### Liquidity Taking Strategy
- **Signal Processing**: Signal thresholds, confirmation windows
- **Execution Parameters**: Aggression levels, participation rates
- **Timing Controls**: Execution delays, timeouts
- **Liquidity Requirements**: Minimum liquidity, book depth requirements

## 🔧 Configuration Management

### Creating Default Configurations

```python
# Create all default configuration files
from config import BacktestConfig, StrategyConfigManager

# Create default backtest configs
config = BacktestConfig()
config.create_default_configs("config")

# Create strategy-specific configs
strategy_manager = StrategyConfigManager()
strategy_manager.create_default_configs()
```

### Customizing Configurations

```python
# Load and modify configuration
config = BacktestConfig.from_json("config/backtest_config.json")

# Update strategy parameters
config.update_strategy_param("spread_bps", 15.0, "market_making")
config.update_strategy_param("momentum_threshold", 0.003, "momentum")

# Save modified configuration
config.to_json("config/my_custom_config.json")
```

### Validation

All configuration classes include validation methods:

```python
# Validate configuration
try:
    config.validate()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 📝 Configuration Templates

### Market Making Template

Use `market_making_template.json` for conservative market making strategies:

- Tight risk controls
- Conservative position sizing
- Dynamic spread adjustment
- Inventory management with lean

### Momentum Template

Use `momentum_template.json` for trend-following strategies:

- Technical indicators (RSI, MACD)
- Multi-timeframe analysis
- Risk management with stops
- Volatility and liquidity filters

## 🛠️ Advanced Usage

### Custom Strategy Configuration

```python
from config.strategy_config import BaseStrategyConfig
from dataclasses import dataclass

@dataclass
class CustomStrategyConfig(BaseStrategyConfig):
    name: str = "custom_strategy"
    custom_param1: float = 0.5
    custom_param2: int = 100
    
    def validate_custom(self):
        if self.custom_param1 <= 0:
            raise ValueError("custom_param1 must be positive")

# Use custom configuration
custom_config = CustomStrategyConfig(custom_param1=0.8)
```

### Configuration from Environment

```python
import os
from config import BacktestConfig

# Load from environment variables
config = BacktestConfig(
    initial_capital=float(os.getenv("INITIAL_CAPITAL", "100000")),
    commission_rate=float(os.getenv("COMMISSION_RATE", "0.0005")),
    strategy_type=os.getenv("STRATEGY_TYPE", "market_making")
)
```

## 🧪 Testing Configuration

Test your configuration setup:

```bash
cd config
python test_config_loading.py
```

This will test:
- BacktestConfig loading and validation
- Strategy configuration creation
- JSON file loading
- Parameter access and modification

## 📚 Configuration Reference

### File Formats

The configuration system supports:
- **JSON**: Primary format for configuration files
- **YAML**: Legacy format (strategy_configs.yaml)
- **Python**: Direct instantiation of configuration classes

### Configuration Precedence

1. Explicitly provided configuration files
2. Default configuration files in `config/` directory
3. Built-in default values in configuration classes

### Environment Integration

Configuration classes are designed to work with:
- Command-line argument parsing
- Environment variable override
- Docker containerization
- Kubernetes ConfigMaps

## 🔄 Migration Guide

### From Legacy YAML Configuration

```python
# Old way (YAML)
import yaml
with open("strategy_configs.yaml") as f:
    config = yaml.safe_load(f)

# New way (JSON + Classes)
from config import BacktestConfig
config = BacktestConfig.from_json("backtest_config.json")
```

### Updating Existing Code

```python
# Replace direct dictionary access
# Old: config["strategy_params"]["market_making"]["spread_bps"]
# New: config.get_strategy_config("market_making")["spread_bps"]

# Use configuration validation
config.validate()

# Use type-safe parameter updates
config.update_strategy_param("spread_bps", 12.0, "market_making")
```

## 🐛 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Configuration file not found
   - Check file paths and current working directory
   - Use absolute paths or ensure files are in expected locations

2. **json.JSONDecodeError**: Malformed JSON
   - Validate JSON syntax using online validators
   - Check for trailing commas or comments in JSON files

3. **KeyError**: Missing strategy configuration
   - Ensure strategy type is defined in strategy_params
   - Check spelling of strategy names

4. **ValueError**: Invalid configuration parameters
   - Run `config.validate()` to identify issues
   - Check parameter ranges and types

### Debug Configuration Loading

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from config import load_backtest_config
config = load_backtest_config()  # Will show debug info
```

---

For more information, see the main project documentation and the configuration class docstrings.
