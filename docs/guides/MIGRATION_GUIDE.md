# Migration Guide

This guide provides step-by-step instructions for migrating between major versions of the HFT Trading Simulator. Each section covers breaking changes, new features, and migration steps.

## Table of Contents

- [Version 1.x to 2.0.0](#version-1x-to-200)
- [Version 1.4.x to 1.5.0](#version-14x-to-150)
- [Version 1.3.x to 1.4.0](#version-13x-to-140)
- [Version 1.2.x to 1.3.0](#version-12x-to-130)
- [General Migration Best Practices](#general-migration-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Version 1.x to 2.0.0

**Migration Difficulty:** High  
**Estimated Time:** 2-4 hours  
**Breaking Changes:** Yes

### Overview

Version 2.0.0 introduces significant architectural changes including real-time trading capabilities, enhanced ML strategies, and a completely rewritten configuration system.

### Breaking Changes

#### 1. Strategy Interface Changes

**Old Interface (v1.x):**
```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data):
        # Old implementation
        return signals
    
    def update_position(self, trade_data):
        # Old implementation
        pass
```

**New Interface (v2.0.0):**
```python
from src.strategies.base_strategy import BaseStrategy
from src.strategies.strategy_framework import StrategySignal

class MyStrategy(BaseStrategy):
    def generate_signals(self, market_data, context=None):
        # New implementation with context
        signals = []
        # ... signal generation logic
        return [StrategySignal(signal_type, side, quantity, price, confidence) 
                for signal in signals]
    
    def on_trade_execution(self, execution_result):
        # New callback method
        pass
    
    def on_market_data_update(self, market_data):
        # New real-time callback
        pass
```

**Migration Steps:**
1. Update strategy method signatures to include `context` parameter
2. Convert signal returns to `StrategySignal` objects
3. Implement new callback methods if using real-time features
4. Update position management to use new portfolio interface

#### 2. Configuration Format Changes

**Old Configuration (v1.x):**
```json
{
    "strategy": {
        "type": "market_making",
        "spread": 0.01,
        "position_limit": 1000
    },
    "execution": {
        "slippage": 0.001
    }
}
```

**New Configuration (v2.0.0):**
```yaml
# config.yaml
strategy:
  class: "MarketMakingStrategy"
  parameters:
    target_spread: 0.01
    max_position: 1000
    inventory_target: 0
    skew_adjustment: true
    
risk_management:
  position_limits:
    max_position_size: 1000
    max_exposure: 100000
  
execution:
  slippage_model:
    type: "linear"
    impact_factor: 0.001
  
real_time:
  enabled: true
  data_feeds:
    - type: "websocket"
      url: "wss://stream.binance.com:9443/ws/btcusdt@ticker"
```

**Migration Steps:**
1. Convert JSON configuration to YAML format
2. Restructure configuration hierarchy
3. Update parameter names (see mapping table below)
4. Add new required sections for real-time and risk management

#### 3. Parameter Name Mappings

| Old Parameter (v1.x) | New Parameter (v2.0.0) | Notes |
|----------------------|------------------------|-------|
| `spread` | `target_spread` | Same functionality |
| `position_limit` | `max_position` | Same functionality |
| `slippage` | `execution.slippage_model.impact_factor` | Now part of slippage model |
| `risk_threshold` | `risk_management.var_threshold` | Enhanced risk framework |
| `data_source` | `real_time.data_feeds[0].type` | Multiple feeds supported |

#### 4. Import Path Changes

**Old Imports (v1.x):**
```python
from src.engine.order_book import OrderBook
from src.strategies.market_making import MarketMaker
from src.execution.simulator import Simulator
```

**New Imports (v2.0.0):**
```python
from src.engine.order_book import OrderBook  # No change
from src.strategies.market_making import MarketMakingStrategy
from src.execution.simulator import ExecutionSimulator
from src.realtime.trading_system import RealTimeTradingSystem  # New
```

### New Features

#### 1. Real-Time Trading System

```python
# New real-time capabilities
from src.realtime.trading_system import RealTimeTradingSystem
from src.realtime.config import RealTimeConfig

config = RealTimeConfig.from_file("config.yaml")
trading_system = RealTimeTradingSystem(config)

# Start real-time trading
await trading_system.start()
```

#### 2. Enhanced ML Strategies

```python
# New ML strategy with 97 features
from src.strategies.ml_strategy import MLStrategy

ml_strategy = MLStrategy(
    model_path="models/trained_model.pkl",
    feature_config="features/ml_features.yaml",
    prediction_threshold=0.7
)
```

#### 3. Advanced Risk Management

```python
# New risk management system
from src.performance.risk_manager import RealTimeRiskManager

risk_manager = RealTimeRiskManager(
    max_position_size=1000,
    max_drawdown=0.05,
    var_threshold=0.02,
    circuit_breaker_enabled=True
)
```

### Migration Checklist

- [ ] **Backup your existing code and configuration**
- [ ] **Update Python version to 3.10+** (required for v2.0.0)
- [ ] **Install new dependencies:**
  ```bash
  pip install -r requirements-realtime.txt
  ```
- [ ] **Convert configuration files:**
  - Convert JSON to YAML format
  - Update parameter names
  - Add new required sections
- [ ] **Update strategy implementations:**
  - Implement new method signatures
  - Convert signal returns to `StrategySignal` objects
  - Add callback methods if needed
- [ ] **Update import statements**
- [ ] **Test migration:**
  ```bash
  python -m pytest tests/migration/ -v
  ```
- [ ] **Run full test suite:**
  ```bash
  python -m pytest tests/ -v
  ```

### Code Migration Examples

#### Example 1: Market Making Strategy

**Before (v1.x):**
```python
class MarketMaker(BaseStrategy):
    def __init__(self, spread=0.01, position_limit=1000):
        self.spread = spread
        self.position_limit = position_limit
    
    def generate_signals(self, market_data):
        bid_price = market_data.best_bid - self.spread/2
        ask_price = market_data.best_ask + self.spread/2
        
        signals = []
        if self.position < self.position_limit:
            signals.append({
                'side': 'BUY',
                'price': bid_price,
                'quantity': 100
            })
        
        return signals
```

**After (v2.0.0):**
```python
from src.strategies.strategy_framework import StrategySignal, SignalType

class MarketMakingStrategy(BaseStrategy):
    def __init__(self, target_spread=0.01, max_position=1000, inventory_target=0):
        super().__init__()
        self.target_spread = target_spread
        self.max_position = max_position
        self.inventory_target = inventory_target
    
    def generate_signals(self, market_data, context=None):
        if not market_data.best_bid or not market_data.best_ask:
            return []
        
        # Calculate fair value with inventory skew
        fair_value = (market_data.best_bid + market_data.best_ask) / 2
        inventory_skew = self.calculate_inventory_skew()
        
        bid_price = fair_value - self.target_spread/2 + inventory_skew
        ask_price = fair_value + self.target_spread/2 + inventory_skew
        
        signals = []
        
        # Generate bid signal
        if self.current_position < self.max_position:
            signals.append(StrategySignal(
                signal_type=SignalType.LIMIT_ORDER,
                side=OrderSide.BUY,
                quantity=self.calculate_order_size(market_data),
                price=bid_price,
                confidence=0.8
            ))
        
        # Generate ask signal
        if self.current_position > -self.max_position:
            signals.append(StrategySignal(
                signal_type=SignalType.LIMIT_ORDER,
                side=OrderSide.SELL,
                quantity=self.calculate_order_size(market_data),
                price=ask_price,
                confidence=0.8
            ))
        
        return signals
    
    def on_trade_execution(self, execution_result):
        # Handle trade execution callback
        self.update_inventory(execution_result)
        self.log_trade(execution_result)
    
    def calculate_inventory_skew(self):
        # New inventory management logic
        current_inventory = self.current_position - self.inventory_target
        return current_inventory * 0.0001  # Small skew adjustment
```

---

## Version 1.4.x to 1.5.0

**Migration Difficulty:** Medium  
**Estimated Time:** 1-2 hours  
**Breaking Changes:** Minor

### Overview

Version 1.5.0 introduces enhanced market making capabilities, statistical arbitrage strategies, and improved visualization components.

### Changes

#### 1. Enhanced Market Making Strategy

The market making strategy now supports inventory management and adaptive spreads.

**Migration Steps:**
1. Update strategy parameter names
2. Add optional inventory management parameters
3. Update visualization components if customized

#### 2. New Statistical Arbitrage Module

```python
# New statistical arbitrage capabilities
from src.strategies.arbitrage_strategy import StatisticalArbitrageStrategy

arb_strategy = StatisticalArbitrageStrategy(
    pairs=[('BTCUSDT', 'ETHUSDT')],
    lookback_window=100,
    z_score_threshold=2.0
)
```

#### 3. Visualization Enhancements

Dark theme and responsive design are now default. Update custom CSS if you have any.

### Migration Checklist

- [ ] Update market making strategy parameters
- [ ] Test new visualization components
- [ ] Update custom CSS for dark theme compatibility
- [ ] Run regression tests

---

## Version 1.3.x to 1.4.0

**Migration Difficulty:** Medium  
**Estimated Time:** 1-2 hours  
**Breaking Changes:** Minor

### Overview

Version 1.4.0 introduces real-time data feeds and advanced execution models.

### Changes

#### 1. Real-Time Data Feed Integration

```python
# New WebSocket data feeds
from src.realtime.data_feeds import WebSocketDataFeed

data_feed = WebSocketDataFeed(
    url="wss://stream.binance.com:9443/ws/btcusdt@ticker",
    symbols=['BTCUSDT', 'ETHUSDT']
)
```

#### 2. Enhanced Execution Models

Market impact and slippage modeling are now more sophisticated.

### Migration Checklist

- [ ] Install WebSocket dependencies: `pip install websocket-client`
- [ ] Update execution model configuration
- [ ] Test real-time data integration
- [ ] Validate execution accuracy

---

## Version 1.2.x to 1.3.0

**Migration Difficulty:** Low  
**Estimated Time:** 30-60 minutes  
**Breaking Changes:** None

### Overview

Version 1.3.0 adds risk management and visualization frameworks with backward compatibility.

### New Features

- Risk management with VaR calculation
- Interactive visualization framework
- Multi-strategy portfolio support

### Migration Checklist

- [ ] Install new dependencies: `pip install plotly bokeh`
- [ ] Optionally configure risk management
- [ ] Test visualization components
- [ ] Update documentation

---

## General Migration Best Practices

### 1. Pre-Migration Checklist

- [ ] **Backup everything:** Code, configuration, data, and results
- [ ] **Review changelog:** Understand all changes and new features  
- [ ] **Check dependencies:** Verify Python version and package compatibility
- [ ] **Plan downtime:** Estimate migration time and plan accordingly
- [ ] **Prepare rollback:** Have a rollback plan ready

### 2. Migration Process

1. **Create development environment:**
   ```bash
   git checkout -b migration-v2.0.0
   ```

2. **Install new version:**
   ```bash
   pip install --upgrade hft-simulator==2.0.0
   ```

3. **Run migration scripts:**
   ```bash
   python scripts/migrate_config.py --from-version=1.5.0
   python scripts/validate_migration.py
   ```

4. **Test thoroughly:**
   ```bash
   python -m pytest tests/ -v
   python run_integration_tests.py --type migration
   ```

5. **Gradual rollout:**
   - Test with sample data first
   - Run parallel systems during transition
   - Monitor performance and accuracy

### 3. Post-Migration Validation

- [ ] **Run full test suite** and ensure all tests pass
- [ ] **Compare results** with previous version using same data
- [ ] **Performance testing** to ensure no regression
- [ ] **Documentation updates** for team members
- [ ] **Monitor production** closely after deployment

### 4. Common Migration Patterns

#### Configuration Migration
```python
# Helper script for config migration
def migrate_config_v1_to_v2(old_config_path, new_config_path):
    with open(old_config_path, 'r') as f:
        old_config = json.load(f)
    
    new_config = {
        'strategy': {
            'class': map_strategy_type(old_config['strategy']['type']),
            'parameters': migrate_parameters(old_config['strategy'])
        },
        'risk_management': create_default_risk_config(),
        'real_time': {'enabled': False}  # Disabled by default
    }
    
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f)
```

#### Strategy Migration
```python
# Base class for migrating strategies
class StrategyMigrator:
    def migrate_v1_to_v2(self, old_strategy):
        new_strategy = self.create_v2_strategy(old_strategy.__class__.__name__)
        new_strategy.migrate_parameters(old_strategy.get_parameters())
        return new_strategy
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors After Migration
```
ImportError: cannot import name 'MarketMaker' from 'src.strategies.market_making'
```

**Solution:** Update import paths according to the migration guide.

#### 2. Configuration Format Errors
```
ValueError: Invalid configuration format
```

**Solution:** Use migration scripts to convert configuration files.

#### 3. Strategy Interface Errors
```
TypeError: generate_signals() takes 2 positional arguments but 3 were given
```

**Solution:** Update strategy method signatures to match new interface.

#### 4. Performance Regression
```
Performance: 10x slower than previous version
```

**Solution:** 
- Check if optimization features are enabled
- Verify configuration parameters
- Run performance profiler

#### 5. Real-Time Connection Issues
```
WebSocketConnectionError: Failed to connect to data feed
```

**Solution:**
- Verify network connectivity
- Check API credentials and permissions
- Validate WebSocket URL format

### Getting Help

1. **Check documentation:** [docs/](docs/)
2. **Search issues:** Look for similar migration issues
3. **Run diagnostics:**
   ```bash
   python scripts/diagnose_migration.py
   ```
4. **Contact support:** Include migration diagnostics output

### Migration Support Tools

#### 1. Configuration Validator
```bash
python scripts/validate_config.py --config config.yaml --version 2.0.0
```

#### 2. Migration Checker
```bash
python scripts/check_migration.py --from 1.5.0 --to 2.0.0
```

#### 3. Performance Comparison
```bash
python scripts/compare_performance.py --before results/v1.5.0/ --after results/v2.0.0/
```

---

## Support

If you encounter issues during migration:

- **Documentation**: [docs/](docs/)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: Open a GitHub issue with migration details
- **Performance**: [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md)

Remember: Migration is a process, not an event. Take time to test thoroughly and validate results before deploying to production.
