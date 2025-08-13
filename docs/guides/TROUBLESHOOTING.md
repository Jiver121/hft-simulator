# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the HFT Trading Simulator. Issues are organized by category with step-by-step solutions.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Data Issues](#data-issues)
- [Strategy Problems](#strategy-problems)
- [Performance Issues](#performance-issues)
- [Real-Time Trading Issues](#real-time-trading-issues)
- [Testing and Validation](#testing-and-validation)
- [Dashboard and Visualization](#dashboard-and-visualization)
- [Error Reference](#error-reference)
- [Getting Help](#getting-help)

---

## Quick Diagnostics

### System Health Check

Run this command to get a quick overview of system health:

```bash
python scripts/health_check.py
```

Expected output:
```
✓ Python version: 3.11.5 (compatible)
✓ Dependencies: All required packages installed
✓ Configuration: Valid format and parameters
✓ Data access: Sample data accessible
✓ Memory: 2.4GB available (sufficient)
✓ System: Ready for trading simulation
```

### Common Quick Fixes

1. **Restart Python kernel** (if using Jupyter)
2. **Clear cache:**
   ```bash
   python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"
   ```
3. **Update dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```
4. **Reset configuration:**
   ```bash
   cp config/default_config.yaml config.yaml
   ```

---

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'src'`

**Cause:** Package not installed or Python path not configured correctly.

**Solution:**
```bash
# Method 1: Install in development mode
pip install -e .

# Method 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
```

### Issue: `ImportError: cannot import name 'pandas'` or similar dependency errors

**Cause:** Missing or incompatible dependencies.

**Solution:**
```bash
# Check Python version (3.10+ required)
python --version

# Install/upgrade dependencies
pip install -r requirements.txt

# For specific environments
pip install -r requirements-ml.txt      # ML features
pip install -r requirements-realtime.txt # Real-time trading
```

### Issue: `pip install` fails with permission errors

**Cause:** System permission or virtual environment issues.

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Then install
pip install -r requirements.txt
```

### Issue: Memory errors during installation

**Cause:** Insufficient memory for large packages.

**Solution:**
```bash
# Install with no cache to save memory
pip install -r requirements.txt --no-cache-dir

# Or install packages individually
pip install numpy pandas scipy --no-cache-dir
pip install -r requirements.txt
```

---

## Configuration Problems

### Issue: `ValueError: Invalid configuration format`

**Cause:** Configuration file format or content issues.

**Diagnosis:**
```bash
python scripts/validate_config.py --config config.yaml
```

**Solutions:**

1. **Check YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

2. **Use configuration template:**
   ```bash
   cp config/default_config.yaml config.yaml
   ```

3. **Common YAML issues:**
   - Indentation must be spaces (not tabs)
   - Values need quotes if they contain special characters
   - Boolean values: `true`/`false` (lowercase)

### Issue: Strategy parameters not recognized

**Cause:** Parameter name mismatch or missing configuration section.

**Solution:**
Check parameter names against documentation:
```python
from src.strategies.market_making import MarketMakingStrategy
help(MarketMakingStrategy.__init__)
```

**Common parameter mappings:**
```yaml
# Correct parameter names
strategy:
  class: "MarketMakingStrategy"
  parameters:
    target_spread: 0.01     # Not 'spread'
    max_position: 1000      # Not 'position_limit'
    inventory_target: 0     # New parameter
```

### Issue: Real-time configuration errors

**Cause:** Missing or invalid real-time settings.

**Solution:**
```yaml
real_time:
  enabled: true
  data_feeds:
    - type: "websocket"
      url: "wss://stream.binance.com:9443/ws/btcusdt@ticker"
      symbols: ["BTCUSDT", "ETHUSDT"]
      reconnect_attempts: 5
```

---

## Data Issues

### Issue: `FileNotFoundError: No such file or directory: 'data/sample/hft_data.csv'`

**Cause:** Sample data not available or incorrect path.

**Solution:**
1. **Check data directory:**
   ```bash
   ls -la data/
   ```

2. **Download sample data:**
   ```bash
   python scripts/download_sample_data.py
   ```

3. **Use alternative data:**
   ```python
   # Generate synthetic data for testing
   from src.data.generators import generate_sample_data
   data = generate_sample_data(days=1, symbol='BTCUSDT')
   ```

### Issue: `ValueError: Invalid data format` or parsing errors

**Cause:** Data format incompatibility or corruption.

**Diagnosis:**
```python
import pandas as pd
df = pd.read_csv('data/your_data.csv')
print(df.info())
print(df.head())
```

**Solution:**
1. **Check required columns:**
   ```python
   required_columns = ['timestamp', 'symbol', 'price', 'volume', 'side']
   missing = set(required_columns) - set(df.columns)
   print(f"Missing columns: {missing}")
   ```

2. **Data cleaning:**
   ```python
   from src.data.preprocessor import DataPreprocessor
   processor = DataPreprocessor()
   clean_data = processor.clean_and_validate(df)
   ```

### Issue: Memory errors with large datasets

**Cause:** Insufficient memory for dataset size.

**Solution:**
1. **Use chunked processing:**
   ```python
   from src.data.optimized_ingestion import OptimizedDataLoader
   loader = OptimizedDataLoader(chunk_size=10000)
   for chunk in loader.load_chunks('large_dataset.csv'):
       process_chunk(chunk)
   ```

2. **Reduce data size:**
   ```python
   # Sample data
   df = pd.read_csv('large_file.csv').sample(frac=0.1)
   
   # Filter by date range
   df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
   ```

### Issue: Inconsistent timestamp formats

**Cause:** Mixed timestamp formats in data.

**Solution:**
```python
from src.data.validators import TimestampValidator

validator = TimestampValidator()
df['timestamp'] = validator.normalize_timestamps(df['timestamp'])
```

---

## Strategy Problems

### Issue: Strategy not generating signals

**Cause:** Market conditions, parameter issues, or logic errors.

**Diagnosis:**
```python
# Debug strategy step by step
strategy = MarketMakingStrategy(target_spread=0.01)
market_data = get_sample_market_data()

# Check input data
print(f"Market data: {market_data}")
print(f"Best bid: {market_data.best_bid}")
print(f"Best ask: {market_data.best_ask}")

# Check strategy state
print(f"Current position: {strategy.current_position}")
print(f"Parameters: {strategy.get_parameters()}")

# Generate signals with debug info
signals = strategy.generate_signals(market_data)
print(f"Generated signals: {signals}")
```

**Solutions:**

1. **Check market data validity:**
   ```python
   if not market_data.best_bid or not market_data.best_ask:
       print("Invalid market data - missing quotes")
   ```

2. **Verify strategy parameters:**
   ```python
   # Ensure spread is reasonable
   spread = market_data.best_ask - market_data.best_bid
   if spread <= 0:
       print("Crossed or locked market")
   ```

3. **Position limits:**
   ```python
   if abs(strategy.current_position) >= strategy.max_position:
       print("Position limit reached")
   ```

### Issue: Strategy performance issues

**Cause:** Parameter tuning, market conditions, or implementation issues.

**Diagnosis:**
```python
from src.performance.metrics import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
# ... run strategy ...
metrics = analyzer.calculate_metrics()
print(f"Sharpe ratio: {metrics.sharpe_ratio}")
print(f"Max drawdown: {metrics.max_drawdown}")
print(f"Win rate: {metrics.win_rate}")
```

**Solutions:**

1. **Parameter optimization:**
   ```python
   from src.strategies.strategy_utils import ParameterOptimizer
   
   optimizer = ParameterOptimizer(strategy_class=MarketMakingStrategy)
   best_params = optimizer.optimize(
       data=historical_data,
       param_ranges={
           'target_spread': (0.001, 0.01),
           'max_position': (100, 1000)
       }
   )
   ```

2. **Risk analysis:**
   ```python
   risk_metrics = analyzer.calculate_risk_metrics()
   if risk_metrics.max_drawdown > 0.1:
       print("High drawdown - consider reducing position size")
   ```

### Issue: ML strategy errors

**Cause:** Model loading, feature calculation, or prediction issues.

**Solution:**
```python
from src.strategies.ml_strategy import MLStrategy

# Check model file
import os
if not os.path.exists('models/trained_model.pkl'):
    print("Model file missing - train model first")

# Debug feature calculation
strategy = MLStrategy(model_path='models/trained_model.pkl')
features = strategy.calculate_features(market_data)
print(f"Feature vector shape: {features.shape}")
print(f"Feature names: {strategy.feature_names}")

# Check predictions
predictions = strategy.model.predict(features.reshape(1, -1))
print(f"Model prediction: {predictions}")
```

---

## Performance Issues

### Issue: Slow backtesting performance

**Cause:** Inefficient data processing, strategy logic, or memory usage.

**Diagnosis:**
```python
import cProfile
cProfile.run('run_backtest()', 'backtest_profile.prof')

# Analyze profile
python -m pstats backtest_profile.prof
```

**Solutions:**

1. **Optimize data processing:**
   ```python
   # Use optimized order book
   from src.engine.optimized_order_book import OptimizedOrderBook
   order_book = OptimizedOrderBook(symbol='BTCUSDT')
   ```

2. **Vectorized operations:**
   ```python
   # Instead of loops, use pandas vectorization
   df['signal'] = np.where(df['price'] > df['sma'], 1, -1)
   ```

3. **Memory optimization:**
   ```python
   # Process in chunks
   chunk_size = 10000
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       process_chunk(chunk)
   ```

### Issue: Memory leaks during long runs

**Cause:** Accumulating data structures or references.

**Diagnosis:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

**Solution:**
```python
# Clear trade history periodically
if len(strategy.trade_history) > 10000:
    strategy.trade_history = strategy.trade_history[-1000:]

# Use data rotation
from src.utils.performance_monitor import MemoryManager
memory_manager = MemoryManager(max_history_size=1000)
```

### Issue: High CPU usage

**Cause:** Inefficient algorithms or excessive calculations.

**Solution:**
```python
# Use numba for hot paths
from numba import jit

@jit(nopython=True)
def fast_calculation(data):
    # Compiled function runs much faster
    return np.mean(data)
```

---

## Real-Time Trading Issues

### Issue: WebSocket connection failures

**Cause:** Network issues, invalid URLs, or API limits.

**Diagnosis:**
```python
# Test connection manually
import websocket

def on_message(ws, message):
    print(f"Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws/btcusdt@ticker",
                           on_message=on_message,
                           on_error=on_error)
ws.run_forever()
```

**Solutions:**

1. **Check network connectivity:**
   ```bash
   ping stream.binance.com
   curl -I https://api.binance.com/api/v3/ping
   ```

2. **Verify WebSocket URL:**
   ```python
   # Test with minimal connection
   from src.realtime.data_feeds import WebSocketDataFeed
   
   feed = WebSocketDataFeed(
       url="wss://stream.binance.com:9443/ws/btcusdt@ticker",
       symbols=['BTCUSDT']
   )
   
   # Enable debug logging
   import logging
   logging.getLogger('websocket').setLevel(logging.DEBUG)
   ```

3. **Handle reconnection:**
   ```python
   # Configure reconnection parameters
   feed = WebSocketDataFeed(
       url="wss://stream.binance.com:9443/ws/btcusdt@ticker",
       symbols=['BTCUSDT'],
       reconnect_attempts=10,
       reconnect_interval=5
   )
   ```

### Issue: Data feed lag or missing data

**Cause:** Network latency, processing delays, or feed issues.

**Solution:**
```python
# Monitor data feed performance
from src.utils.performance_monitor import DataFeedMonitor

monitor = DataFeedMonitor()
monitor.track_feed(data_feed)

# Check statistics
stats = monitor.get_statistics()
print(f"Average latency: {stats['avg_latency_ms']:.2f} ms")
print(f"Messages per second: {stats['msg_rate']:.2f}")
print(f"Dropped messages: {stats['dropped_count']}")
```

### Issue: Order execution delays

**Cause:** Processing bottlenecks or broker API issues.

**Diagnosis:**
```python
from src.utils.performance_monitor import ExecutionMonitor

monitor = ExecutionMonitor()
# ... execute orders ...
latency_stats = monitor.get_latency_statistics()
print(f"Average execution time: {latency_stats['mean']:.2f} ms")
print(f"95th percentile: {latency_stats['p95']:.2f} ms")
```

**Solution:**
```python
# Optimize order processing
from src.realtime.order_management import OptimizedOrderManager

order_manager = OptimizedOrderManager(
    batch_size=10,           # Process orders in batches
    async_processing=True,   # Asynchronous execution
    priority_queue=True      # Priority-based ordering
)
```

---

## Testing and Validation

### Issue: Tests failing after updates

**Cause:** Code changes, dependency updates, or environment differences.

**Diagnosis:**
```bash
# Run specific test with verbose output
python -m pytest tests/test_specific.py::test_function -v -s

# Check test environment
python -m pytest --collect-only
```

**Solutions:**

1. **Update test data:**
   ```bash
   python scripts/generate_test_data.py
   ```

2. **Check test configuration:**
   ```python
   # In conftest.py
   @pytest.fixture
   def sample_config():
       return {
           'strategy': {'class': 'MarketMakingStrategy'},
           'risk_management': {'max_position': 1000}
       }
   ```

3. **Mock external dependencies:**
   ```python
   @patch('src.realtime.data_feeds.WebSocketDataFeed')
   def test_with_mock(mock_feed):
       mock_feed.return_value.connect.return_value = True
       # Test logic
   ```

### Issue: Coverage reports showing low coverage

**Cause:** Tests not covering all code paths or missing test files.

**Solution:**
```bash
# Generate detailed coverage report
python -m pytest --cov=src --cov-report=html --cov-report=term-missing

# Check which lines are not covered
python -m coverage report --show-missing
```

### Issue: Integration tests failing

**Cause:** Component interaction issues or configuration problems.

**Solution:**
```python
# Debug integration step by step
def test_integration_debug():
    # Test each component individually
    order_book = OrderBook()
    assert order_book is not None
    
    strategy = MarketMakingStrategy()
    assert strategy is not None
    
    # Test interaction
    signals = strategy.generate_signals(sample_data)
    assert len(signals) > 0
    
    # Test full integration
    simulator = ExecutionSimulator(order_book, strategy)
    results = simulator.run_backtest(sample_data)
    assert results.total_pnl is not None
```

---

## Dashboard and Visualization

### Issue: Dashboard not loading or showing blank page

**Cause:** Flask/SocketIO issues, port conflicts, or frontend errors.

**Diagnosis:**
```bash
# Check if port is available
netstat -an | grep 8080  # Linux/Mac
netstat -an | findstr 8080  # Windows

# Check server logs
python run_dashboard.py --debug
```

**Solutions:**

1. **Use different port:**
   ```python
   app.run(host='0.0.0.0', port=8081, debug=True)
   ```

2. **Check browser console** for JavaScript errors

3. **Verify SocketIO connection:**
   ```javascript
   // In browser console
   socket.on('connect', function() {
       console.log('Connected to server');
   });
   ```

### Issue: Charts not displaying data

**Cause:** Data format issues, JavaScript errors, or API problems.

**Solution:**
```python
# Check data format for Plotly
import json

chart_data = {
    'x': list(range(10)),
    'y': [i**2 for i in range(10)],
    'type': 'scatter'
}

# Ensure JSON serializable
json_data = json.dumps(chart_data)
print("Data is JSON serializable")
```

### Issue: Real-time updates not working

**Cause:** WebSocket connection or event handling issues.

**Solution:**
```python
# Debug SocketIO events
from flask_socketio import emit

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to HFT Simulator'})

@socketio.on('request_data')
def handle_data_request():
    # Send test data
    emit('market_data', {'price': 50000, 'timestamp': time.time()})
```

---

## Error Reference

### Common Error Messages

#### `AttributeError: 'NoneType' object has no attribute 'best_bid'`
- **Cause:** Market data is None or not initialized
- **Solution:** Check data loading and ensure market data is valid

#### `KeyError: 'timestamp'`
- **Cause:** Missing required column in data
- **Solution:** Verify data format and column names

#### `ValueError: operands could not be broadcast together`
- **Cause:** Array shape mismatch in calculations
- **Solution:** Check array dimensions and reshape if necessary

#### `ConnectionError: Failed to establish a new connection`
- **Cause:** Network connectivity or API endpoint issues
- **Solution:** Check internet connection and API status

#### `MemoryError: Unable to allocate array`
- **Cause:** Insufficient memory for operation
- **Solution:** Use chunked processing or reduce data size

### Exit Codes

- **0**: Success
- **1**: General error
- **2**: Configuration error
- **3**: Data error
- **4**: Network error
- **5**: Memory error

---

## Getting Help

### Self-Help Resources

1. **Documentation:**
   - [README.md](README.md) - Quick start guide
   - [docs/](docs/) - Detailed documentation
   - [notebooks/](notebooks/) - Tutorial notebooks

2. **Diagnostic Tools:**
   ```bash
   python scripts/health_check.py        # System health
   python scripts/validate_config.py     # Configuration validation
   python scripts/diagnose_performance.py # Performance analysis
   ```

3. **Log Analysis:**
   ```bash
   # Check logs for errors
   grep "ERROR" logs/hft_simulator.log
   grep "WARN" logs/hft_simulator.log
   ```

### Reporting Issues

When reporting issues, include:

1. **System Information:**
   ```bash
   python scripts/system_info.py > system_info.txt
   ```

2. **Configuration:** Sanitized version of your config file

3. **Error Messages:** Full stack trace

4. **Reproduction Steps:** Minimal example that reproduces the issue

5. **Expected vs Actual Behavior:** What should happen vs what actually happens

### Community Support

- **GitHub Issues:** For bugs and feature requests
- **Discussions:** For questions and community help
- **Documentation:** For tutorials and guides

### Professional Support

For production deployments or custom development:

- Performance optimization consulting
- Custom strategy development
- Production deployment assistance
- Training and workshops

---

## Preventive Measures

### Regular Maintenance

1. **Update Dependencies:**
   ```bash
   pip list --outdated
   pip install --upgrade package-name
   ```

2. **Clean Cache:**
   ```bash
   python -c "import shutil; shutil.rmtree('__pycache__', ignore_errors=True)"
   find . -name "*.pyc" -delete
   ```

3. **Monitor Performance:**
   ```bash
   python scripts/performance_monitor.py --continuous
   ```

### Best Practices

1. **Use Version Control:** Track all configuration changes
2. **Regular Testing:** Run test suite after changes
3. **Monitor Resources:** Watch memory and CPU usage
4. **Backup Data:** Regular backups of configuration and results
5. **Documentation:** Keep internal documentation updated

Remember: Prevention is better than troubleshooting. Follow best practices and monitor your system regularly to avoid issues.
