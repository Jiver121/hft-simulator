# 🚀 HFT Simulator - Complete Setup Guide

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#cross-platform-support)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Complete setup guide for the High-Frequency Trading Simulator with unified interface and advanced ML features.**

## 📋 Table of Contents

- [🎯 Quick Start (5 minutes)](#-quick-start-5-minutes)
- [📦 Installation Options](#-installation-options)
- [🎮 Usage Examples](#-usage-examples)
- [⚙️ Advanced Configuration](#️-advanced-configuration)
- [🚨 Troubleshooting](#-troubleshooting)
- [🔧 Development Setup](#-development-setup)

## 🎯 Quick Start (5 minutes)

### Prerequisites

```bash
# Check Python version (3.10+ required)
python --version

# Should output: Python 3.10.x or higher
```

### 1️⃣ Clone and Setup

```bash
# Clone repository
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Test Installation

```bash
# Create sample data
python main.py --mode create-sample-data

# Run basic backtest
python main.py --mode backtest --data ./data/ --output ./results/

# View results
python main.py --mode analysis --input ./results/backtest_summary.json
```

### 3️⃣ Try Advanced Features

```bash
# Complete system demo (showcases everything)
python main.py --mode demo

# Real-time dashboard
python main.py --mode dashboard
# Then open: http://127.0.0.1:8080
```

**🎉 Success!** Your HFT Simulator is ready to use.

---

## 📦 Installation Options

### Option 1: Basic Installation (Recommended)

**Features**: CSV backtesting, basic strategies, performance analysis
**Requirements**: ~500MB disk space

```bash
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: Full Installation (All Features)

**Features**: ML strategies (500+ features), real-time trading, advanced analytics
**Requirements**: ~2GB disk space, 8GB+ RAM

```bash
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt      # ML dependencies
pip install -r requirements-realtime.txt # Real-time dependencies
```

### Option 3: Docker Installation

**Features**: Containerized, isolated environment
**Requirements**: Docker installed

```bash
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator
docker build -t hft-simulator .
docker run -p 8080:8080 hft-simulator
```

### Option 4: Development Installation

**Features**: All features + development tools
**Requirements**: ~3GB disk space

```bash
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .  # Editable install
```

---

## 🎮 Usage Examples

### Basic Usage

```bash
# 1. Create test data
python main.py --mode create-sample-data

# 2. Run basic backtest
python main.py --mode backtest --data ./data/AAPL_data.csv --output ./results/

# 3. Analyze results
python main.py --mode analysis --input ./results/backtest_summary.json
```

### Intermediate Usage

```bash
# Multi-symbol backtesting
python main.py --mode backtest --data ./data/ --strategy momentum --output ./results/momentum/

# Custom date range
python main.py --mode backtest --data ./data/ --start-date 2024-01-01 --end-date 2024-06-30 --output ./results/

# Verbose logging
python main.py --mode backtest --data ./data/ --output ./results/ --verbose
```

### Advanced Usage (Requires Full Installation)

```bash
# ML-powered backtesting with 500+ features
python main.py --mode ml-backtest --symbols BTCUSDT,ETHUSDT --data ./data/

# Real-time multi-asset trading simulation
python main.py --mode realtime --symbols BTCUSDT,ETHUSDT,BNBUSDT --duration 300

# Enhanced dashboard with ML insights
python main.py --mode dashboard --enhanced

# Complete system demonstration
python main.py --mode demo --advanced --output ./demo_results.json
```

### Command Reference

| Command | Description | Requirements |
|---------|-------------|--------------|
| `create-sample-data` | Generate test datasets | Basic |
| `backtest` | CSV backtesting | Basic |
| `analysis` | Results analysis | Basic |
| `ml-backtest` | ML strategy backtesting | Full |
| `realtime` | Real-time trading sim | Full |
| `dashboard` | Web dashboard | Full |
| `demo` | Complete demo | Full |

---

## ⚙️ Advanced Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Data Configuration
DATA_PATH=./data
RESULTS_PATH=./results

# Trading Configuration  
DEFAULT_SYMBOL=BTCUSDT
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=1000

# Real-time Configuration
WEBSOCKET_URL=wss://stream.binance.com:9443/ws
API_RATE_LIMIT=1000
RECONNECTION_DELAY=5

# Dashboard Configuration
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8080
DASHBOARD_DEBUG=true
```

### Custom Configuration Files

Create `config/trading_config.json`:

```json
{
  "strategy_config": {
    "market_making": {
      "spread_bps": 10.0,
      "order_size": 100,
      "max_inventory": 500
    },
    "momentum": {
      "momentum_threshold": 0.001,
      "order_size": 200,
      "max_positions": 5
    }
  },
  "risk_config": {
    "max_position_size": 1000,
    "max_order_size": 100,
    "risk_limit": 10000.0
  },
  "execution_config": {
    "commission_rate": 0.0005,
    "slippage_bps": 1.0,
    "tick_size": 0.01
  }
}
```

Use custom config:
```bash
python main.py --mode backtest --config ./config/trading_config.json --data ./data/
```

### Performance Tuning

For large datasets or intensive operations:

```bash
# Enable parallel processing
python main.py --mode backtest --data ./data/ --parallel --workers 4

# Optimize memory usage
export PYTHONHASHSEED=0
export MALLOC_TRIM_THRESHOLD_=100000

# Run with optimizations
python -O main.py --mode ml-backtest --symbols BTCUSDT --data ./data/large_dataset.csv
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```
ImportError: No module named 'src.strategies.ml_strategy'
```

**Solution**: Install ML dependencies or use basic mode:
```bash
pip install -r requirements-ml.txt
# OR use basic mode:
python main.py --mode backtest --data ./data/
```

#### 2. Memory Issues
```
MemoryError: Unable to allocate array
```

**Solution**: Use data chunking or reduce dataset size:
```bash
# Process smaller chunks
python main.py --mode backtest --data ./data/small_dataset.csv

# Or increase system memory/swap
```

#### 3. Dashboard Not Loading
```
Dashboard failed to start on port 8080
```

**Solution**: Check port availability and dependencies:
```bash
# Check if port is in use
netstat -an | grep 8080

# Install dashboard dependencies
pip install -r requirements-realtime.txt

# Try different port
python main.py --mode dashboard --port 8081
```

#### 4. WebSocket Connection Issues
```
Real-time data feed connection failed
```

**Solution**: Check internet connection and firewall:
```bash
# Test connection manually
ping stream.binance.com

# Use mock data for testing
python main.py --mode realtime --mock-data
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
# Enable debug logging
python main.py --mode backtest --data ./data/ --verbose

# Check log files
cat logs/hft_simulator.log

# Enable Python debug
python -u -v main.py --mode backtest --data ./data/
```

### System Requirements Check

```bash
# Check Python version
python --version

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"

# Check disk space
df -h .

# Check dependencies
pip check
```

---

## 🔧 Development Setup

### For Contributors

```bash
# Clone repository
git clone https://github.com/your-username/hft-simulator.git
cd hft-simulator

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate  # Windows: venv-dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
isort src/

# Build documentation
cd docs/
make html
```

### IDE Configuration

#### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm
1. Open project in PyCharm
2. Configure interpreter: Settings → Python Interpreter → Add → Existing environment
3. Select `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows)
4. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

### Building Distribution

```bash
# Build package
python setup.py sdist bdist_wheel

# Check package
twine check dist/*

# Test installation
pip install dist/hft_simulator-*.whl
```

---

## 🌐 Cross-Platform Support

### Windows
```cmd
# Use Command Prompt or PowerShell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --mode demo
```

### macOS
```bash
# Install Python 3.10+ via Homebrew if needed
brew install python@3.10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --mode demo
```

### Linux (Ubuntu/Debian)
```bash
# Install Python 3.10+ if needed
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py --mode demo
```

---

## 📚 Next Steps

After successful installation:

1. **📖 Read Documentation**: Check `docs/` for detailed guides
2. **🎓 Try Notebooks**: Explore `notebooks/` for interactive tutorials  
3. **🔍 Examine Examples**: Review `examples/` for usage patterns
4. **⚙️ Customize**: Modify `config/` files for your needs
5. **🧪 Experiment**: Try different strategies and parameters
6. **📈 Analyze**: Use the dashboard and analytics tools
7. **🚀 Extend**: Add your own strategies and features

---

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/your-username/hft-simulator/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/hft-simulator/discussions)
- 📖 **Documentation**: [docs/](docs/)
- 🎯 **Examples**: [examples/](examples/)

---

**🎉 Happy Trading!** Your HFT Simulator is ready for professional-grade algorithmic trading simulation and research.
