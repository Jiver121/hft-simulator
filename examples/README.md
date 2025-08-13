# Examples and Demo Scripts

This directory contains example scripts and demo applications that showcase the HFT Simulator functionality.

## Directory Contents

### Core Demos
- `demo.py` - Basic demonstration of core functionality
- `simple_demo.py` - Simple introductory example
- `complete_system_demo.py` - Comprehensive system demonstration
- `demo_complete_system.py` - Alternative complete system demo
- `demo_dashboard_complete.py` - Complete dashboard demonstration

### Specialized Examples
- `ml_trading_demo.py` - Machine learning trading strategy demonstration
- `multi_asset_demo.py` - Multi-asset trading example
- `realtime_trading_demo.py` - Real-time trading demonstration

### Dashboard Examples
- `run_dashboard.py` - Dashboard runner script
- `simple_dashboard.py` - Simple dashboard example

### Validation and Testing Examples
- `full_system_verification.py` - Complete system validation
- `system_verification.py` - System verification example
- `validate_realtime_system.py` - Real-time system validation
- `test_execution_simulator.py` - Execution simulator testing

### Performance and Analysis
- `analyze_profile.py` - Performance profiling and analysis
- `profile_backtest.py` - Backtesting profiler

### WebSocket Examples
- `quick_websocket_test.py` - Quick WebSocket connectivity test

## Running Examples

Each example script can be run independently:

```bash
# Basic demo
python examples/demo.py

# ML trading demo
python examples/ml_trading_demo.py

# Dashboard demo
python examples/run_dashboard.py

# System verification
python examples/full_system_verification.py
```

## Requirements

Most examples require the main project dependencies. Some specialized examples may have additional requirements:

- Real-time examples may require API keys for data feeds
- ML examples require machine learning libraries (see `requirements-ml.txt`)
- Dashboard examples may require web dependencies

## Configuration

Examples use configuration files from the main project. Some examples may create temporary configuration files or use embedded configuration.

## Example Categories

### 1. Basic Demonstrations
Start with these examples to understand core functionality:
- `demo.py` - Overview of main features
- `simple_demo.py` - Step-by-step basic example

### 2. Advanced Features
Explore advanced capabilities:
- `ml_trading_demo.py` - Machine learning integration
- `multi_asset_demo.py` - Multiple asset handling
- `complete_system_demo.py` - Full system capabilities

### 3. Real-time Systems
Real-time trading and data handling:
- `realtime_trading_demo.py` - Live trading simulation
- `quick_websocket_test.py` - WebSocket connectivity
- `validate_realtime_system.py` - Real-time validation

### 4. Visualization and Monitoring
Dashboard and visualization examples:
- `run_dashboard.py` - Full dashboard application
- `simple_dashboard.py` - Basic dashboard setup
- `demo_dashboard_complete.py` - Complete dashboard demo

### 5. Testing and Validation
System verification and testing:
- `full_system_verification.py` - Comprehensive testing
- `system_verification.py` - Basic verification
- `test_execution_simulator.py` - Execution testing

### 6. Performance Analysis
Performance monitoring and analysis:
- `analyze_profile.py` - Performance profiling
- `profile_backtest.py` - Backtesting analysis

## Development and Customization

These examples serve as templates for:
- Building custom trading strategies
- Implementing real-time data feeds
- Creating custom dashboards
- Developing performance monitoring tools

Feel free to modify and extend these examples for your specific use cases.
