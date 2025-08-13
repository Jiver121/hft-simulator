# Testing Structure

This document describes the reorganized testing structure for the HFT Simulator project.

## Directory Structure

```
tests/
├── unit/                          # Unit tests for individual components
├── integration/                   # Integration tests for system interactions  
├── performance/                   # Performance and benchmarking tests
├── fixtures/                      # Test data and fixtures
├── output/                        # Test output and artifacts
├── conftest.py                    # Pytest configuration and shared fixtures
└── README.md                      # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
Tests for individual components and modules:
- `test_components_direct.py` - Direct component testing
- `test_debug_order_book.py` - Order book debugging tests
- `test_direct_matching.py` - Core order matching logic
- `test_direct_modules.py` - Direct module testing
- `test_final_matching.py` - Final matching algorithm tests
- `test_market_data.py` - Market data handling tests
- `test_market_order_fixes.py` - Market order fix validation
- `test_optimized_order_book.py` - Optimized order book tests
- `test_order_book.py` - Standard order book tests
- `test_order_matching_scenarios.py` - Order matching scenarios
- `test_order_types.py` - Order type handling tests
- `test_strategies.py` - Trading strategy tests

### Integration Tests (`tests/integration/`)
Tests for system-level interactions and workflows:
- `test_advanced_features.py` - Advanced feature integration
- `test_binance_websocket.py` - WebSocket integration with Binance
- `test_dashboard.py` - Dashboard functionality integration
- `test_edge_conditions.py` - Edge case handling
- `test_end_to_end.py` - Complete end-to-end workflows
- `test_enhanced_system.py` - Enhanced system features
- `test_enhanced_websocket_feed.py` - Enhanced WebSocket feed
- `test_full_trading_session.py` - Complete trading session simulation
- `test_metrics_accuracy.py` - Metrics accuracy validation
- `test_multi_strategy_portfolio.py` - Multi-strategy portfolio testing
- `test_order_flow_complete.py` - Complete order flow testing
- `test_order_flow_integration.py` - Order flow integration
- `test_order_flow_pnl.py` - Order flow P&L calculation
- `test_progress_tracking.py` - Progress tracking functionality
- `test_realtime_basic.py` - Basic real-time functionality
- `test_realtime_system.py` - Real-time system integration
- `test_strategy_execution_scenarios.py` - Strategy execution scenarios
- `test_strategy_integration.py` - Strategy system integration
- `test_system_comprehensive.py` - Comprehensive system testing
- `test_system_stress.py` - System stress testing
- `test_system_stress_concurrent.py` - Concurrent stress testing
- `test_websocket_connection.py` - WebSocket connection testing

### Performance Tests (`tests/performance/`)
Performance benchmarks and optimization validation:
- `test_benchmarks.py` - General performance benchmarks
- `test_dashboard_performance.py` - Dashboard performance testing
- `test_optimizations.py` - Optimization validation
- `test_optimization_benchmarks.py` - Optimization benchmarks
- `test_optimized_components.py` - Optimized component testing
- `test_regression.py` - Performance regression testing

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run by Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Performance tests only
pytest tests/performance/
```

### Run Specific Test Files
```bash
# Run a specific test file
pytest tests/unit/test_order_book.py

# Run with verbose output
pytest -v tests/integration/test_dashboard.py

# Run with coverage
pytest --cov=src tests/unit/
```

## Test Naming Convention

All test files follow the `test_*.py` naming convention as required by pytest.

## Configuration

Test configuration is centralized in `conftest.py` which includes:
- Shared fixtures
- Test setup and teardown
- Common test utilities
- Pytest configuration

## Test Data

Test fixtures and data files are stored in the `fixtures/` directory to keep test data organized and reusable across different test suites.

## Test Output

Test outputs, logs, and artifacts are stored in the `output/` directory to avoid cluttering the main test directories.
