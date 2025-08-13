# Order Matching Unit Tests

This document describes the comprehensive unit tests created to verify the order matching engine fixes.

## Overview

The unit tests in `tests/unit/test_order_matching_scenarios.py` are designed to thoroughly validate the order matching functionality after bug fixes. These tests ensure that:

1. **Market orders match correctly** with existing limit orders
2. **Limit orders cross spreads** and generate appropriate trades  
3. **Partial fills** are handled accurately
4. **Trade objects** are created with correct price, volume, and timestamps
5. **Price-time priority** is enforced properly

## Test Coverage

### Core Matching Scenarios
- `test_market_buy_matches_existing_ask_limit`: Tests market buy orders matching with ask limit orders
- `test_market_sell_matches_existing_bid_limit`: Tests market sell orders matching with bid limit orders
- `test_limit_order_crossing_scenarios`: Tests aggressive limit orders that cross the spread
- `test_partial_fill_scenarios`: Tests orders that partially fill against multiple price levels
- `test_trade_object_creation_accuracy`: Validates trade object properties and calculations

### Validation Tests
- `test_price_time_priority_enforcement`: Ensures proper FIFO order matching
- `test_edge_cases_and_error_conditions`: Tests error handling and boundary conditions
- `test_multiple_symbol_isolation`: Verifies order book isolation between symbols

### Performance Tests
- `test_high_volume_matching_performance`: Tests performance with high order volume
- `test_memory_usage_stability`: Validates memory usage during extended operation

## Running the Tests

### Quick Start
```bash
# Run the test suite with the provided runner
python run_matching_tests.py
```

### Manual Test Execution
```bash
# Run only the matching scenario tests
python -m unittest tests.unit.test_order_matching_scenarios.TestOrderMatchingScenarios -v

# Run performance tests
python -m unittest tests.unit.test_order_matching_scenarios.TestOrderMatchingPerformance -v

# Run all tests in the module
python -m unittest tests.unit.test_order_matching_scenarios -v
```

### Individual Test Execution
```bash
# Run a specific test
python -m unittest tests.unit.test_order_matching_scenarios.TestOrderMatchingScenarios.test_market_buy_matches_existing_ask_limit -v
```

## Test Structure

### Test Classes
1. **TestOrderMatchingScenarios**: Core functional tests for order matching logic
2. **TestOrderMatchingPerformance**: Performance and load testing

### Test Helpers
- `_create_order()`: Creates test orders with consistent timestamps
- `_assert_trade_validity()`: Validates trade object properties
- Setup/teardown methods ensure clean test environment

### Test Data
- Uses consistent base timestamps for reproducible results
- Creates realistic order scenarios with proper price/volume relationships
- Tests both successful matches and edge cases

## Key Test Scenarios

### Market Order Matching
```python
# Test verifies:
# 1. Ask limit order added to book without immediate trades
# 2. Market buy order matches and generates trade
# 3. Trade has correct price (ask price), volume, and order references
# 4. Order statuses updated correctly (FILLED/PARTIAL)
```

### Limit Order Crossing  
```python
# Test verifies:
# 1. Aggressive buy limit crosses spread (price > best ask)
# 2. Trade executes at resting order price (price improvement)
# 3. Proper order ID references in trade object
# 4. Same verification for aggressive sell limit orders
```

### Partial Fills
```python
# Test verifies:
# 1. Large order fills against multiple price levels
# 2. Correct volume matching across levels
# 3. Price-time priority enforcement
# 4. Remaining volume tracking on partially filled orders
```

### Trade Object Validation
```python
# Test verifies:
# 1. Trade price/volume accuracy to 2 decimal places
# 2. Trade value calculation (price * volume)
# 3. Timestamp accuracy and ordering
# 4. Proper buy/sell order ID references
# 5. Aggressor side identification
# 6. Unique trade ID generation
```

## Expected Results

### Successful Test Run
When all tests pass, you should see:
```
✅ Market orders match with existing limit orders
✅ Limit orders cross spreads correctly  
✅ Partial fills are handled properly
✅ Trade objects are created with accurate data
✅ Price-time priority is enforced
```

### Common Failure Scenarios
If tests fail, common issues may include:
- Order matching logic not generating trades
- Incorrect trade prices or volumes
- Missing or incorrect order ID references
- Price-time priority violations
- Order status not updating properly

## Test Design Principles

### Comprehensive Coverage
- Tests cover all major order matching scenarios
- Both success and failure cases included
- Edge cases and boundary conditions tested

### Isolation and Independence
- Each test is independent and can run standalone
- Clean setup/teardown ensures no test interference
- Separate matching engines for multi-symbol tests

### Realistic Scenarios
- Uses realistic price and volume values
- Tests time-based ordering with millisecond precision
- Simulates real-world order flow patterns

### Clear Assertions
- Detailed assertion messages explain expected vs actual results
- Trade validation helper provides comprehensive checks
- Test failures include diagnostic information

## Troubleshooting

### Import Errors
If you get import errors when running tests:
```bash
# Make sure you're in the project root directory
cd /path/to/hft-simulator

# Verify Python path includes src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use the provided test runner which handles paths automatically
python run_matching_tests.py
```

### Test Failures
If specific tests fail:
1. Check the detailed error messages in test output
2. Verify the order book implementation handles the test scenario
3. Check that trade generation logic is working correctly
4. Ensure order status updates are happening properly

### Performance Issues
If performance tests fail:
1. Check system load during test execution
2. Verify memory constraints are reasonable for your system
3. Consider running tests on a dedicated test environment

## Integration with CI/CD

These tests are designed to be integration-ready:
- Exit codes indicate success/failure for automated systems
- Detailed XML output can be generated with appropriate test runners
- Tests run in reasonable time for CI pipelines
- Clear pass/fail criteria for automated verification

## Future Enhancements

Potential areas for test expansion:
- Stop-loss and other advanced order types
- Market maker rebate calculations
- Latency measurements and benchmarking  
- Multi-threaded order processing scenarios
- Network simulation and failure recovery testing

---

**Note**: These tests are specifically designed to verify the order matching engine fixes. They should all pass if the core matching logic is working correctly. Any failures indicate issues that need to be addressed in the matching engine implementation.
