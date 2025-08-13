# Error Recovery Guide

This guide documents error handling, recovery, and failover strategies in the HFT simulator platform.

## Table of Contents

1. [Error Handling](#error-handling)
2. [Input Validation](#input-validation)
3. [Input Sanitization](#input-sanitization)
4. [Order Book State Recovery](#order-book-state-recovery)
5. [Circuit Breakers](#circuit-breakers)
6. [Structured Logging](#structured-logging)
7. [WebSocket Reconnection](#websocket-reconnection)
8. [Graceful Degradation](#graceful-degradation)
9. [Testing Error Handling](#testing-error-handling)
10. [Troubleshooting](#troubleshooting)

## Error Handling

### Centralized Error Handling

All errors are routed through the `ErrorHandler` class in `src/utils/error_handler.py`:

```python
from src.utils.error_handler import ErrorHandler

error_handler = ErrorHandler()
correlation_id = error_handler.log_error(
    error=ValueError("Invalid order price"),
    extra_info={"order_id": "12345", "price": -100}
)
```

### Error Categories

- **Validation Errors**: Invalid order parameters (price, quantity, symbol)
- **Circuit Breaker Trips**: Market condition safety triggers
- **Recovery Errors**: Snapshot/restore failures
- **Connection Errors**: WebSocket disconnections and timeouts

## Input Validation

### Order Parameter Validation

All order parameters are validated before processing:

```python
from src.utils.validation import validate_price, validate_quantity, validate_symbol

# Validate order components
try:
    validate_price(order.price)      # Must be positive number
    validate_quantity(order.volume)  # Must be positive integer
    validate_symbol(order.symbol)    # Must be uppercase 1-10 chars
except OrderValidationError as e:
    # Handle validation error with detailed message
    pass
```

### Validation Rules

- **Price**: Must be positive float/int
- **Quantity**: Must be positive integer
- **Symbol**: Must be uppercase string (1-10 characters)

## Input Sanitization

### CSV Data Sanitization

```python
from src.utils.validation import sanitize_csv_row

# Clean CSV input
raw_row = ["AAPL", "100.50", "  200  ", "BUY,LIMIT"]
clean_row = sanitize_csv_row(raw_row)
# Result: ["AAPL", "100.50", "200", "BUYLIMIT"]
```

### API Response Sanitization

```python
from src.utils.validation import sanitize_api_payload

# Deep clean API responses
api_response = {
    "symbol": "  AAPL  ",
    "orders": [{"price": " 100.50 ", "qty": "200"}]
}
clean_response = sanitize_api_payload(api_response)
```

## Order Book State Recovery

### Creating Snapshots

```python
# Create snapshot of current order book state
success = order_book.create_snapshot()
if not success:
    logger.error("Failed to create order book snapshot")
```

### Restoring from Snapshots

```python
# Restore from most recent snapshot
success = order_book.restore_from_snapshot()
if success:
    logger.info("Order book restored successfully")
else:
    logger.error("Failed to restore order book")
```

### Manual Recovery

```python
from src.engine.recovery import OrderBookRecovery

# Manual snapshot creation
OrderBookRecovery.snapshot(order_book, correlation_id="manual_backup")

# Manual restore
restored_book = OrderBookRecovery.restore(correlation_id="manual_restore")
```

## Circuit Breakers

### Configuration

```python
from src.performance.circuit_breaker import CircuitBreaker

# Initialize with custom threshold
circuit_breaker = CircuitBreaker(threshold_pct=5)  # 5% price move limit
```

### Monitoring Market Conditions

```python
# Check before processing order
current_price = order.price
reference_price = last_trade_price or current_price
total_liquidity = bid_volume + ask_volume

if circuit_breaker.check(current_price, reference_price, total_liquidity):
    # Trading halted - reject order
    order.status = OrderStatus.REJECTED
    return []
```

### Reset Circuit Breaker

```python
# Reset after market conditions stabilize
circuit_breaker.reset()
```

## Structured Logging

### Correlation IDs

Every operation generates a unique correlation ID for tracing:

```python
# Automatic correlation ID generation
corr_id = error_handler.log_info("Processing order ABC123")

# Use correlation ID in subsequent operations
error_handler.log_warning("Order validation warning", corr_id)
```

### Log Levels

- **INFO**: Normal operations, successful connections
- **WARNING**: Circuit breaker trips, connection retries
- **ERROR**: Validation failures, snapshot errors

## WebSocket Reconnection

### Automatic Reconnection

```python
from src.utils.websocket_reconnect import WebSocketReconnector

reconnector = WebSocketReconnector(
    initial_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    max_retries=10
)

# Attempt connection with retry logic
async def connect_websocket():
    # Your WebSocket connection logic here
    return True

success = await reconnector.connect_with_retry(
    connect_func=connect_websocket,
    correlation_id="ws_reconnect"
)
```

### Exponential Backoff

Reconnection delays follow exponential backoff:
- Attempt 1: 1 second
- Attempt 2: 2 seconds
- Attempt 3: 4 seconds
- ...
- Max delay: 60 seconds

## Graceful Degradation

### Performance Monitoring

```python
# Monitor system performance
if error_rate > threshold:
    # Switch to safer, slower algorithms
    use_simple_matching = True
    disable_advanced_features = True
```

### Fallback Strategies

- **High Error Rate**: Disable complex order types
- **Low Memory**: Reduce order book depth
- **High Latency**: Switch to batch processing

## Testing Error Handling

### Running Tests

```bash
# Run error handling test suite
python -m pytest tests/test_error_handling.py -v
```

### Test Categories

- **Validation Tests**: Input parameter validation
- **Circuit Breaker Tests**: Price move and liquidity triggers
- **Recovery Tests**: Snapshot/restore functionality
- **Error Handler Tests**: Logging and correlation IDs

## Troubleshooting

### Common Issues

1. **OrderValidationError**: Check order parameters (price > 0, quantity > 0, valid symbol)
2. **Circuit Breaker Tripped**: Large price move or zero liquidity detected
3. **Snapshot Failed**: Check disk space and file permissions
4. **WebSocket Disconnected**: Network issues or server maintenance

### Debug Steps

1. Check correlation ID in logs for error tracing
2. Verify order parameters meet validation requirements
3. Confirm circuit breaker thresholds are appropriate
4. Test recovery mechanisms with known good snapshots

### Log Analysis

```bash
# Search logs by correlation ID
grep "CorrID: abc123" application.log

# Find circuit breaker events
grep "Circuit breaker tripped" application.log

# Monitor error rates
grep "ERROR" application.log | wc -l
```

---

## Integration Examples

### Complete Error Handling Setup

```python
from src.engine.order_book import OrderBook
from src.utils.error_handler import ErrorHandler
from src.performance.circuit_breaker import CircuitBreaker

# Initialize with error handling
book = OrderBook("AAPL")
error_handler = ErrorHandler()

# Process order with full error handling
try:
    trades = book.add_order(order)
except ValueError as e:
    corr_id = error_handler.log_error(e, extra_info={"order_id": order.order_id})
    # Handle error appropriately
```

For further customization or integration, refer to code comments in respective modules.

