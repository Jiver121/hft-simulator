# HFT Simulator - Batch Backtesting System

This document explains how to use the comprehensive batch backtesting system for the HFT Simulator.

## Overview

The batch backtesting system allows you to:
- Run backtests on single CSV files or entire directories
- Filter data by date ranges
- Use different trading strategies (market making, momentum)
- Generate comprehensive performance reports
- Save results in JSON format for further analysis

## Quick Start

### 1. Create Sample Data (for testing)

```bash
python main.py --mode create-sample-data
```

This will create sample market data in the `./data/` directory with files for AAPL, MSFT, and GOOGL.

### 2. Run Single File Backtest

```bash
python main.py --mode backtest --data ./data/aapl_data.csv --output ./logs/backtest_results.json
```

### 3. Run Batch Backtest on Directory

```bash
python main.py --mode backtest --data ./data/ --output ./logs/
```

### 4. Run with Date Range Filter

```bash
python main.py --mode backtest --data ./data/ --start-date 2024-01-01 --end-date 2024-01-31 --output ./logs/
```

## Command Line Options

### Required Arguments

- `--mode`: Execution mode (`backtest` or `create-sample-data`)
- `--data`: Path to CSV file or directory (required for backtest mode)
- `--output`: Output path for results (required for backtest mode)

### Optional Arguments

- `--start-date`: Start date for filtering data (YYYY-MM-DD format)
- `--end-date`: End date for filtering data (YYYY-MM-DD format)
- `--strategy`: Trading strategy to use (`market_making` or `momentum`)
- `--config`: Path to configuration JSON file
- `--parallel`: Enable parallel processing (future feature)
- `--workers`: Number of parallel workers (default: 1)
- `--verbose`: Enable verbose logging

## Trading Strategies

### Market Making Strategy
- Places buy and sell orders around the current market price
- Captures bid-ask spread while managing inventory risk
- Parameters:
  - `spread_bps`: Spread in basis points (default: 10.0)
  - `order_size`: Size of each order (default: 100)
  - `max_inventory`: Maximum inventory position (default: 500)

### Momentum Strategy
- Buys on positive price momentum, sells on negative momentum
- Uses recent price history to identify trends
- Parameters:
  - `momentum_threshold`: Minimum return threshold (default: 0.001)
  - `order_size`: Size of each order (default: 200)
  - `max_positions`: Maximum number of positions (default: 5)

## Configuration

You can customize the backtesting parameters using a JSON configuration file:

```json
{
  "strategy_type": "market_making",
  "initial_capital": 100000.0,
  "commission_rate": 0.0005,
  "slippage_bps": 1.0,
  "max_position_size": 1000,
  "max_order_size": 100,
  "risk_limit": 10000.0,
  "tick_size": 0.01,
  "fill_model": "realistic",
  "enable_logging": true,
  "save_snapshots": false,
  "strategy_params": {
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
  }
}
```

Use it with:
```bash
python main.py --mode backtest --data ./data/ --output ./logs/ --config ./config/backtest_config.json
```

## CSV Data Format

Your CSV files should contain the following columns:

### Required Columns
- `timestamp`: Trade timestamp (ISO format or pandas-parseable)
- `symbol`: Trading symbol
- `price`: Trade price
- `volume`: Trade volume

### Optional Columns
- `bid`: Best bid price
- `ask`: Best ask price
- `bid_volume`: Bid volume
- `ask_volume`: Ask volume
- `trade_type`: Type of trade (e.g., 'trade', 'quote')

### Example CSV Format
```csv
timestamp,symbol,price,volume,bid,ask,bid_volume,ask_volume,trade_type
2024-01-01 09:30:00,AAPL,150.00,1000,149.98,150.02,500,600,trade
2024-01-01 09:30:01,AAPL,150.01,800,149.99,150.03,400,700,trade
```

## Output Files

The system generates several types of output files:

### 1. Individual Result Files (when output is a directory)
- `{symbol}_backtest_{timestamp}.json`: Individual backtest results
- Contains detailed trading results, performance metrics, and configuration

### 2. Consolidated Results (when output is a JSON file)
- Single JSON file with all results
- Includes summary statistics and individual results

### 3. Summary Report
- `backtest_summary.json`: High-level performance summary
- Aggregated statistics across all backtests
- Console output with key metrics

## Example Usage Scenarios

### 1. Quick Test with Sample Data
```bash
# Create sample data
python main.py --mode create-sample-data

# Run backtest on all sample files
python main.py --mode backtest --data ./data/ --output ./logs/ --verbose
```

### 2. Historical Analysis
```bash
# Run backtest on historical data for specific date range
python main.py --mode backtest \
  --data ./historical_data/ \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --output ./results/january_2024/ \
  --strategy market_making \
  --config ./config/aggressive_mm.json
```

### 3. Strategy Comparison
```bash
# Market making strategy
python main.py --mode backtest --data ./data/ --strategy market_making --output ./results/mm/

# Momentum strategy
python main.py --mode backtest --data ./data/ --strategy momentum --output ./results/momentum/
```

### 4. Single File Analysis
```bash
# Analyze specific stock
python main.py --mode backtest \
  --data ./data/AAPL_2024.csv \
  --output ./results/aapl_detailed.json \
  --verbose
```

## Performance Metrics

The system calculates comprehensive performance metrics:

### Trading Metrics
- Total P&L
- Number of trades
- Win rate
- Fill rate
- Average trade size

### Risk Metrics
- Maximum drawdown
- Sharpe ratio (when benchmark data available)
- Portfolio volatility
- Risk-adjusted returns

### Execution Metrics
- Average latency
- Slippage analysis
- Order execution statistics

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the `src/` directory is in your Python path
2. **Data Format Issues**: Check that your CSV files have the required columns
3. **Date Format Problems**: Use ISO format dates (YYYY-MM-DD) for filtering
4. **Memory Issues**: For large datasets, consider processing files individually

### Debug Mode
Use `--verbose` flag to enable detailed logging:
```bash
python main.py --mode backtest --data ./data/ --output ./logs/ --verbose
```

### Log Files
Check the logs directory for detailed execution logs and error messages.

## Integration with Existing System

This batch backtesting system integrates seamlessly with the existing HFT Simulator components:

- **ExecutionSimulator**: Core backtesting engine
- **DataIngestion**: CSV data loading and preprocessing
- **Fill Models**: Realistic and perfect execution models
- **Strategy Framework**: Extensible strategy system
- **Performance Analytics**: Comprehensive metrics calculation

## Next Steps

After running backtests, you can:

1. **Analyze Results**: Use the JSON output files for detailed analysis
2. **Visualize Performance**: Import results into notebooks for charting
3. **Parameter Optimization**: Iterate on strategy parameters
4. **Production Deployment**: Use validated strategies in live trading

## Support

For issues or questions:
1. Check the console output and log files
2. Verify CSV data format
3. Test with sample data first
4. Use verbose mode for debugging
