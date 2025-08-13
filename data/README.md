# CSV Data for Backtesting

This folder contains sample CSV files with historical market data for backtesting the HFT trading simulator.

## Data Format

All CSV files follow the same structure with three columns:
- `timestamp`: Date and time in YYYY-MM-DD HH:MM:SS format
- `price`: Asset price as a float (e.g., 45000.50)
- `volume`: Trading volume as a float (e.g., 1.234)

### Example Structure:
```csv
timestamp,price,volume
2024-01-01 00:00:00,45000.50,1.234
2024-01-01 00:00:01,45001.00,0.567
2024-01-01 00:00:02,45000.75,2.145
```

## Available Data Files

1. **BTCUSDT_sample.csv** - Bitcoin minute-by-minute data (60 seconds)
2. **ETHUSDT_sample.csv** - Ethereum minute-by-minute data (60 seconds)
3. **AAPL_sample.csv** - Apple stock minute-by-minute data (60 seconds)
4. **BTCUSDT_extended.csv** - Bitcoin extended data (24+ hours with 15-minute intervals)

## Usage Instructions

1. **Loading Data**: Use pandas to load CSV files:
   ```python
   import pandas as pd
   df = pd.read_csv('data/BTCUSDT_sample.csv')
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   ```

2. **Data Validation**: Ensure data quality:
   ```python
   # Check for missing values
   print(df.isnull().sum())
   
   # Check data types
   print(df.dtypes)
   
   # Verify timestamp ordering
   print(df['timestamp'].is_monotonic_increasing)
   ```

3. **Backtesting Integration**: 
   - Set the CSV file path in your backtesting configuration
   - Ensure timestamp format matches your system requirements
   - Verify price and volume ranges are realistic for the asset

## Data Quality Notes

- **Timestamps**: All timestamps are in sequential order
- **Price Movement**: Prices include realistic small movements typical of HFT scenarios
- **Volume**: Volumes vary to simulate different market conditions
- **No Gaps**: Data contains no missing timestamps within each file's timeframe

## Extending Data

To add more data or different assets:

1. Follow the same CSV format (timestamp,price,volume)
2. Ensure timestamps are properly formatted and sequential
3. Use realistic price movements and volume patterns
4. Name files descriptively (e.g., SYMBOL_timeframe.csv)

## Real Data Sources

For production use, consider downloading real historical data from:
- Binance API (cryptocurrency)
- Alpha Vantage (stocks)
- Yahoo Finance (stocks)
- IEX Cloud (stocks)
- Quandl (various assets)

Remember to respect API rate limits and terms of service when downloading real market data.
