#!/usr/bin/env python3
"""
Fetch OHLCV Data
----------------
Downloads 1-minute OHLCV data for MAGS ETF using yfinance.

Fetches:
- Ticker: MAGS
- Interval: 1 minute
- Period: 8 days (maximum allowed by yfinance for 1m data)

Output:
- data_files/raw_mags_1m.csv (OHLCV data in UTC timezone)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

print("Starting OHLCV download...")
ticker = yf.Ticker("MAGS")
data = ticker.history(period="8d", interval="1m")
print(f"Downloaded {len(data)} rows")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# Format: keep only OHLCV columns, lowercase names
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
data.columns = [col.lower() for col in data.columns]
data.index = data.index.tz_convert('UTC')
data.index.name = 'Datetime'

print(f"Final shape: {data.shape}")
output_path = "data_files/raw_mags_1m.csv"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output_path)
print(f"Saved to {output_path}")
