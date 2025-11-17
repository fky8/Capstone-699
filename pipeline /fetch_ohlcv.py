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
- data_files/master_raw_mags_1m.csv (Aggregated OHLCV data in UTC timezone)
- data_files/raw_mags_1m_YYYYMMDD_to_YYYYMMDD.csv (Latest 8-day fetch with date span)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

print("Starting OHLCV download...")
ticker = yf.Ticker("MAGS")
new_data = ticker.history(period="8d", interval="1m")
print(f"Downloaded {len(new_data)} rows")
print(f"Date range: {new_data.index[0]} to {new_data.index[-1]}")

# Format: keep only OHLCV columns, lowercase names
new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
new_data.columns = [col.lower() for col in new_data.columns]
new_data.index = new_data.index.tz_convert('UTC')
new_data.index.name = 'Datetime'

# Create dated filename for the new fetch
start_date = new_data.index[0].strftime('%Y%m%d')
end_date = new_data.index[-1].strftime('%Y%m%d')
dated_filename = f"data_files/raw_mags_1m_{start_date}_to_{end_date}.csv"
Path(dated_filename).parent.mkdir(parents=True, exist_ok=True)

# Save the new fetch with date span in filename
new_data.to_csv(dated_filename)
print(f"\nSaved latest fetch to {dated_filename}")

# Master file path
master_path = "data_files/master_raw_mags_1m.csv"

# Load existing master data if it exists
if Path(master_path).exists():
    print(f"\nLoading existing master file...")
    master_data = pd.read_csv(master_path, index_col=0, parse_dates=True)
    print(f"Existing data: {len(master_data)} rows ({master_data.index[0]} to {master_data.index[-1]})")
    
    # Combine and remove duplicates (keep latest)
    combined = pd.concat([master_data, new_data])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    new_rows = len(combined) - len(master_data)
    print(f"Added {new_rows} new rows")
    print(f"Master data now: {len(combined)} rows ({combined.index[0]} to {combined.index[-1]})")
else:
    print(f"\nCreating new master file...")
    combined = new_data
    print(f"Master data: {len(combined)} rows")

# Save master file
combined.to_csv(master_path)
print(f"Saved to {master_path}")
