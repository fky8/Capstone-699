# High-Frequency Regime Detection for MAGS ETF

**Volatility regime detection using 1-minute data, news sentiment, and GARCH forecasting.**

---

## Overview

This pipeline detects volatility regimes for the MAGS ETF (Magnificent Seven equal-weight) using:
- 1-minute OHLCV data (yfinance)
- Real-time news collection (Finnhub API, continuous updates)
- FinBERT sentiment analysis
- GARCH(1,1) conditional volatility forecasting  
- 3-of-5 Forward Voting regime labeling

**Current Data (as of Nov 8, 2025):**
- 1,942 samples (Nov 3-7, 2025) - 5 trading days
- 17 features (13 news + 3 GARCH + 1 label)
- Final dataset in Eastern Time format

---

## Project Structure

```
regime-rotation-mvp/
├── pipeline/                       # All data pipeline scripts
│   ├── fetch_ohlcv.py             # Fetch 1-min OHLCV from yfinance
│   ├── news_data.py               # Fetch news from Finnhub API
│   ├── news_sentiment.py          # FinBERT sentiment analysis
│   ├── news_features.py           # Aggregate news features
│   ├── garch_features.py          # GARCH(1,1) volatility forecasting
│   ├── create_labels.py           # 3-of-5 voting regime labels
│   ├── combine_features.py        # Combine all features + filter by date
│   ├── config_hf.yaml             # Configuration file
│   ├── requirements_hf.txt        # Python dependencies
│   ├── Makefile                   # Simple build automation
│   ├── run_pipeline.sh            # Complete pipeline script
│   └── data_files/                # All generated CSV files
│       ├── raw_mags_1m.csv        # OHLCV data
│       ├── news_raw.csv           # Raw news articles
│       ├── news_sentiment.csv     # Sentiment scores
│       ├── features_news.csv      # News features
│       ├── features_garch.csv     # GARCH features
│       ├── labels.csv             # Regime labels
│       └── features_combined.csv  # FINAL OUTPUT
│
├── scripts/                       # Background news collector
│   └── continuous_collector.py    # Runs every 30 minutes
│
└── .env                           # API keys (FINNHUB_API_KEY)
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd pipeline
pip install -r requirements_hf.txt
```

### 2. Set API Key

```bash
export FINNHUB_API_KEY="finnhub_api_key"
```

Get API key at https://finnhub.io/register

### 3. Run Complete Pipeline

**Using Makefile:**
```bash
make        # Run complete pipeline
make clean  # Remove all CSV files
make rebuild # Clean and rebuild everything
```

**Manual Execution:**
```bash
python fetch_ohlcv.py                    # Step 1: OHLCV data (8 days max)
python news_data.py --config config_hf.yaml  # Step 2: News data
python news_sentiment.py --config config_hf.yaml  # Step 3: Sentiment (~2 min)
python news_features.py --config config_hf.yaml   # Step 4: News features
python garch_features.py --config config_hf.yaml  # Step 5: GARCH (~3 min)
python create_labels.py --config config_hf.yaml   # Step 6: Labels
python combine_features.py               # Step 7: Combine + filter
```

---

## Data Sources

### 1. Price Data
- **Asset**: MAGS ETF (Magnificent Seven equal-weight)
- **Frequency**: 1-minute bars
- **Period**: Past 8 days (yfinance limitation: only retrieve the past 8 days)
- **Features**: open, high, low, close, volume

### 2. News Data
- **Source**: Finnhub API
- **Companies**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Lookback**: 7 days (API window)
- **Collection**: Continuous updates every 30 minutes
- **Features**: headline, summary, datetime, source

### 3. Sentiment Analysis
- **Model**: FinBERT (ProsusAI/finbert)
- **Output**: Positive, neutral, negative probabilities + net sentiment
- **Aggregation**: Minute-level rolling windows (5m, 15m, 60m)

---

## Features Generated

### News Features (13)
- `news_cnt_5m`, `news_cnt_15m`, `news_cnt_60m` - News count in windows
- `sent_sum_pos_5m`, `sent_sum_pos_15m`, `sent_sum_pos_60m` - Positive sentiment sum
- `sent_sum_neg_5m`, `sent_sum_neg_15m`, `sent_sum_neg_60m` - Negative sentiment sum
- `sent_net_5m`, `sent_net_15m`, `sent_net_60m` - Net sentiment (pos - neg)
- `sent_ewm_60m` - Exponentially weighted moving average sentiment

### GARCH Features (3)
- `pred_vol` - GARCH(1,1) **5-bar ahead** forecast (predictive)
- `hist_vol` - 30-bar rolling historical volatility (baseline)
- `vol_ratio` - pred_vol / hist_vol ratio

### Labels (1)
- `regime_label` - Low/Neutral/High/Unknown (3-of-5 voting method)

---

## Regime Labeling: 3-of-5 Forward Voting

**Forward-looking regime detection with majority voting:**

### Algorithm
1. **Baseline**: 30-bar EWMA realized volatility (smooth baseline)
2. **Forward**: 5-bar forward realized volatility (ground truth)
3. **Ratio**: `q_t = forward_vol / baseline_vol`
4. **Voting Window**: 5 consecutive ratios `{q_t, q_{t+1}, ..., q_{t+4}}`
5. **Thresholds**:
   - High: `q ≥ 1.2` (future vol 20%+ above baseline)
   - Low: `q ≤ 0.8` (future vol 20%+ below baseline)
   - Neutral: `0.8 < q < 1.2`
6. **Decision**: Need 3+ votes (60% majority)
7. **Min-Hold**: 5 bars before transition (prevents oscillation)
8. **Latency**: 9 bars (decision for bar t available at t+9)

### Results
- **Distribution**: 54.5% Low, 29.8% Neutral, 15.3% High, 0.4% Unknown

---

## Output Format

**All features combined CSV**: `pipeline/data_files/features_combined.csv`

**Specifications:**
- **Samples**: 1,942 rows
- **Date Range**: Nov 3, 2025 09:30 AM - Nov 7, 2025 3:59 PM (Eastern Time)
- **Features**: 17 columns (13 news + 3 GARCH + 1 label)
- **Index**: DatetimeIndex in ET format
- **Frequency**: 1-minute bars during market hours (9:30 AM - 4:00 PM ET)

**Sample rows:**
```
Datetime,news_cnt_5m,news_cnt_15m,...,pred_vol,hist_vol,vol_ratio,regime_label
2025-11-03 09:30:00,0.0,0.0,...,0.331,0.726,0.456,Low
2025-11-03 09:31:00,0.0,0.0,...,3.831,0.727,5.269,Low
...
2025-11-07 15:59:00,0.0,0.0,...,0.546,0.222,2.462,Unknown
```

---

## Configuration

Edit `pipeline/config_hf.yaml`:

### Key Settings
```yaml
data:
  ticker: MAGS
  interval: 1m
  lookback_days: 7
  news_lookback_days: 7

features:
  news:
    count_windows: [5, 15, 60]      # Minutes
    sentiment_windows: [5, 15, 60]  # Minutes
    ewm_halflife: 60                # EWMA half-life
  
  garch:
    p: 1
    q: 1
    min_observations: 100
    forecast_horizon: 5              # 5-bar ahead forecast
    vol_ratio_window: 30

labels_voting:
  baseline_halflife: 30    # EWMA baseline (30 bars)
  forward_window: 5        # Voting window (5 bars)
  high_threshold: 1.2      # High if ratio ≥ 1.2
  low_threshold: 0.8       # Low if ratio ≤ 0.8
  min_hold: 5              # Min bars before transition
  votes_needed: 3          # 3-of-5 majority
```

---

## Pipeline Automation

### Makefile
```bash
make        # Run complete pipeline
make clean  # Remove all CSV files
make rebuild # Clean and rebuild everything
make help   # Show help
```
---

## Data Limitations

### Critical: API Constraints

**yfinance OHLCV**: Maximum 8 days of 1-minute data
   - Re-run `fetch_ohlcv.py` regularly to aggreagte data if target >8 days dataset
   - Current data expires when it falls outside 8-day window


---

## Future Enhancements

1. **Technical features**: Add back if needed (returns, momentum, volatility ratios)
2. **Model training**: XGBoost classifier for regime prediction
3. **Real-time inference**: WebSocket streaming with live predictions
4. **Multi-horizon**: Forecast regimes at 5min, 15min, 1hr horizons
5. **Additional data**: Options flow, social media sentiment, macro indicators
6. **Backtesting**: Simulate trading strategies based on regime predictions
