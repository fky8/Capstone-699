# High-Frequency Regime Detection for MAGS ETF

**Volatility regime detection using 1-minute data, news sentiment, and GARCH forecasting.**

---

## Overview

This pipeline detects volatility regimes for the MAGS ETF (Magnificent Seven equal-weight) using:
- 1-minute OHLCV data (yfinance)
- News collection (Finnhub API)
- FinBERT sentiment analysis
- Trump Truth Social posts (FactBase API)
- Twitter-RoBERTa sentiment analysis
- GARCH(1,1) conditional volatility forecasting  
- 3-of-5 Forward Voting regime labeling

**Current Data (as of Dec 7, 2025):**
- 8,722 samples (Nov 4 - Dec 5, 2025) - 23 trading days
- 30 features (13 news + 3 GARCH + 13 Trump + 1 label)
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
│   ├── trump_data.py              # Fetch Trump Truth Social posts
│   ├── trump_sentiment.py         # Twitter-RoBERTa sentiment analysis
│   ├── trump_features.py          # Aggregate Trump features
│   ├── garch_features.py          # GARCH(1,1) volatility forecasting
│   ├── create_labels.py           # 3-of-5 voting regime labels
│   ├── combine_features.py        # Combine all features + filter by date
│   ├── restore_datasets.py        # Backup/restore dataset utility
│   ├── config_hf.yaml             # Configuration file
│   ├── requirements_hf.txt        # Python dependencies
│   ├── Makefile                   # Simple build automation
│   ├── run_pipeline.sh            # Complete pipeline script
│   └── data_files/                # All generated CSV files
│       ├── master_raw_mags_1m.csv # Aggregated OHLCV data (all history)
│       ├── raw_mags_1m_YYYYMMDD_to_YYYYMMDD.csv # Dated fetch archives
│       ├── news_raw.csv           # Raw news articles
│       ├── news_sentiment.csv     # Sentiment scores
│       ├── features_news.csv      # News features
│       ├── trump_raw.csv          # Raw Trump posts
│       ├── trump_sentiment.csv    # Trump sentiment scores
│       ├── features_trump.csv     # Trump features
│       ├── features_garch.csv     # GARCH features
│       ├── labels.csv             # Regime labels
│       └── features_combined.csv  # FINAL OUTPUT
│
└── .env                           # API keys (FINNHUB_API_KEY)
```

---

## Quick Start

### 1. Install Dependencies

### 2. Set API Key

```bash
export FINNHUB_API_KEY="finnhub_api_key"
```

Get API key at https://finnhub.io/register

### 3. Run Pipeline

**Use Makefile:**
```bash
make        # Run complete pipeline
make clean  # Remove all CSV files
make rebuild # Clean and rebuild everything
```

---

## Data Sources

### 1. Price Data
- **Asset**: MAGS ETF (Magnificent Seven equal-weight)
- **Frequency**: 1-minute bars
- **Period**: Past 8 days per fetch (yfinance limitation)
- **Aggregation**: Automatically appended to `master_raw_mags_1m.csv` with duplicate removal
- **Archive**: Each fetch saved as `raw_mags_1m_YYYYMMDD_to_YYYYMMDD.csv` with date span
- **Features**: open, high, low, close, volume

### 2. News Data
- **Source**: Finnhub API
- **Companies**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Lookback**: 7 days (API window)
- **Features**: headline, summary, datetime, source

### 3. Sentiment Analysis
- **News Model**: FinBERT (ProsusAI/finbert)
- **Trump Model**: Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Output**: Positive, neutral, negative probabilities + net sentiment
- **Aggregation**: Minute-level rolling windows (5m, 15m, 60m)

### 4. Trump Truth Social Data
- **Source**: FactBase API (https://rollcall.com/wp-json/factbase/v1/twitter)
- **Content**: Trump's Truth Social posts (2025 data)
- **Lookback**: 30 days (API window)
- **Filtering**: Excludes media-only posts (images, videos without text)
- **Features**: content, datetime, engagement metrics

---

## Features Generated

### News Features (13) - BAR-BASED WINDOWS
- `news_cnt_5m`, `news_cnt_15m`, `news_cnt_60m` - News count in last N trading bars
- `sent_sum_pos_5m`, `sent_sum_pos_15m`, `sent_sum_pos_60m` - Positive sentiment sum
- `sent_sum_neg_5m`, `sent_sum_neg_15m`, `sent_sum_neg_60m` - Negative sentiment sum
- `sent_net_5m`, `sent_net_15m`, `sent_net_60m` - Net sentiment (pos - neg)
- `sent_ewm_60m` - Exponentially weighted moving average sentiment (60-bar half-life)

**Note**: News features use bar-based windows (e.g., 60m = last 60 trading bars) because news is a discrete event stream that only occurs during market hours.

### Trump Features (13) - TIME-BASED WINDOWS
- `trump_cnt_5m`, `trump_cnt_15m`, `trump_cnt_60m` - Trump post count in last N actual minutes
- `trump_pos_5m`, `trump_pos_15m`, `trump_pos_60m` - Positive sentiment sum
- `trump_neg_5m`, `trump_neg_15m`, `trump_neg_60m` - Negative sentiment sum
- `trump_net_5m`, `trump_net_15m`, `trump_net_60m` - Net sentiment (pos - neg)
- `trump_ewm_60m` - Exponentially weighted moving average sentiment (60-minute half-life)

**Note**: Trump features use time-based windows (e.g., 60m = actual 60 minutes of clock time) because Truth Social is a continuous 24/7 stream that includes pre-market, post-market, and overnight activity. This captures the full temporal dynamics of social media sentiment.

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
- **Distribution**: 52.3% Low, 31.1% Neutral, 16.3% High, 0.2% Unknown

---

## Output Format

**All features combined CSV**: `pipeline/data_files/features_combined.csv`

**Specifications:**
- **Samples**: 5,962 rows
- **Date Range**: Nov 4, 2025 09:30 AM - Nov 25, 2025 03:59 PM (Eastern Time)
- **Features**: 30 columns (13 news + 13 Trump + 3 GARCH + 1 label)
- **Index**: DatetimeIndex in ET format
- **Frequency**: 1-minute bars during market hours (9:30 AM - 4:00 PM ET)

**Sample rows:**
```
Datetime,news_cnt_5m,news_cnt_15m,...,trump_cnt_5m,trump_cnt_15m,...,pred_vol,hist_vol,vol_ratio,regime_label
2025-11-04 09:30:00,0.0,0.0,...,0.0,0.0,...,0.331,0.726,0.456,Low
2025-11-04 09:31:00,0.0,0.0,...,0.0,0.0,...,3.831,0.727,5.269,Low
...
2025-11-25 15:59:00,0.0,1.0,...,0.0,0.0,...,0.546,0.222,2.462,High
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
    count_windows: [5, 15, 60]      # Trading bars (bar-based)
    sentiment_windows: [5, 15, 60]  # Trading bars (bar-based)
    ewm_halflife: 60                # EWMA half-life (bars)
  
  trump:
    count_windows: [5, 15, 60]      # Actual minutes (time-based)
    sentiment_windows: [5, 15, 60]  # Actual minutes (time-based)
    ewm_halflife: 60                # EWMA half-life (minutes)
  
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

## Data Collection Flow

### Automated Aggregation Pipeline

**Each pipeline run:**

1. **Fetch**: `fetch_ohlcv.py` downloads 8-day OHLCV data from yfinance
2. **Archive**: Saves fetch as `raw_mags_1m_{start_date}_to_{end_date}.csv` 
3. **Aggregate**: Appends new data to `master_raw_mags_1m.csv`
4. **Deduplicate**: Removes duplicate timestamps (keeps latest)
5. **Process**: All feature scripts (`news_features.py`, `garch_features.py`, `create_labels.py`) read from master file
6. **Filter**: `combine_features.py` applies Nov 4, 2025 cutoff for final dataset

**File Organization:**
- `master_raw_mags_1m.csv` - Full aggregated history (Oct 29 - Nov 25: 7,517 rows)
- `raw_mags_1m_YYYYMMDD_to_YYYYMMDD.csv` - Dated fetch archives for reference
- `features_combined.csv` - Final filtered dataset (Nov 4 - Nov 25: 5,962 rows)

**Benefits:**
- Overcomes yfinance 8-day limitation through continuous aggregation
- Dated archives provide audit trail of data collection
- No manual file management needed
- Automatic duplicate removal ensures data integrity

---

## Data Limitations

### API Constraints

**yfinance OHLCV**: Maximum 8 days per fetch
   - Pipeline automatically aggregates across runs
   - Run regularly (daily/hourly) to maintain continuous history
   - Master file grows indefinitely with each fetch


---

## Future Enhancements

1. **Technical features**: Add back if needed (returns, momentum, volatility ratios)
2. **Model training**: XGBoost classifier for regime prediction
3. **Real-time inference**: WebSocket streaming with live predictions
4. **Multi-horizon**: Forecast regimes at 5min, 15min, 1hr horizons
5. **Additional data**: Options flow, macro indicators
6. **Backtesting**: Simulate trading strategies based on regime predictions
7. **Correlation analysis**: Evaluate Trump sentiment → MAGS returns relationship

---

## Trump Feature Integration

### Overview
The pipeline now includes Trump Truth Social sentiment as a real-time market signal. Trump's social media activity has historically correlated with market volatility, particularly for tech stocks in the MAGS ETF.

### Implementation Details

**Data Collection** (`trump_data.py`):
- Source: FactBase WordPress API
- Endpoint: https://rollcall.com/wp-json/factbase/v1/twitter
- Data: Trump's Truth Social posts from 2025
- Filtering: Excludes media-only posts (images/videos without text)
- Fields: content, datetime, engagement metrics (likes, comments, shares)

**Sentiment Analysis** (`trump_sentiment.py`):
- Model: Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- Rationale: Optimized for social media text, better than FinBERT for informal language
- Batch processing: 16 posts at a time
- Output: positive, neutral, negative probabilities + net_sentiment

**Feature Engineering** (`trump_features.py`):
- 13 features across 5m, 15m, 60m windows
- TIME-BASED windows: Uses actual clock time [T-60min, T), not trading bars
- Captures pre-market, post-market, overnight activity
- Features:
  * Count: Number of posts in window
  * Sentiment sums: Positive, negative, net (pos - neg)
  * EWMA: Exponentially weighted moving average (60-min half-life)

### Bar-Based vs Time-Based Windows

**Design Decision**: Different window types for different data sources

**News Features (Bar-Based)**:
- Window: Last N trading bars (e.g., 60m = last 60 bars)
- Rationale: News is a discrete event stream during market hours only
- Behavior: 9:30 AM bar looks back 60 bars (~60 minutes on same/previous day)
- Advantage: Captures carryover effects from previous trading day

**Trump Features (Time-Based)**:
- Window: Last N actual minutes (e.g., 60m = 8:30-9:30 AM actual time)
- Rationale: Social media is a continuous 24/7 stream
- Behavior: 9:30 AM bar looks back to 8:30 AM same day (includes pre-market)
- Advantage: Captures overnight sentiment shifts and pre-market reactions

This heterogeneity is intentional - it respects the fundamental nature of each data source and allows the model to learn different temporal dynamics.

### Quick Usage

**Running Trump Pipeline:**
```bash
# Fresh start (collect new data)
cd pipeline
python trump_data.py --start 2025-10-29
python trump_sentiment.py
python trump_features.py

# One-line (if data exists)
python trump_data.py && python trump_sentiment.py && python trump_features.py
```

**Dataset Management:**
```bash
# Check status
python restore_datasets.py --status

# Create backup
python restore_datasets.py --backup
python restore_datasets.py --backup --tag "before_experiment"

# Restore to original state
python restore_datasets.py --restore
python restore_datasets.py --restore 20251125_123456

# Clean Trump files only
python restore_datasets.py --clean-trump
```

**Regenerate with Different Dates:**
```bash
python restore_datasets.py --clean-trump
python trump_data.py --start 2025-11-01 --end 2025-11-15
python trump_sentiment.py
python trump_features.py
```

### Pipeline Integration

**Makefile targets**:
```bash
make trump_data       # Collect Trump posts
make trump_sentiment  # Run sentiment analysis
make trump_features   # Generate features
make combine          # Combine all features
make all              # Run complete pipeline
```

**Pipeline order**:
1. fetch_ohlcv
2. news_data → news_sentiment → news_features
3. garch_features
4. trump_data → trump_sentiment → trump_features
5. create_labels
6. combine_features

### Timezone Handling

All data maintained in UTC throughout pipeline:
- Price data: UTC (from yfinance)
- Trump posts: UTC (collected in UTC, aligned in UTC)
- Features: Generated in UTC
- Final output: Converted to Eastern Time for user convenience

This ensures proper alignment when filtering Trump posts by time windows.

### Coverage Statistics

From Nov 4-25, 2025 dataset (5,962 bars):
- Trump posts collected: 509
- Posts aligned to market hours: 497
- Bars with non-zero 60m count: 28.4%
- Bars with EWMA sentiment: 100% (exponential decay persists)
- Sentiment distribution: 46.2% positive, 29.4% neutral, 24.1% negative

### Troubleshooting

**"No posts collected":**
- Check date range matches MAGS data period
- Verify FactBase API is accessible
- Try reducing `--max-pages` if timeout occurs

**"No backups found":**
- Run `python restore_datasets.py --backup` first
- Check `data_files/backups/` directory exists

**Timezone errors:**
- All data stored in UTC, converted to ET for output
- Ensure pandas and pytz are up to date: `pip install -U pandas pytz`

**Feature alignment issues:**
- Trump features align to MAGS minute bars automatically
- Pre-market news attributed to market open (9:30 AM)
- Verify price data has proper timezone-aware datetime index

---