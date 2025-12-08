"""
Trump social media feature engineering for high-frequency regime detection.
Aggregates Trump post sentiment into minute-level features.

Mirrors news_features.py structure but for Trump Truth Social posts.
"""

from typing import List, Optional
import yaml

import pandas as pd
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def align_trump_to_minutes(
    trump_df: pd.DataFrame,
    price_df: pd.DataFrame,
    target_tz: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Align Trump post timestamps to minute bars.
    
    Round UP to next minute to ensure price data can reflect the post impact.
    Example: Post at 9:45:30 → aligned to 9:46:00 bar
    
    IMPORTANT: Both trump_df and price_df must be in the SAME timezone for proper comparison.
    The function keeps datetime_minute in the same timezone as price_df.index.
    
    Args:
        trump_df: Trump DataFrame with datetime in UTC
        price_df: Price DataFrame with datetime index (typically UTC)
        target_tz: Target timezone (for reference, not used in comparison)
    
    Returns:
        Trump DataFrame aligned to price bars with datetime_minute in same timezone as price_df
    """
    import pytz
    
    trump_df = trump_df.copy()
    
    # Ensure datetime is timezone-aware
    if trump_df['datetime'].dt.tz is None:
        trump_df['datetime'] = trump_df['datetime'].dt.tz_localize('UTC')
    
    # Get the timezone of price data index
    price_tz = price_df.index.tz if price_df.index.tz is not None else pytz.UTC
    
    # Convert to price data timezone
    trump_df_in_price_tz = trump_df['datetime'].dt.tz_convert(price_tz)
    
    # Round UP to next minute (in the same timezone as price data)
    trump_df['datetime_minute'] = trump_df_in_price_tz.dt.ceil('min')
    
    # Keep all posts within the price data range
    # Now both are in the same timezone, comparison is valid
    min_dt = price_df.index.min()
    max_dt = price_df.index.max()
    trump_df = trump_df[(trump_df['datetime_minute'] >= min_dt) & (trump_df['datetime_minute'] <= max_dt)]
    
    return trump_df


def compute_trump_counts(
    trump_df: pd.DataFrame,
    price_df: pd.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """
    Compute rolling Trump post counts using ACTUAL TIME windows (not bar-based).
    
    For each bar at time T, counts posts in the time window [T-window, T).
    Example: At 9:30 AM, 60m window counts posts from 8:30-9:30 AM (real time).
    
    This differs from news features which use bar-based rolling windows.
    Trump posts occur 24/7, so we need real-time windows.
    
    Args:
        trump_df: Trump DataFrame with datetime_minute (aligned timestamps)
        price_df: Price DataFrame with datetime index
        windows: List of window sizes in minutes (default [5, 15, 60])
    
    Returns:
        DataFrame with Trump post count features indexed by minute
    """
    
    # Create base DataFrame with all minute bars
    features = pd.DataFrame(index=price_df.index)
    
    # For each window size, count posts in the actual time window
    for window in windows:
        col_name = f'trump_cnt_{window}m'
        counts = []
        
        for current_time in price_df.index:
            # Define time window: [current_time - window minutes, current_time)
            window_start = current_time - pd.Timedelta(minutes=window)
            
            # Count posts in this time window
            posts_in_window = trump_df[
                (trump_df['datetime_minute'] >= window_start) & 
                (trump_df['datetime_minute'] < current_time)
            ]
            counts.append(len(posts_in_window))
        
        features[col_name] = counts
    
    return features


def compute_trump_sentiment_aggregates(
    trump_df: pd.DataFrame,
    price_df: pd.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """
    Compute aggregated Trump sentiment features using ACTUAL TIME windows.
    
    For each bar at time T, aggregates sentiment from posts in [T-window, T).
    Example: At 9:30 AM, 60m window aggregates posts from 8:30-9:30 AM (real time).
    
    Args:
        trump_df: Trump DataFrame with sentiment scores and datetime_minute
        price_df: Price DataFrame with datetime index
        windows: List of window sizes in minutes
    
    Returns:
        DataFrame with Trump sentiment features indexed by minute
    """
    
    features = pd.DataFrame(index=price_df.index)
    
    # For each window size, aggregate sentiment in the actual time window
    for window in windows:
        sum_pos_list = []
        sum_neg_list = []
        net_list = []
        
        for current_time in price_df.index:
            # Define time window: [current_time - window minutes, current_time)
            window_start = current_time - pd.Timedelta(minutes=window)
            
            # Get posts in this time window
            posts_in_window = trump_df[
                (trump_df['datetime_minute'] >= window_start) & 
                (trump_df['datetime_minute'] < current_time)
            ]
            
            if len(posts_in_window) > 0:
                sum_pos_list.append(posts_in_window['positive'].sum())
                sum_neg_list.append(posts_in_window['negative'].sum())
                net_list.append(posts_in_window['net_sentiment'].mean())
            else:
                sum_pos_list.append(0.0)
                sum_neg_list.append(0.0)
                net_list.append(0.0)
        
        # Store features
        features[f'trump_sum_pos_{window}m'] = sum_pos_list
        features[f'trump_sum_neg_{window}m'] = sum_neg_list
        features[f'trump_net_{window}m'] = net_list
    
    return features


def compute_trump_ewm_sentiment(
    trump_df: pd.DataFrame,
    price_df: pd.DataFrame,
    halflife: int = 60
) -> pd.DataFrame:
    """
    Compute exponentially-weighted moving average of Trump sentiment using ACTUAL TIME.
    
    Uses time-based exponential decay where older posts have exponentially decaying weights.
    The decay is based on actual time difference, not bar count.
    
    Args:
        trump_df: Trump DataFrame with sentiment scores and datetime_minute
        price_df: Price DataFrame with datetime index
        halflife: Half-life in minutes for exponential decay
    
    Returns:
        DataFrame with EWMA sentiment feature
    """
    
    features = pd.DataFrame(index=price_df.index)
    ewma_values = []
    
    # Decay factor: weight = exp(-ln(2) * time_diff / halflife)
    decay_constant = np.log(2) / halflife
    
    for current_time in price_df.index:
        # Get all posts up to current time
        historical_posts = trump_df[trump_df['datetime_minute'] <= current_time]
        
        if len(historical_posts) == 0:
            ewma_values.append(0.0)
            continue
        
        # Calculate time differences in minutes
        time_diffs = (current_time - historical_posts['datetime_minute']).dt.total_seconds() / 60
        
        # Calculate exponential weights
        weights = np.exp(-decay_constant * time_diffs)
        
        # Weighted average of sentiment
        weighted_sentiment = (historical_posts['net_sentiment'] * weights).sum() / weights.sum()
        ewma_values.append(weighted_sentiment)
    
    features['trump_sent_ewma'] = ewma_values
    
    return features


def create_trump_features(
    trump_file: str = "data_files/trump_sentiment.csv",
    price_file: str = "data_files/master_raw_mags_1m.csv",
    output_file: str = "data_files/features_trump.csv",
    windows: List[int] = [5, 15, 60],
    ewm_halflife: int = 60,
    target_tz: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Create Trump social media features for MAGS minute bars using ACTUAL TIME windows.
    
    Key difference from news features: Trump features use real-time windows, not bar-based.
    Example: At 9:30 AM, the 60m window looks at posts from 8:30-9:30 AM (actual time),
    not the previous 60 bars (which could span overnight when market was closed).
    
    This is appropriate because Trump posts occur 24/7, unlike news which is discrete events.
    
    Args:
        trump_file: Path to Trump sentiment CSV
        price_file: Path to price data CSV
        output_file: Path to save features
        windows: Time window sizes in ACTUAL MINUTES (default [5, 15, 60])
        ewm_halflife: EWMA half-life in ACTUAL MINUTES (default 60)
        target_tz: Target timezone
    
    Returns:
        DataFrame with Trump features indexed by datetime
    """
    
    # Load Trump posts
    trump_df = pd.read_csv(trump_file)
    trump_df['datetime'] = pd.to_datetime(trump_df['datetime'], utc=True)
    
    # Load price data
    price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    
    # Ensure index is datetime
    if not isinstance(price_df.index, pd.DatetimeIndex):
        price_df.index = pd.to_datetime(price_df.index)
    price_df.index.name = 'datetime'
    
    # Set timezone for price data
    import pytz
    target_pytz = pytz.timezone(target_tz)
    if price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize(target_pytz, ambiguous='infer', nonexistent='shift_forward')
    
    # Align Trump posts to minute bars
    trump_aligned = align_trump_to_minutes(trump_df, price_df, target_tz)
    
    # Compute features
    # 1. Post counts (3 features: 5m, 15m, 60m)
    count_features = compute_trump_counts(trump_aligned, price_df, windows)
    
    # 2. Sentiment aggregates (9 features: pos/neg/net × 3 windows)
    sent_features = compute_trump_sentiment_aggregates(trump_aligned, price_df, windows)
    
    # 3. EWMA sentiment (1 feature)
    ewm_features = compute_trump_ewm_sentiment(trump_aligned, price_df, ewm_halflife)
    
    # Combine all features
    features = pd.concat([count_features, sent_features, ewm_features], axis=1)
    
    # Save
    features.to_csv(output_file)
    
    return features


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create Trump social media features for regime detection"
    )
    parser.add_argument(
        '--trump-file',
        type=str,
        default='data_files/trump_sentiment.csv',
        help="Path to Trump sentiment CSV (default: trump_sentiment.csv)"
    )
    parser.add_argument(
        '--price-file',
        type=str,
        default='data_files/master_raw_mags_1m.csv',
        help="Path to price data CSV"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_files/features_trump.csv',
        help="Output file for Trump features"
    )
    
    args = parser.parse_args()
    
    create_trump_features(
        trump_file=args.trump_file,
        price_file=args.price_file,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
