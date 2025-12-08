"""
News feature engineering for high-frequency regime detection.
Aggregates news sentiment into minute-level features.
"""

from typing import List, Optional
import yaml

import pandas as pd
import numpy as np



def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def align_news_to_minutes(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    target_tz: str = 'America/New_York'
) -> pd.DataFrame:
    """
    Align news timestamps to minute bars.
    
    Round UP to next minute to ensure price data can reflect the news impact.
    Example: News at 9:45:30 → aligned to 9:46:00 bar
    
    Args:
        news_df: News DataFrame with datetime in UTC
        price_df: Price DataFrame with datetime index in target timezone
        target_tz: Target timezone
    
    Returns:
        News DataFrame aligned to price bars
    """
    import pytz
    
    # Convert news datetime to target timezone
    target_pytz = pytz.timezone(target_tz)
    news_df = news_df.copy()
    news_df['datetime'] = news_df['datetime'].dt.tz_convert(target_pytz)
    
    # Round UP to next minute (ceiling) to ensure price data reflects news
    news_df['datetime_minute'] = news_df['datetime'].dt.ceil('min')
    
    # Get price data range and convert to target timezone
    min_dt = price_df.index.min().tz_convert(target_pytz)
    max_dt = price_df.index.max().tz_convert(target_pytz)
    
    # For each trading day, clip news timestamps to that day's market hours
    # News before market open gets attributed to first bar (market open)
    # News after market close gets attributed to last bar (market close)
    news_df['date'] = news_df['datetime_minute'].dt.date
    
    def clip_to_market_hours(group):
        date = group.name
        # Get that day's price range
        day_price = price_df[price_df.index.tz_convert(target_pytz).date == date]
        if len(day_price) > 0:
            day_min = day_price.index.min().tz_convert(target_pytz)
            day_max = day_price.index.max().tz_convert(target_pytz)
            group['datetime_minute'] = group['datetime_minute'].clip(lower=day_min, upper=day_max)
        return group
    
    news_df = news_df.groupby('date', group_keys=False).apply(clip_to_market_hours)
    news_df = news_df.drop(columns=['date'])
    
    # Filter out news from dates outside the price data date range
    min_date = min_dt.date()
    max_date = max_dt.date()
    news_df = news_df[(news_df['datetime_minute'].dt.date >= min_date) & 
                      (news_df['datetime_minute'].dt.date <= max_date)]
    
    return news_df


def compute_news_counts(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """
    Compute rolling news counts for multiple windows.
    
    Args:
        news_df: News DataFrame with datetime_minute
        price_df: Price DataFrame with datetime index
        windows: List of window sizes in minutes
    
    Returns:
        DataFrame with news count features indexed by minute
    """
    
    # Create base DataFrame with all minute bars
    features = pd.DataFrame(index=price_df.index)
    
    # Count news per minute (all tickers combined for MAGS)
    news_per_minute = news_df.groupby('datetime_minute').size()
    
    # Compute rolling counts
    for window in windows:
        col_name = f'news_cnt_{window}m'
        features[col_name] = news_per_minute.reindex(features.index, fill_value=0).rolling(
            window=window,
            min_periods=1
        ).sum()
        
    
    return features


def compute_sentiment_aggregates(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    windows: List[int] = [5, 15, 60]
) -> pd.DataFrame:
    """
    Compute aggregated sentiment features.
    
    Args:
        news_df: News DataFrame with sentiment scores
        price_df: Price DataFrame with datetime index
        windows: List of window sizes in minutes
    
    Returns:
        DataFrame with sentiment features indexed by minute
    """
    
    features = pd.DataFrame(index=price_df.index)
    
    # Aggregate sentiment per minute
    sent_agg = news_df.groupby('datetime_minute').agg({
        'sent_positive': 'sum',
        'sent_negative': 'sum',
        'sent_net': 'mean'
    })
    
    # Compute rolling aggregates
    for window in windows:
        # Sum of positive sentiment
        col_name = f'sent_sum_pos_{window}m'
        features[col_name] = sent_agg['sent_positive'].reindex(features.index, fill_value=0).rolling(
            window=window,
            min_periods=1
        ).sum()
        
        # Sum of negative sentiment
        col_name = f'sent_sum_neg_{window}m'
        features[col_name] = sent_agg['sent_negative'].reindex(features.index, fill_value=0).rolling(
            window=window,
            min_periods=1
        ).sum()
        
        # Net sentiment (mean)
        col_name = f'sent_net_{window}m'
        features[col_name] = sent_agg['sent_net'].reindex(features.index, fill_value=0).rolling(
            window=window,
            min_periods=1
        ).mean()
    
    
    return features


def compute_ewm_sentiment(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    halflife: int = 60
) -> pd.DataFrame:
    """
    Compute exponentially-weighted moving average sentiment.
    
    Args:
        news_df: News DataFrame with sentiment scores
        price_df: Price DataFrame with datetime index
        halflife: Half-life in minutes
    
    Returns:
        DataFrame with EWMA sentiment feature
    """
    
    features = pd.DataFrame(index=price_df.index)
    
    # Get net sentiment per minute
    sent_per_minute = news_df.groupby('datetime_minute')['sent_net'].mean()
    sent_series = sent_per_minute.reindex(features.index, fill_value=0)
    
    # Compute EWMA
    features[f'sent_ewm_{halflife}m'] = sent_series.ewm(
        halflife=halflife,
        min_periods=1
    ).mean()
    
    
    return features


def compute_topic_counts(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    keywords: List[str]
) -> pd.DataFrame:
    """
    Compute topic counts based on keyword matches.
    
    Args:
        news_df: News DataFrame with matched_keywords column
        price_df: Price DataFrame with datetime index
        keywords: List of keywords to track
    
    Returns:
        DataFrame with topic count features
    """
    
    features = pd.DataFrame(index=price_df.index)
    
    # Count articles per keyword per minute
    for keyword in keywords:
        keyword_clean = keyword.lower().replace(' ', '_')
        col_name = f'topic_{keyword_clean}'
        
        # Check if keyword appears in matched_keywords
        news_df[f'has_{keyword}'] = news_df['matched_keywords'].str.contains(
            keyword,
            case=False,
            na=False
        )
        
        # Count per minute
        keyword_counts = news_df[news_df[f'has_{keyword}']].groupby('datetime_minute').size()
        features[col_name] = keyword_counts.reindex(features.index, fill_value=0)
        
        # Drop temporary column
        news_df = news_df.drop(columns=[f'has_{keyword}'])
    
    
    return features


def compute_all_news_features(
    news_df: pd.DataFrame,
    price_df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Compute all news features.
    
    Args:
        news_df: News DataFrame with sentiment
        price_df: Price DataFrame
        config: Configuration dictionary
    
    Returns:
        DataFrame with all news features
    """
    
    # Get config
    count_windows = config['features']['news']['count_windows']
    sent_windows = config['features']['news']['sentiment_windows']
    ewm_halflife = config['features']['news']['ewm_halflife']
    keywords = config['data'].get('news_keywords', [])  # Optional, default to empty list
    target_tz = config['data']['timezone']
    
    # Align news to minute bars
    news_df = align_news_to_minutes(news_df, price_df, target_tz)
    
    # Compute features
    count_features = compute_news_counts(news_df, price_df, count_windows)
    sent_features = compute_sentiment_aggregates(news_df, price_df, sent_windows)
    ewm_features = compute_ewm_sentiment(news_df, price_df, ewm_halflife)
    
    # Topic counts (optional)
    if config['features']['news']['track_keywords']:
        topic_features = compute_topic_counts(news_df, price_df, keywords)
        all_features = pd.concat([count_features, sent_features, ewm_features, topic_features], axis=1)
    else:
        all_features = pd.concat([count_features, sent_features, ewm_features], axis=1)
    
    # Forward fill missing values
    all_features = all_features.fillna(method='ffill').fillna(0)
    
    
    return all_features


def main():
    """CLI entry point for news feature engineering."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Compute news features")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    data_dir = config['output']['data_dir']
    
    # Load price data directly from master file
    ticker = config['data']['ticker']
    interval = config['data']['interval']
    price_path = f"{data_dir}/master_raw_{ticker.lower()}_{interval}.csv"
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Load news sentiment data directly
    news_path = f"{data_dir}/news_sentiment.csv"
    news_df = pd.read_csv(news_path, parse_dates=['datetime'])
    if news_df['datetime'].dt.tz is None:
        news_df['datetime'] = news_df['datetime'].dt.tz_localize('UTC')
    
    # Compute features
    news_features = compute_all_news_features(news_df, price_df, config)
    
    # Save features
    output_path = f"{data_dir}/features_news.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    news_features.to_csv(output_path)
    
    print(f"News features saved to {output_path}")
    print(f"Shape: {news_features.shape}")


if __name__ == "__main__":
    main()
