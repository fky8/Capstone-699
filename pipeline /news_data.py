"""
News data fetcher for Magnificent Seven stocks.
Fetches company news using Finnhub API with rate limiting.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import yaml

import pandas as pd
import requests



def load_config(config_path: str = "config_hf.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def fetch_finnhub_news(
    ticker: str,
    from_date: str,
    to_date: str,
    api_key: str
) -> List[Dict]:
    """
    Fetch company news from Finnhub API.
    
    Args:
        ticker: Stock ticker
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        api_key: Finnhub API key
    
    Returns:
        List of news articles
    """
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': from_date,
        'to': to_date,
        'token': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        articles = response.json()
        
        return articles
    
    except requests.exceptions.RequestException as e:
        return []


def fetch_news_for_tickers(
    tickers: List[str],
    lookback_days: int = 30,
    api_key: Optional[str] = None,
    rate_limit_delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch news for multiple tickers with rate limiting.
    
    Args:
        tickers: List of stock tickers
        lookback_days: Number of days to look back
        api_key: Finnhub API key (reads from env if not provided)
        rate_limit_delay: Delay between requests (seconds)
        preferred_sources: List of preferred news sources (e.g., ['Bloomberg', 'Reuters'])
        filter_by_source: If True, filter articles by preferred_sources
    
    Returns:
        DataFrame with columns: ticker, datetime, title, summary, url, source
    """
    import os
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get('FINNHUB_API_KEY')
        if not api_key:
            print ("Finnhub API key not provided. Set FINNHUB_API_KEY env var or pass api_key argument.")
            return pd.DataFrame()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    
    all_articles = []
    
    for ticker in tickers:
        
        articles = fetch_finnhub_news(ticker, from_date, to_date, api_key)
        
        # Process articles
        for article in articles:
            all_articles.append({
                'ticker': ticker,
                'datetime': pd.to_datetime(article.get('datetime', 0), unit='s', utc=True),
                'title': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'category': article.get('category', '')
            })
        
        # Rate limiting
        time.sleep(rate_limit_delay)
    
    if not all_articles:
        print("No news articles fetched.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_articles)

    
    return df


def deduplicate_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate news articles using URL hashes and fuzzy title matching.
    
    Args:
        df: News DataFrame
    
    Returns:
        Deduplicated DataFrame
    """
    initial_count = len(df)
    
    # Remove duplicates by URL
    df = df.drop_duplicates(subset=['url'], keep='first')
    
    # Create title hash for fuzzy matching
    df['title_hash'] = df['title'].apply(
        lambda x: hashlib.md5(x.lower().strip().encode()).hexdigest()[:8]
    )
    
    # Remove duplicates by title hash
    df = df.drop_duplicates(subset=['title_hash'], keep='first')
    df = df.drop(columns=['title_hash'])
    
    removed_count = initial_count - len(df)
    
    return df


def filter_by_keywords(
    df: pd.DataFrame,
    keywords: List[str]
) -> pd.DataFrame:
    """
    Filter news articles by keywords (case-insensitive).
    
    Args:
        df: News DataFrame
        keywords: List of keywords to filter by
    
    Returns:
        Filtered DataFrame with additional 'matched_keywords' column
    """
    if not keywords:
        df['matched_keywords'] = ''
        return df
    
    def find_keywords(text: str) -> str:
        text_lower = text.lower()
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        return ','.join(matched)
    
    # Find keywords in title + summary
    df['text_combined'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
    df['matched_keywords'] = df['text_combined'].apply(find_keywords)
    df = df.drop(columns=['text_combined'])
    
    # Keep articles that match at least one keyword, or all if no keywords matched
    initial_count = len(df)
    df_filtered = df[df['matched_keywords'] != ''].copy()
    
    if len(df_filtered) == 0:
        return df
    
    
    return df_filtered


def flag_market_hours(df: pd.DataFrame, market_tz: str = 'America/New_York') -> pd.DataFrame:
    """
    No-op function for backwards compatibility with continuous collector.
    Market hours flags have been removed from the pipeline.
    
    Args:
        df: News DataFrame
        market_tz: Ignored (kept for API compatibility)
    
    Returns:
        Unmodified DataFrame
    """
    # No-op: market hours features removed from pipeline
    return df


def save_news_data(df: pd.DataFrame, output_path: str) -> None:
    """Save news data to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_news_data(input_path: str) -> pd.DataFrame:
    """Load news data from CSV."""
    df = pd.read_csv(input_path, parse_dates=['datetime'])
    # Ensure datetime is UTC
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    return df


def main():
    """CLI entry point for news fetching."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch news for Magnificent Seven stocks")
    parser.add_argument(
        "--config",
        type=str,
        default="config_hf.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Finnhub API key (or set FINNHUB_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    tickers = config['data']['magnificent_seven']
    lookback_days = config['data']['news_lookback_days']
    
    # Fetch news
    df = fetch_news_for_tickers(
        tickers=tickers,
        lookback_days=lookback_days,
        api_key=args.api_key,
        rate_limit_delay=1.0
    )
    
    if df.empty:
        return
    
    # Deduplicate
    df = deduplicate_news(df)
    
    # Note: Keyword filtering removed - using all news articles
    
    # Save data
    output_dir = config['output']['data_dir']
    output_path = f"{output_dir}/news_raw.csv"
    save_news_data(df, output_path)
    
    # Display summary


if __name__ == "__main__":
    main()
