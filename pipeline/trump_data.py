"""
Trump Social Media Data Fetcher
--------------------------------
Fetches Trump's Truth Social and Twitter posts from FactBase API.

API: https://rollcall.com/wp-json/factbase/v1/twitter
Coverage: Twitter archive (2009-2021) + Truth Social (2022-2025)

Usage:
    python trump_data.py --start 2025-10-29 --end 2025-11-24
    python trump_data.py --start 2025-10-29 --end 2025-11-24 --platform truthsocial
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import yaml

import pandas as pd
import requests


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def fetch_factbase_api(
    page: int = 1,
    platform: str = 'all',
    sort: str = 'date',
    sort_order: str = 'desc',
    page_size: int = 50
) -> Dict:
    """
    Fetch posts from FactBase API.
    
    Args:
        page: Page number (starts at 1)
        platform: 'all', 'truthsocial', or 'twitter'
        sort: 'date' or 'relevance'
        sort_order: 'asc' or 'desc'
        page_size: Number of posts per page (default 50)
    
    Returns:
        API response dict with 'meta', 'data', and 'stats' keys
    """
    
    url = "https://rollcall.com/wp-json/factbase/v1/twitter"
    
    # Platform mapping
    platform_map = {
        'truthsocial': 'truth_social',
        'twitter': 'x_twitter',
        'all': 'all'
    }
    api_platform = platform_map.get(platform, platform)
    
    params = {
        'platform': api_platform,
        'sort': sort,
        'sort_order': sort_order,
        'page': page,
        'format': 'json'
    }
    
    headers = {
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"PI request failed: {e}")
        return {'meta': {}, 'data': [], 'stats': {}}


def parse_post(post_data: Dict) -> Optional[Dict]:
    """
    Parse a post from FactBase API response.
    
    Args:
        post_data: Raw post dict from API
    
    Returns:
        Parsed post dict or None if parsing fails
    """
    try:
        # Parse timestamp
        date_str = post_data.get('date')
        if not date_str:
            return None
        
        dt = pd.to_datetime(date_str)
        
        # Get post text (handle video/media posts)
        text = post_data.get('text', '')
        social_data = post_data.get('social', {})
        post_text = social_data.get('post_text', text)
        
        # Skip posts without meaningful text content
        # Drop media-only posts: [Video], [Image], [Photo], etc.
        if not post_text or len(post_text.strip()) < 3:
            return None
        
        post_text_clean = post_text.strip()
        media_only_patterns = ['[Video]', '[Image]', '[Photo]', '[Media]', '[Link]']
        if post_text_clean in media_only_patterns:
            return None
        
        # Platform
        platform = post_data.get('platform', 'Unknown')
        
        # URL
        post_url = post_data.get('post_url', '')
        
        # Metadata
        favorite_count = social_data.get('favorite_count', 0)
        repost_count = social_data.get('repost_count', 0)
        
        return {
            'datetime': dt,
            'text': post_text_clean,
            'platform': platform,
            'url': post_url,
            'favorites': favorite_count,
            'reposts': repost_count,
            'deleted': post_data.get('deleted_flag', False)
        }
    
    except Exception as e:
        return None


def fetch_posts(
    start_date: datetime,
    end_date: datetime,
    platform: str = 'all',
    max_pages: int = 200,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Fetch Trump posts within date range.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        platform: 'truthsocial', 'twitter', or 'all'
        max_pages: Maximum pages to fetch
        delay: Delay between requests (seconds)
    
    Returns:
        DataFrame with columns: datetime, text, platform, url, favorites, reposts, deleted
    """
    
    all_posts = []
    should_continue = True
    page = 1
    
    while should_continue and page <= max_pages:
        # Fetch page
        response = fetch_factbase_api(
            page=page,
            platform=platform,
            sort='date',
            sort_order='desc'
        )
        
        posts_data = response.get('data', [])
        meta = response.get('meta', {})
        
        if not posts_data:
            break
        
        # Parse posts
        page_posts = []
        earliest_date = None
        latest_date = None
        
        for post_data in posts_data:
            parsed = parse_post(post_data)
            
            if parsed:
                post_dt = parsed['datetime']
                
                # Track date range on this page
                if earliest_date is None or post_dt < earliest_date:
                    earliest_date = post_dt
                if latest_date is None or post_dt > latest_date:
                    latest_date = post_dt
                
                # Filter by date range
                if start_date <= post_dt <= end_date:
                    page_posts.append(parsed)
        
        all_posts.extend(page_posts)
        
        # Check if we've passed the start date
        if earliest_date and earliest_date < start_date:
            should_continue = False
        
        # Check if there are more pages
        pagination = meta.get('pagination', {})
        if not pagination.get('next_page'):
            should_continue = False
        
        page += 1
        
        # Rate limiting
        if should_continue:
            time.sleep(delay)
    
    # Convert to DataFrame
    if all_posts:
        df = pd.DataFrame(all_posts)
        df = df.sort_values('datetime').reset_index(drop=True)
        df['text_length'] = df['text'].str.len()
        return df
    else:
        return pd.DataFrame()


def save_data(df: pd.DataFrame, output_dir: str = "data_files"):
    """Save collected posts to CSV."""
    if len(df) == 0:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate filename
    start_str = df['datetime'].min().strftime('%Y%m%d')
    end_str = df['datetime'].max().strftime('%Y%m%d')
    filename = f"trump_raw_{start_str}_to_{end_str}.csv"
    filepath = output_path / filename
    
    # Drop temporary columns
    df_save = df.drop(columns=['text_length'], errors='ignore')
    
    # Save
    df_save.to_csv(filepath, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Trump social media posts from FactBase API"
    )
    
    # Calculate default start date (30 days ago)
    default_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    parser.add_argument(
        '--start',
        type=str,
        default=default_start,
        help="Start date (YYYY-MM-DD). Default: 30 days ago"
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). If not provided, uses today"
    )
    parser.add_argument(
        '--platform',
        type=str,
        default='all',
        choices=['truthsocial', 'twitter', 'all'],
        help="Platform filter"
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=200,
        help="Maximum pages to fetch"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data_files',
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Parse dates (make timezone-aware for Eastern Time)
    start_date = pd.to_datetime(args.start).tz_localize('US/Eastern')
    
    # Use today if end date not provided
    if args.end is None:
        end_date = pd.Timestamp.now(tz='US/Eastern').normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        end_date = pd.to_datetime(args.end).tz_localize('US/Eastern') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    # Fetch posts
    df = fetch_posts(
        start_date=start_date,
        end_date=end_date,
        platform=args.platform,
        max_pages=args.max_pages
    )
    
    # Save
    if len(df) > 0:
        save_data(df, args.output_dir)


if __name__ == "__main__":
    main()
