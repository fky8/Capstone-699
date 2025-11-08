#!/usr/bin/env python3
"""
Combine Features Script
-----------------------
Combines all feature files into a single features_combined.csv

Input files (from pipeline/data_files/):
- features_news.csv (news features)
- features_garch.csv (GARCH volatility)
- labels.csv (regime labels)

Output:
- features_combined.csv (filtered from Nov 3, 2025 9:30 AM ET onward)

Author: SIADS 699 Capstone
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def load_features(data_dir: Path):
    """Load all feature files"""
    print("Loading feature files...")
    
    # Load news features
    news_file = data_dir / "features_news.csv"
    if not news_file.exists():
        raise FileNotFoundError(f"News features not found: {news_file}")
    df_news = pd.read_csv(news_file, index_col=0, parse_dates=True)
    print(f"News features: {df_news.shape[1]} features, {len(df_news)} samples")
    
    # Load GARCH features
    garch_file = data_dir / "features_garch.csv"
    if not garch_file.exists():
        raise FileNotFoundError(f"GARCH features not found: {garch_file}")
    df_garch = pd.read_csv(garch_file, index_col=0, parse_dates=True)
    print(f"GARCH features: {df_garch.shape[1]} features, {len(df_garch)} samples")
    
    # Load labels
    labels_file = data_dir / "labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels not found: {labels_file}")
    df_labels = pd.read_csv(labels_file, index_col=0, parse_dates=True)
    print(f"Labels: {df_labels.shape[1]} columns, {len(df_labels)} samples")
    
    return df_news, df_garch, df_labels


def combine_features(df_news, df_garch, df_labels):
    """Combine all features on timestamp index"""
    print("\nCombining features...")
    
    # Start with news features (baseline)
    df = df_news.copy()
    print(f"Starting with news features: {df.shape}")
    
    # Join GARCH features
    df = df.join(df_garch, how='left')
    print(f"After adding GARCH features: {df.shape}")
    
    # Join labels (select only regime_label column)
    df = df.join(df_labels[['regime_label']], how='left')
    print(f"After adding labels: {df.shape}")
    
    # Drop any rows with all NaN values
    before = len(df)
    df = df.dropna(how='all')
    after = len(df)
    if before > after:
        print(f" Dropped {before - after} rows with all NaN values")
    
    return df


def validate_features(df):
    """Validate combined features"""
    print("\n" + "="*60)
    print("FEATURE VALIDATION")
    print("="*60)
    
    # Check for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n Features with missing values:")
        for col, count in missing.items():
            pct = 100 * count / len(df)
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("No missing values")
    
    # Check for infinite values
    inf_cols = []
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
    
    if inf_cols:
        print(f"\n Features with infinite values: {inf_cols}")
    else:
        print("No infinite values")
    
    # Basic statistics
    print(f"\nTotal features: {df.shape[1]}")
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Trading days: {len(pd.Series(df.index.date).unique())}")
    
    # Feature breakdown
    news_cols = [c for c in df.columns if c.startswith(('news_', 'avg_', 'sentiment_', 'ewma_', 'topic_', 'session_'))]
    garch_cols = [c for c in df.columns if c in ['pred_vol', 'hist_vol', 'vol_ratio']]
    label_cols = [c for c in df.columns if c == 'regime_label']
    
    print(f"\nFeature breakdown:")
    print(f"  News: {len(news_cols)} features")
    print(f"  GARCH: {len(garch_cols)} features")
    print(f"  Labels: {len(label_cols)} column")
    
    # Label distribution
    if 'regime_label' in df.columns:
        print(f"\nLabel distribution:")
        label_counts = df['regime_label'].value_counts()
        for label, count in label_counts.items():
            pct = 100 * count / len(df)
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # Check for Unknown labels
        unknown_count = (df['regime_label'] == 'Unknown').sum()
        if unknown_count > 0:
            unknown_pct = 100 * unknown_count / len(df)
            print(f"   Unknown labels: {unknown_count} ({unknown_pct:.1f}%)")
    
    return True


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("COMBINE FEATURES PIPELINE")
    print("="*60)
    
    # Determine paths - use pipeline/data_files like other scripts
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data_files"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"\nData directory: {data_dir}")
    
    try:
        # Load all features
        df_news, df_garch, df_labels = load_features(data_dir)
        
        # Combine features
        df_combined = combine_features(df_news, df_garch, df_labels)
        
        # Convert index to Eastern Time and remove timezone info
        print("\nConverting timezone to Eastern Time...")
        original_tz = df_combined.index.tz
        df_combined.index = df_combined.index.tz_convert('America/New_York').tz_localize(None)
        print(f"Converted from {original_tz} to America/New_York (timezone-naive)")
        
        # Filter data from Nov 3, 2025 market open (9:30 AM ET) onward
        print("\nFiltering data from Nov 3, 2025 market open...")
        cutoff_date = pd.Timestamp('2025-11-03 09:30:00')
        rows_before = len(df_combined)
        df_combined = df_combined[df_combined.index >= cutoff_date]
        rows_after = len(df_combined)
        print(f"Removed {rows_before - rows_after} rows before {cutoff_date}")
        print(f"Kept {rows_after} rows from {df_combined.index[0]} to {df_combined.index[-1]}")
        
        # Validate
        validate_features(df_combined)
        
        # Save combined features
        output_file = data_dir / "features_combined.csv"
        df_combined.to_csv(output_file)
        print(f"\nSaved combined features to: {output_file}")
        print(f"  Shape: {df_combined.shape}")
        print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
        
        print("\n" + "="*60)
        print("FEATURE COMBINATION COMPLETE")
        print("="*60)
        print(f"\nFinal output: {output_file}")
        print(f"Total features: {df_combined.shape[1]}")
        print(f"Total samples: {len(df_combined)}")
        print("\nReady for modeling! ")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
