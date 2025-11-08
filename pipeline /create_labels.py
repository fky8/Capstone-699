"""
Volatility Regime Labeling - 3-of-5 Forward Voting Method.

Uses forward-looking 5-bar window with majority voting:
- Baseline: 30-bar EWMA realized volatility
- Forward: 5-bar ground-truth realized volatility
- Ratio: forward_vol / baseline_vol
- Decision: 3-of-5 votes determine regime
"""

from pathlib import Path
import yaml
import argparse

import pandas as pd
import numpy as np



def load_config(config_path: str = "config_hf.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_hf_data(file_path: str) -> pd.DataFrame:
    """Load high-frequency OHLCV data."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def compute_baseline_vol(returns: pd.Series, halflife: int = 30) -> pd.Series:
    """
    Compute baseline volatility using EWMA.
    
    Args:
        returns: Price returns series
        halflife: EWMA half-life in bars (default: 30 bars = 30 minutes)
    
    Returns:
        Baseline volatility series (EWMA std)
        Note: First 'halflife' bars will be NaN (insufficient history)
    """
    # Require full halflife bars for baseline calculation
    baseline = returns.ewm(halflife=halflife, min_periods=halflife).std()
    return baseline


def compute_forward_vol(returns: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute forward realized volatility.
    
    Args:
        returns: Price returns series
        window: Forward window size (default: 5 bars)
    
    Returns:
        Forward realized volatility covering [t, t+window)
    """
    # Rolling std looking forward (use reverse rolling)
    forward_vol = returns[::-1].rolling(window=window, min_periods=window).std()[::-1]
    return forward_vol


def label_regimes_voting(
    baseline_vol: pd.Series,
    forward_vol: pd.Series,
    high_threshold: float = 1.2,
    low_threshold: float = 0.8,
    min_hold: int = 5,
    votes_needed: int = 3,
    window_size: int = 5
) -> pd.DataFrame:
    """
    Label volatility regimes using 3-of-5 forward voting.
    
    Decision at time t uses ratios from t, t+1, t+2, t+3, t+4.
    Final decision available at t+9 (when t+4's forward window completes).
    
    Args:
        baseline_vol: Baseline (slow) volatility
        forward_vol: Forward (fast) realized volatility
        high_threshold: Ratio threshold for High regime (≥ 1.2)
        low_threshold: Ratio threshold for Low regime (≤ 0.8)
        min_hold: Minimum bars to hold state before allowing change
        votes_needed: Votes needed for regime change (3 out of 5)
        window_size: Voting window size (5 bars)
    
    Returns:
        DataFrame with regime labels and diagnostics
    """
    
    # Compute ratio
    ratio = forward_vol / baseline_vol
    
    # State machine variables
    state = "Neutral"
    hold_time = 0
    
    # Output lists
    labels = []
    ratios_list = []
    votes_high = []
    votes_neutral = []
    votes_low = []
    hold_times = []
    decision_times = []
    
    # Iterate through time series
    for i, t in enumerate(ratio.index):
        # Get voting window: [t, t+1, t+2, t+3, t+4]
        if i + window_size > len(ratio):
            # Not enough forward data for voting
            labels.append("Unknown")
            ratios_list.append(np.nan)
            votes_high.append(0)
            votes_neutral.append(0)
            votes_low.append(0)
            hold_times.append(hold_time)
            decision_times.append(pd.NaT)
            continue
        
        # Get 5 ratios for voting
        S_t = ratio.iloc[i:i+window_size].values
        
        # Skip if any NaN in voting window
        if np.any(np.isnan(S_t)):
            labels.append("Unknown")
            ratios_list.append(ratio.iloc[i])
            votes_high.append(0)
            votes_neutral.append(0)
            votes_low.append(0)
            hold_times.append(hold_time)
            decision_times.append(pd.NaT)
            continue
        
        # Count votes
        nH = np.sum(S_t >= high_threshold)
        nL = np.sum(S_t <= low_threshold)
        nN = np.sum((S_t > low_threshold) & (S_t < high_threshold))
        
        # Determine new state based on voting
        new_state = None
        if nH >= votes_needed:
            new_state = "High"
        elif nL >= votes_needed:
            new_state = "Low"
        elif nN >= votes_needed:
            new_state = "Neutral"
        # else: no clear winner, stay in current state
        
        # Apply min_hold rule
        if new_state is not None:
            if new_state == state:
                # Same state, continue
                pass
            elif hold_time >= min_hold:
                # Held long enough, allow transition
                state = new_state
                hold_time = 0
            # else: can't transition yet (hold_time < min_hold), stay
        
        # Record
        labels.append(state)
        ratios_list.append(ratio.iloc[i])
        votes_high.append(int(nH))
        votes_neutral.append(int(nN))
        votes_low.append(int(nL))
        hold_times.append(hold_time)
        
        # Decision time is t+9 (when forward window of t+4 completes)
        if i + 9 < len(ratio):
            decision_times.append(ratio.index[i + 9])
        else:
            decision_times.append(pd.NaT)
        
        hold_time += 1
    
    # Create output dataframe
    result = pd.DataFrame({
        'regime_label': labels,
        'vol_ratio': ratios_list,
        'votes_high': votes_high,
        'votes_neutral': votes_neutral,
        'votes_low': votes_low,
        'hold_time': hold_times,
        'decision_time': decision_times
    }, index=ratio.index)
    
    # Map string labels to numeric
    label_map = {'Low': 0, 'Neutral': 1, 'High': 2, 'Unknown': -1}
    result['regime_numeric'] = result['regime_label'].map(label_map)
    
    # Log statistics
    
    counts = result['regime_label'].value_counts()
    total = len(result)
    
    for label in ['Low', 'Neutral', 'High', 'Unknown']:
        if label in counts.index:
            count = counts[label]
            pct = count / total * 100
    
    # Regime change statistics
    regime_changes = (result['regime_label'] != result['regime_label'].shift()).sum() - 1
    
    # Average dwell time per regime
    for regime in ['High', 'Low', 'Neutral']:
        mask = result['regime_label'] == regime
        if mask.any():
            # Find consecutive regime blocks
            blocks = (mask != mask.shift()).cumsum()[mask]
            avg_dwell = blocks.value_counts().mean()
            max_dwell = blocks.value_counts().max()
            min_dwell = blocks.value_counts().min()
    
    # Voting statistics
    valid_votes = result[result['regime_label'] != 'Unknown']
    
    return result


def main():
    """CLI entry point for 3-of-5 voting regime labeling."""
    parser = argparse.ArgumentParser(description="Create regime labels with 3-of-5 forward voting")
    parser.add_argument(
        "--config",
        type=str,
        default="config_hf.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    data_dir = config['output']['data_dir']
    ticker = config['data']['ticker']
    interval = config['data']['interval']
    
    # Get voting parameters from config (with defaults)
    voting_config = config.get('labels_voting', {})
    
    baseline_halflife = voting_config.get('baseline_halflife', 30)
    forward_window = voting_config.get('forward_window', 5)
    high_threshold = voting_config.get('high_threshold', 1.2)
    low_threshold = voting_config.get('low_threshold', 0.8)
    min_hold = voting_config.get('min_hold', 5)
    votes_needed = voting_config.get('votes_needed', 3)
    
    # Load price data
    price_df = load_hf_data(f"{data_dir}/raw_{ticker.lower()}_{interval}.csv")
    
    if price_df.empty:
        return
    
    # Compute returns
    returns = price_df['close'].pct_change()
    
    # Compute baseline volatility (30-bar EWMA)
    baseline_vol = compute_baseline_vol(returns, halflife=baseline_halflife)
    
    # Check how many bars have NaN baseline
    nan_baseline_count = baseline_vol.isna().sum()
    
    # Compute forward volatility (5-bar forward realized vol)
    forward_vol = compute_forward_vol(returns, window=forward_window)
    
    # Generate labels
    result = label_regimes_voting(
        baseline_vol=baseline_vol,
        forward_vol=forward_vol,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        min_hold=min_hold,
        votes_needed=votes_needed,
        window_size=forward_window
    )
    
    # Save labels
    output_path = f"{data_dir}/labels.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    
    # Display examples
    for regime in ['Low', 'Neutral', 'High']:
        samples = result[result['regime_label'] == regime].head(3)
    
    print(f"\nLabels saved to: {output_path}")
    print(f"Total samples: {len(result)}")


if __name__ == "__main__":
    main()
