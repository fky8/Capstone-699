"""
GARCH volatility forecasting for high-frequency data.
Implements GARCH(1,1) with expanding window estimation.
"""

from pathlib import Path
from typing import Optional
import yaml

import pandas as pd
import numpy as np
from arch import arch_model



def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_garch_volatility_expanding(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    horizon: int = 20,
    min_observations: int = 100,
    rescale: bool = True
) -> pd.Series:
    """
    Compute GARCH volatility forecast using expanding window.
    
    Args:
        returns: Return series (percentage returns recommended)
        p: GARCH lag order
        q: ARCH lag order
        horizon: Forecast horizon (steps ahead)
        min_observations: Minimum observations before starting forecasts
        rescale: Whether to rescale returns for numerical stability
    
    Returns:
        Series of forecasted volatility (annualized)
    """
    
    # Drop NaNs
    returns = returns.dropna()
    
    if len(returns) < min_observations:
        return pd.Series(np.nan, index=returns.index)
    
    # Convert to percentage returns if not already (for numerical stability)
    if rescale and returns.abs().mean() < 1:
        returns_scaled = returns * 100
    else:
        returns_scaled = returns
    
    forecasts = []
    forecast_index = []
    
    # Expanding window forecasting
    for i in range(min_observations, len(returns_scaled)):
        if i % 500 == 0:
            print(f"Processing bar {i}/{len(returns_scaled)}...")
        
        try:
            # Fit on data up to i
            window_data = returns_scaled.iloc[:i]
            
            # Fit GARCH model
            model = arch_model(
                window_data,
                vol='Garch',
                p=p,
                q=q,
                rescale=False  # We already rescaled if needed
            )
            
            model_fit = model.fit(disp='off', show_warning=False)
            
            # Forecast horizon-step ahead variance
            forecast = model_fit.forecast(horizon=horizon, reindex=False)
            forecast_var = forecast.variance.values[-1, -1]  # Last row, last column
            
            # Convert to volatility (from variance)
            if rescale and returns.abs().mean() < 1:
                # Unscale: divide by 100
                forecast_vol_raw = np.sqrt(forecast_var) / 100
            else:
                forecast_vol_raw = np.sqrt(forecast_var)
            
            # Annualize: vol_annual = vol_minute * sqrt(minutes_per_year)
            # minutes_per_year = 252 * 390
            annual_factor = np.sqrt(252 * 390)
            forecast_vol_ann = forecast_vol_raw * annual_factor
            
            forecasts.append(forecast_vol_ann)
            forecast_index.append(returns.index[i])
        
        except Exception as e:
            forecasts.append(np.nan)
            forecast_index.append(returns.index[i])
    
    # Create series aligned with original returns
    garch_vol = pd.Series(forecasts, index=forecast_index)
    garch_vol = garch_vol.reindex(returns.index)
    
    
    return garch_vol


def compute_historical_volatility(
    returns: pd.Series,
    window: int = 30
) -> pd.Series:
    """
    Compute rolling historical volatility.
    
    Args:
        returns: Return series
        window: Rolling window size in minutes
    
    Returns:
        Series of annualized historical volatility
    """
    # Rolling std
    hist_vol = returns.rolling(window=window, min_periods=max(10, window//4)).std()
    
    # Annualize
    annual_factor = np.sqrt(252 * 390)
    hist_vol_ann = hist_vol * annual_factor
    
    return hist_vol_ann


def compute_volatility_ratio(
    pred_vol: pd.Series,
    hist_vol: pd.Series
) -> pd.Series:
    """
    Compute volatility ratio.
    
    Args:
        pred_vol: Predicted volatility
        hist_vol: Historical volatility
    
    Returns:
        Volatility ratio series
    """
    vol_ratio = pred_vol / hist_vol
    
    # Handle division by zero or very small values
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)
    
    return vol_ratio


def compute_all_garch_features(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Compute all GARCH-based features.
    
    Args:
        df: Price DataFrame with OHLCV columns
        config: Configuration dictionary
    
    Returns:
        DataFrame with GARCH features
    """
    
    # Get config
    p = config['features']['garch']['p']
    q = config['features']['garch']['q']
    horizon = config['features']['garch']['forecast_horizon']
    min_obs = config['features']['garch']['min_observations']
    vol_window = config['features']['garch']['vol_ratio_window']
    
    # Compute returns
    returns = df['close'].pct_change()
    
    # Compute GARCH forecast
    pred_vol = compute_garch_volatility_expanding(
        returns=returns,
        p=p,
        q=q,
        horizon=horizon,
        min_observations=min_obs
    )
    
    # Compute historical volatility
    hist_vol = compute_historical_volatility(returns, window=vol_window)
    
    # Compute volatility ratio
    vol_ratio = compute_volatility_ratio(pred_vol, hist_vol)
    
    # Combine features
    features = pd.DataFrame({
        'pred_vol': pred_vol,
        'hist_vol': hist_vol,
        'vol_ratio': vol_ratio
    }, index=df.index)
    
    # Forward fill NaNs
    features = features.fillna(method='ffill')
    
    
    return features


def main():
    """CLI entry point for GARCH feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute GARCH features")
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
    ticker = config['data']['ticker']
    interval = config['data']['interval']
    
    # Load price data directly from master file
    price_path = f"{data_dir}/master_raw_{ticker.lower()}_{interval}.csv"
    price_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    if price_df.empty:
        return
    
    # Compute features
    garch_features = compute_all_garch_features(price_df, config)
    
    # Save features
    output_path = f"{data_dir}/features_garch.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    garch_features.to_csv(output_path)
    
    # Display summary
    
    # Show volatility ratio distribution


if __name__ == "__main__":
    main()
