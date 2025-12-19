"""Technical indicator calculations."""

import pandas as pd
from typing import List


def add_lagged_returns(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Add lagged return features to capture momentum and mean-reversion patterns.
    
    Lagged returns allow the model to learn from recent price momentum. For example,
    a positive ret_lag_1 indicates yesterday's price went up, which may signal
    continuation (momentum) or reversal (mean-reversion) depending on the asset.
    
    Args:
        df: DataFrame with 'daily_ret' column
        lags: List of lag periods (e.g., [1, 2, 5, 10])
    
    Returns:
        DataFrame with new columns: ret_lag_1, ret_lag_2, etc.
    """
    df = df.copy()
    for k in lags:
        df[f'ret_lag_{k}'] = df['daily_ret'].shift(k)
    return df


def add_rolling_stats(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Add rolling mean and volatility (std) of returns.
    
    Rolling mean captures the average recent performance, while rolling std measures
    recent volatility. High volatility often precedes regime changes or indicates
    uncertainty, while low volatility may signal stable trends.
    
    Args:
        df: DataFrame with 'daily_ret' column
        windows: List of rolling window sizes (e.g., [5, 10, 20] for weekly, biweekly, monthly)
    
    Returns:
        DataFrame with new columns: ret_roll_mean_5, ret_roll_std_5, etc.
    """
    df = df.copy()
    for w in windows:
        df[f'ret_roll_mean_{w}'] = df['daily_ret'].rolling(w).mean()
        df[f'ret_roll_std_{w}'] = df['daily_ret'].rolling(w).std()
    return df


def add_moving_averages_and_bbands(df: pd.DataFrame, windows: list[int], n_std: float = 2.0) -> pd.DataFrame:
    """
    Add moving averages and Bollinger Bands for trend and volatility analysis.
    
    Moving averages smooth price action to identify trends. Bollinger Bands add
    volatility envelopes around the MA: when price touches the upper band, it may
    be overbought; lower band suggests oversold. The bands widen during volatile
    periods and contract during calm periods.
    
    Args:
        df: DataFrame with 'Adj Close' column
        windows: List of MA window sizes (e.g., [5, 20] for short/long term)
        n_std: Number of standard deviations for Bollinger Bands (default 2.0)
    
    Returns:
        DataFrame with columns: ma_5, price_roll_std_5, bb_upper_5, bb_lower_5, etc.
    """
    df = df.copy()
    for w in windows:
        # Moving average (middle Bollinger Band)
        ma = df['Adj Close'].rolling(w).mean()
        df[f'ma_{w}'] = ma
        
        # Rolling standard deviation of price
        price_std = df['Adj Close'].rolling(w).std()
        df[f'price_roll_std_{w}'] = price_std
        
        # Bollinger Bands
        df[f'bb_upper_{w}'] = ma + n_std * price_std
        df[f'bb_lower_{w}'] = ma - n_std * price_std
    
    return df


def add_rsi(df: pd.DataFrame, window: int = 14, price_col: str = "Adj Close") -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI) momentum oscillator.
    
    RSI measures the speed and magnitude of recent price changes on a 0-100 scale.
    RSI > 70 typically indicates overbought conditions (potential reversal down),
    while RSI < 30 indicates oversold conditions (potential reversal up). RSI helps
    identify momentum exhaustion and potential trend reversals.
    
    Args:
        df: DataFrame with price column
        window: RSI lookback period (default 14, standard in technical analysis)
        price_col: Name of price column to use (default 'Adj Close')
    
    Returns:
        DataFrame with new column: rsi_14 (or rsi_{window})
    """
    df = df.copy()
    
    # Calculate price changes
    delta = df[price_col].diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate exponential moving averages
    alpha = 1 / window
    avg_gain = gain.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df[f'rsi_{window}'] = rsi
    
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate all feature engineering steps to create a model-ready DataFrame.
    
    This function combines multiple technical indicators:
    - Lagged returns for momentum/mean-reversion signals
    - Rolling statistics for trend and volatility measures
    - Moving averages and Bollinger Bands for support/resistance levels
    - RSI for overbought/oversold conditions
    
    The resulting feature matrix provides a comprehensive view of price dynamics
    across multiple timeframes, enabling ML models to capture complex patterns.
    
    Args:
        df: DataFrame with DateTimeIndex and columns ['Adj Close', 'daily_ret']
    
    Returns:
        Clean DataFrame with all features, NaN rows removed (due to initial rolling windows)
    """
    df = df.copy()
    
    # Add all technical features
    df = add_lagged_returns(df, lags=[1, 2, 5, 10])
    df = add_rolling_stats(df, windows=[5, 10, 20])
    df = add_moving_averages_and_bbands(df, windows=[5, 20])
    df = add_rsi(df, window=14)
    
    # Remove rows with NaN values from initial rolling windows
    df = df.dropna()
    
    return df
