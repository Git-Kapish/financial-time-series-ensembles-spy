"""Download OHLCV data using yfinance."""

import argparse
from pathlib import Path
import pandas as pd
import yfinance


def download_ohlcv(symbol: str, start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Download daily OHLCV data for a given symbol using yfinance.
    
    Args:
        symbol: Stock or index ticker symbol (e.g., "SPY", "AAPL")
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format (None for today)
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    df = yfinance.download(symbol, start=start, end=end, auto_adjust=False)
    df = df.reset_index()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data for a stock or index")
    parser.add_argument("--symbol", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    args = parser.parse_args()
    
    # Download data
    df = download_ohlcv(args.symbol)
    
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_path = output_dir / f"{args.symbol}_daily.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Downloaded {args.symbol} data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Saved to: {output_path}")
