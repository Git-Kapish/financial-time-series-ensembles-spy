"""Trading utility metrics used in SPY experiments."""

import numpy as np
import pandas as pd


def compute_equity_curve(returns: pd.Series) -> pd.Series:
    """Return cumulative equity curve from a series of returns."""

    return (1 + returns).cumprod()


def compute_sharpe_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """Compute annualized Sharpe ratio; return 0 if variance is zero."""

    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(trading_days)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def get_strategy_returns(model, X, future_returns, threshold: float = 0.5) -> pd.Series:
    """Convert predicted probabilities into long-only strategy returns."""

    proba = model.predict_proba(X)[:, 1]
    signal = (proba >= threshold).astype(int)
    strategy_ret = signal * future_returns.values
    return pd.Series(strategy_ret, index=X.index, name="strategy_ret")
