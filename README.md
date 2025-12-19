# Financial Time Series Forecasting with Ensembles

This project predicts SPY (S&P 500 ETF) next-day price direction using daily OHLCV data from Yahoo Finance, technical indicators, and ensemble machine learning models (Random Forest, XGBoost). We implement a complete pipeline from feature engineering to model training with proper time-series validation, evaluating both traditional ML metrics (Accuracy, ROC-AUC) and realistic trading performance metrics (Sharpe ratio, maximum drawdown) to assess real-world applicability of directional forecasting strategies.

## Problem Formulation

We work with adjusted close prices $P_t$ and compute simple daily returns:

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

The binary classification target is defined as:

$$y_t = \begin{cases} 1 & \text{if } r_{t+1} > 0 \\ 0 & \text{otherwise} \end{cases}$$

We model returns rather than raw prices because returns are approximately stationary, exhibit better statistical properties for modeling, and directly relate to investment performance. The goal is to predict whether tomorrow's return will be positive (market up) or negative (market down) based on today's technical features.

## Data

**Source:** Daily OHLCV data for SPY (S&P 500 ETF) retrieved from Yahoo Finance using the `yfinance` Python library.

**Time Period:** Approximately 2010 to present (~4,000 daily observations).

**Columns:**
- **Open** - Opening price
- **High** - Highest price during the trading day
- **Low** - Lowest price during the trading day  
- **Close** - Closing price
- **Adj Close** - Adjusted closing price (accounts for dividends and splits)
- **Volume** - Number of shares traded

All feature engineering and modeling use the Adjusted Close price to ensure historical consistency.

## Features

Technical features are computed using `src.features.technical` module. Each feature captures different market dynamics:

- **Lagged daily returns** (1, 2, 5, 10 days): Capture momentum and mean-reversion patterns; recent price movements often exhibit continuation or reversal tendencies.

- **Rolling mean and standard deviation of returns** (windows: 5, 10, 20 days): Measure trend direction and volatility; high volatility periods often signal increased risk or regime changes.

- **Price-based moving averages** (5, 20 days): Smooth price trends and act as dynamic support/resistance levels; crossovers signal trend changes.

- **Bollinger Bands** (upper/lower bands, 5 and 20-day windows): Measure price volatility bands using 2 standard deviations; price touching bands may indicate overbought/oversold conditions.

- **RSI(14)** (Relative Strength Index): Momentum oscillator scaled 0-100; values >70 suggest overbought, <30 suggest oversold conditions, helping identify potential reversals.

## Models and Training

**Models evaluated:**
- **Baseline:** Logistic Regression with standardized features (scikit-learn Pipeline with StandardScaler)
- **Ensemble:** RandomForestClassifier with hyperparameter tuning
- **Ensemble:** XGBClassifier (Gradient Boosting) with hyperparameter tuning  
- **Meta-Ensemble:** RF+XGB probability averaging (simple ensemble of best models)

**Time-based train/validation/test split:**
- **Train:** First 70% of chronological observations
- **Validation:** Next 15% of observations
- **Test:** Final 15% of observations

**Critical for time series:** No random shuffling. Data is split chronologically to prevent look-ahead bias. Hyperparameter tuning uses `TimeSeriesSplit` for cross-validation, ensuring each fold respects temporal ordering.

**Model selection criterion:** ROC-AUC score on validation set, which captures the model's ability to rank predictions (important for trading strategies that can use probability thresholds).

## Evaluation Metrics

### Machine Learning Metrics

Standard classification metrics computed on train, validation, and test sets:

- **Accuracy:** Overall correctness of predictions
- **Precision:** Of predicted "up" days, fraction that were actually up
- **Recall:** Of actual "up" days, fraction correctly predicted
- **ROC-AUC:** Area under ROC curve; measures ranking quality (0.5 = random, 1.0 = perfect)

### Trading Metrics

To evaluate real-world applicability, we simulate a **simple long-only strategy**:

**Strategy rule:** 
- If model predicts "up" for next day (P(up) ≥ 0.5), hold long position
- Otherwise, hold cash (no position)

**Strategy return on day $t+1$:**

$$r_{\text{strat}, t+1} = \text{signal}_t \times r_{t+1}$$

where $\text{signal}_t \in \{0, 1\}$ is the model's prediction.

**Cumulative equity curve:**

$$\text{Equity}_t = \prod_{u \leq t} (1 + r_{\text{strat}, u})$$

**Annualized Sharpe ratio:**

$$\text{Sharpe} = \frac{\bar{r}_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{252}$$

where $\bar{r}_{\text{daily}}$ is mean daily return and $\sigma_{\text{daily}}$ is standard deviation of daily returns.

**Maximum drawdown:**

$$\text{Max Drawdown} = \min_t \left( \frac{\text{Equity}_t}{\max_{u \leq t} \text{Equity}_u} - 1 \right)$$

This measures the worst peak-to-trough decline in equity value.

## Results Summary

Performance metrics on the **test set** (most recent 15% of data):

| Model | Acc (test) | ROC-AUC (test) | Sharpe (test) | Max Drawdown (test) |
|-------|------------|----------------|---------------|---------------------|
| Logistic Regression | 0.5650 | 0.4897 | 1.1116 | -0.1970 |
| Random Forest | 0.4733 | 0.5099 | 0.7399 | -0.1466 |
| XGBoost | 0.4267 | 0.4414 | 0.6935 | -0.0702 |
| RF+XGB Ensemble | 0.4250 | 0.4449 | — | — |
| Buy & Hold | — | — | 1.2124 | -0.1876 |

**Key interpretations:**

- **ROC-AUC near 0.5** across all models indicates weak predictive signal from technical features alone; daily price movements are extremely difficult to forecast.

- **Buy-and-hold outperforms** all ML strategies on risk-adjusted basis (highest Sharpe), suggesting that timing the market with these features provides no edge over passive investing.

- **Trade-off observed:** XGBoost achieves smallest drawdown (-7.02%) but lowest returns, indicating overly conservative predictions. Logistic regression balances returns and risk better than complex ensembles.

- **Overfitting evident:** Complex models (RF, XGBoost) show strong training performance but fail to generalize, highlighting the challenge of learning genuine patterns vs. noise in financial time series.

## Project Structure

```
.
├── data/
│   └── raw/                    # Downloaded OHLCV data from yfinance
│       └── SPY_daily.csv
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis (price, returns, distributions)
│   └── 02_features_and_models.ipynb  # Feature engineering and ensemble modeling
├── src/
│   ├── data/
│   │   └── download_data.py   # Script to download data via yfinance
│   ├── features/
│   │   └── technical.py       # Technical indicator computation functions
│   ├── models/
│   │   ├── baseline.py        # Baseline model implementations
│   │   └── ensemble.py        # Ensemble model utilities
│   └── eval/
│       └── metrics.py         # Trading performance metrics
├── figures/                    # Saved plots from analysis
│   ├── price_and_returns.png
│   └── equity_curves_test.png
├── results/                    # Saved CSV files with metrics
│   ├── model_metrics.csv
│   └── trading_metrics.csv
├── requirements.txt
└── README.md
```

## How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download SPY data from Yahoo Finance:**
```bash
python -m src.data.download_data
```

To download a different symbol:
```bash
python -m src.data.download_data --symbol AAPL
```

**3. Run exploratory data analysis:**
```bash
jupyter notebook notebooks/01_eda.ipynb
```

**4. Run feature engineering and model training:**
```bash
jupyter notebook notebooks/02_features_and_models.ipynb
```

The second notebook will:
- Build technical features
- Train and tune ensemble models with time-series cross-validation
- Evaluate ML metrics and trading performance
- Generate plots and save results to `figures/` and `results/`

## Key Plots and Artifacts

The following files are generated by running the notebooks:

- **`figures/price_and_returns.png`** - SPY price history and daily returns distribution (from 01_eda.ipynb)
- **`figures/equity_curves_test.png`** - Equity curves comparing all strategies on test set (from 02_features_and_models.ipynb)
- **`results/model_metrics.csv`** - ML metrics (Accuracy, Precision, Recall, ROC-AUC) for all models on train/val/test
- **`results/trading_metrics.csv`** - Trading metrics (Cumulative Return, Sharpe, Max Drawdown) for all strategies on test set

See the code snippets in `notebooks/02_features_and_models.ipynb` for how these artifacts are saved:

```python
# Example: Save model comparison metrics
comparison_df.to_csv('../results/model_metrics.csv')

# Example: Save trading metrics
trading_metrics_df.to_csv('../results/trading_metrics.csv')

# Example: Save equity curve plot
plt.savefig('../figures/equity_curves_test.png', dpi=300, bbox_inches='tight')
```

## Conclusions and Future Work

This project demonstrates a complete pipeline for financial time series forecasting, but reveals the fundamental challenge: **predicting daily market direction from technical indicators alone provides minimal edge**. Key lessons:

1. **Market efficiency:** Short-term price movements are largely unpredictable using publicly available technical data
2. **Model complexity ≠ performance:** Simple logistic regression matched or exceeded complex ensembles  
3. **Overfitting risk:** Sophisticated models easily overfit to noise in financial data
4. **Trading costs matter:** Real-world transaction costs and slippage would further erode already modest returns

**Potential improvements:**
- Incorporate fundamental data (earnings, economic indicators)
- Use alternative data sources (sentiment, order flow)
- Extend prediction horizon (weekly/monthly vs. daily)
- Implement more sophisticated position sizing and risk management
- Explore deep learning architectures (LSTMs, Transformers) for sequence modeling

## License

MIT
