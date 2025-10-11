# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for nicer plots
sns.set_style('whitegrid')

# --------------------
# 1. Load data
# --------------------

# Example: load stock price CSVs
# Suppose you have CSV files like “AAPL.csv”, “MSFT.csv”, etc.
def load_stock_data(filepaths: dict) -> dict:
    """
    filepaths: dict of {ticker: path_to_csv}
    returns dict of DataFrames, each indexed by Date
    """
    stocks = {}
    for ticker, path in filepaths.items():
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        # ensure sorted
        df = df.sort_index()
        stocks[ticker] = df
    return stocks

# Example usage
filepaths = {
    'AAPL': 'data/AAPL.csv',
    'MSFT': 'data/MSFT.csv',
    # add more
}
stocks = load_stock_data(filepaths)

# --------------------
# 2. Preprocess / align data
# --------------------

def get_close_prices(stocks: dict) -> pd.DataFrame:
    """
    Given dict of stocks DataFrames, return a DataFrame of closing prices
    columns = tickers, index = Date
    """
    df_close = pd.DataFrame({ticker: df['Close'] for ticker, df in stocks.items()})
    return df_close

df_close = get_close_prices(stocks)

# forward‑fill / backward‑fill missing data
df_close = df_close.ffill().bfill()

# --------------------
# 3. Compute returns
# --------------------

def compute_returns(prices: pd.DataFrame, freq: str = 'daily') -> pd.DataFrame:
    """
    freq: 'daily' for simple returns, or 'monthly'
    """
    if freq == 'daily':
        returns = prices.pct_change().dropna()
    elif freq == 'monthly':
        returns = prices.resample('M').ffill().pct_change().dropna()
    else:
        raise ValueError("freq must be 'daily' or 'monthly'")
    return returns

df_returns = compute_returns(df_close, freq='daily')

# --------------------
# 4. Performance metrics
# --------------------

def compute_rolling_metrics(returns: pd.DataFrame, window: int = 30):
    """
    For each ticker, compute rolling mean & volatility
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_vol = returns.rolling(window).std()
    return rolling_mean, rolling_vol

rolling_mean, rolling_vol = compute_rolling_metrics(df_returns, window=30)

def sharpe_ratio(returns: pd.DataFrame, risk_free_rate: float = 0.0):
    """
    Compute (mean(r) - rf) / std(r)
    """
    excess = returns - risk_free_rate
    return excess.mean() / excess.std()

sr = sharpe_ratio(df_returns)

# --------------------
# 5. Correlation & macro integration
# --------------------

# Suppose you have macro data CSVs
macro = pd.read_csv('data/macro.csv', parse_dates=['Date'], index_col='Date')  
# e.g. columns: GDP, Inflation, Unemployment, etc.

# Align macro and returns
macro_monthly = macro.resample('M').ffill()
returns_monthly = compute_returns(df_close, freq='monthly')

# Combine (example: align dates)
combined = returns_monthly.join(macro_monthly, how='inner')

# Compute correlation between macro variables and market returns
corr_mat = combined.corr()

# --------------------
# 6. Visualization
# --------------------

def plot_time_series(prices: pd.DataFrame):
    prices.plot(figsize=(12, 6))
    plt.title("Stock Prices Over Time")
    plt.ylabel("Price")
    plt.show()

def plot_correlation_matrix(corr: pd.DataFrame):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

# Use the plotting functions
plot_time_series(df_close)
plot_correlation_matrix(corr_mat)

# --------------------
# 7. Exporting results
# --------------------

# Save correlation matrix to Excel
corr_mat.to_excel('output/correlation_matrix.xlsx')

# Save key DataFrames
rolling_mean.to_csv('output/rolling_mean.csv')
rolling_vol.to_csv('output/rolling_vol.csv')
sr.to_csv('output/sharpe_ratio.csv')  # note: sr is a Series

# --------------------
# 8. (Optional) Interactive / dashboard
# --------------------

# You can wrap some of this into Streamlit or Dash to make interactive plots.
