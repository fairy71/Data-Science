import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load stock data ---
ticker = "AAPL"  # Apple Inc.
period = "6mo"   # Last 6 months
interval = "1d"  # Daily data

df = yf.download(ticker, period=period, interval=interval)
print("Raw data:")
print(df.head())

# --- 2. Calculate Moving Averages & Volatility ---
df['MA20'] = df['Close'].rolling(window=20).mean()        # 20-day Moving Average
df['MA50'] = df['Close'].rolling(window=50).mean()        # 50-day Moving Average
df['Volatility'] = df['Close'].rolling(window=20).std()   # 20-day Rolling Volatility

# --- 3. Plot the Data ---
plt.figure(figsize=(14, 7))

# Price and MAs
plt.plot(df['Close'], label='Closing Price', color='blue', linewidth=1.5)
plt.plot(df['MA20'], label='MA20', color='orange', linestyle='--')
plt.plot(df['MA50'], label='MA50', color='green', linestyle='--')

# Plot title and labels
plt.title(f"{ticker} Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

# --- 4. Optional: Plot Volatility ---
plt.figure(figsize=(14, 4))
plt.plot(df['Volatility'], label='Volatility (20-day STD)', color='red')
plt.title(f"{ticker} Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.tight_layout()
plt.show()
