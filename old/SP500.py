import yfinance as yf
import pandas as pd

# Fetch S&P 500 data
sp500 = yf.Ticker("^GSPC")
data = sp500.history(start="2023-01-01", end="2024-08-27")

# Save to CSV
data['Close'].to_csv('sp500_data.csv')