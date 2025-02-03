import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.cache = {}

    def fetch_stock_data(self, symbol, period='1mo', interval='1h'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            # Reset index to make datetime accessible as 'Date' column
            df.reset_index(inplace=True)
            if 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'Date'}, inplace=True)

            # Add O'Neil's volume analysis features
            df = self.prepare_volume_features(df)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_volume_features(self, df):
        """Prepare volume-related features for analysis"""
        # Basic volume features
        df['log_volume'] = np.log1p(df['Volume'])
        df['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ma50'] = df['Volume'].rolling(window=50).mean()

        # O'Neil's volume ratios
        df['volume_ratio'] = df['Volume'] / df['volume_ma20']
        df['volume_trend'] = (df['Volume'] - df['volume_ma20']) / df['volume_ma20']

        # Price-volume relationship
        df['price_volume_trend'] = df['Close'].pct_change() * df['volume_ratio']

        # Accumulation/Distribution Line
        df['money_flow_multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['Volume']
        df['acc_dist_line'] = df['money_flow_volume'].cumsum()

        return df

    def get_real_time_data(self, symbol):
        """Get real-time data with cache management"""
        now = datetime.now()
        cache_ttl = timedelta(minutes=1)

        if symbol not in self.cache or (now - self.cache[symbol]['timestamp']) > cache_ttl:
            df = self.fetch_stock_data(symbol, period='1d', interval='1m')
            if df is not None:
                self.cache[symbol] = {
                    'data': df,
                    'timestamp': now
                }

        return self.cache.get(symbol, {}).get('data', None)