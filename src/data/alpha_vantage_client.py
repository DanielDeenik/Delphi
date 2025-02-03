import os
import requests
import pandas as pd
from datetime import datetime
import time

class AlphaVantageClient:
    """Client for fetching market data from Alpha Vantage API"""

    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = 'https://www.alphavantage.co/query'
        self.cache = {}

    def fetch_daily_adjusted(self, symbol):
        """Fetch daily adjusted time series data"""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Time Series (Daily)' not in data:
                print(f"Error fetching data for {symbol}: {data.get('Note', 'Unknown error')}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

            # Map column names - ensure exact matches with Alpha Vantage response
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adjusted_Close',
                '6. volume': 'Volume',
                '7. dividend amount': 'Dividend',
                '8. split coefficient': 'Split_Coefficient'
            }

            # Check for missing columns
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in API response for {symbol}: {missing_columns}")
                return None

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric, replacing any errors with NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Forward fill any NaN values
            df = df.ffill()  # Using ffill() instead of deprecated fillna(method='ffill')

            print(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")

            if 'Volume' not in df.columns:
                print(f"Error: Volume column missing in processed data for {symbol}")
                return None

            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_intraday_data(self, symbol, interval='1min'):
        """Fetch intraday data with cache management"""
        cache_key = f"{symbol}_{interval}"
        now = datetime.now()

        # Check cache
        if cache_key in self.cache:
            last_fetch = self.cache[cache_key]['timestamp']
            if (now - last_fetch).seconds < 60:  # Cache for 1 minute
                return self.cache[cache_key]['data']

        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': 'compact',
            'apikey': self.api_key
        }

        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            key = f'Time Series ({interval})'
            if key not in data:
                print(f"Error fetching intraday data: {data.get('Note', 'Unknown error')}")
                return None

            df = pd.DataFrame.from_dict(data[key], orient='index')

            # Rename columns
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert numeric columns
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Update cache
            self.cache[cache_key] = {
                'data': df,
                'timestamp': now
            }

            return df

        except Exception as e:
            print(f"Error fetching intraday data: {str(e)}")
            return None