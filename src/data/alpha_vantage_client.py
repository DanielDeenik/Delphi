import os
import requests
import pandas as pd
from datetime import datetime
import logging
from ..services.sqlite_storage_service import SQLiteStorageService
from ..services.timeseries_storage_service import TimeSeriesStorageService

logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Client for fetching market data from Alpha Vantage API with cloud storage integration"""

    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = 'https://www.alphavantage.co/query'

        # Use SQLite as primary storage
        self.storage_service = SQLiteStorageService()

        # Keep BigQuery storage for future migration
        self.bigquery_storage = TimeSeriesStorageService() if os.getenv('USE_BIGQUERY', 'false').lower() == 'true' else None

    def fetch_daily_adjusted(self, symbol, force_refresh=False):
        """
        Fetch daily adjusted time series data with cloud storage integration

        Args:
            symbol: Trading symbol
            force_refresh: If True, bypass cache and fetch fresh data
        """
        try:
            # Check cloud storage first if not forcing refresh
            if not force_refresh:
                end_date = datetime.now()
                start_date = end_date - pd.Timedelta(days=90)  # Last 90 days
                cached_data = self.storage_service.get_market_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if not cached_data.empty:
                    logger.info(f"Retrieved cached data for {symbol} from SQLite storage")
                    return cached_data

            # Fetch from Alpha Vantage if cache miss or force refresh
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Time Series (Daily)' not in data:
                logger.error(f"Error fetching data for {symbol}: {data.get('Note', 'Unknown error')}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

            # Map column names
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
                logger.warning(f"Missing columns in API response for {symbol}: {missing_columns}")
                return None

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Forward fill any NaN values
            df = df.ffill()

            logger.info(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")

            # Store in SQLite
            if self.storage_service.store_market_data(symbol, df):
                logger.info(f"Successfully stored {symbol} data in SQLite")
            else:
                logger.warning(f"Failed to store {symbol} data in SQLite")

            # Also store in BigQuery if enabled
            if self.bigquery_storage:
                if self.bigquery_storage.store_market_data(symbol, df):
                    logger.info(f"Successfully stored {symbol} data in BigQuery")
                else:
                    logger.warning(f"Failed to store {symbol} data in BigQuery")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_intraday_data(self, symbol, interval='1min'):
        """Fetch intraday data with cloud storage integration"""
        try:
            # Check cloud storage first
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(hours=1)  # Last hour
            cached_data = self.storage_service.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            if not cached_data.empty:
                return cached_data

            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': 'compact',
                'apikey': self.api_key
            }

            response = requests.get(self.base_url, params=params)
            data = response.json()

            key = f'Time Series ({interval})'
            if key not in data:
                logger.error(f"Error fetching intraday data: {data.get('Note', 'Unknown error')}")
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

            # Store in SQLite
            if self.storage_service.store_market_data(symbol, df):
                logger.info(f"Successfully stored intraday data for {symbol} in SQLite")
            else:
                logger.warning(f"Failed to store intraday data for {symbol} in SQLite")

            # Also store in BigQuery if enabled
            if self.bigquery_storage:
                if self.bigquery_storage.store_market_data(symbol, df, partition_size='1H'):
                    logger.info(f"Successfully stored intraday data for {symbol} in BigQuery")
                else:
                    logger.warning(f"Failed to store intraday data for {symbol} in BigQuery")

            return df

        except Exception as e:
            logger.error(f"Error fetching intraday data: {str(e)}")
            return None