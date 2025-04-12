"""
Alpha Vantage data source.

This module provides a data source for Alpha Vantage.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import logging
import os

from delphi.core.data.sources.base import DataSource

# Configure logger
logger = logging.getLogger(__name__)

class AlphaVantageClient(DataSource):
    """Client for Alpha Vantage API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://www.alphavantage.co/query',
                request_interval: float = 0.2, **kwargs):
        """Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (default: from environment variable)
            base_url: Base URL for the API
            request_interval: Interval between requests in seconds
            **kwargs: Additional arguments
        """
        self.base_url = base_url
        self.request_interval = request_interval
        self.last_request_time = 0
        
        # Get API key from environment variable if not provided
        if api_key is None:
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        
        super().__init__(api_key=api_key, **kwargs)
    
    def _validate_credentials(self) -> bool:
        """Validate the API key.
        
        Returns:
            True if API key is valid, False otherwise
        """
        if not self.api_key:
            logger.error("Alpha Vantage API key not provided")
            return False
        
        return True
    
    def _rate_limited_request(self, params: Dict) -> Dict:
        """Make a rate-limited request to Alpha Vantage API.
        
        Args:
            params: Request parameters
            
        Returns:
            Response JSON
        """
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        
        # Make request
        self.last_request_time = time.time()
        response = requests.get(self.base_url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            logger.error(f"Error in Alpha Vantage API request: {response.status_code} {response.text}")
            return {}
        
        return response.json()
    
    def fetch_daily(self, symbol: str, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """Fetch daily data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with daily data
        """
        try:
            # Determine output size
            outputsize = kwargs.get('outputsize', 'full')
            if start_date and (datetime.now() - start_date).days < 100:
                outputsize = 'compact'
            
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize
            }
            
            # Make the API request
            logger.info(f"Fetching {outputsize} daily data for {symbol}...")
            data = self._rate_limited_request(params)
            
            # Check for errors
            if 'Time Series (Daily)' not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching data for {symbol}: {error_msg}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            
            # Map column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split_coefficient'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Filter by date range
            if start_date:
                df = df[df['date'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['date'] <= pd.Timestamp(end_date)]
            
            # Sort by date (newest first)
            df = df.sort_values('date', ascending=False)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_intraday(self, symbol: str, interval: str = '1min', 
                      start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """Fetch intraday data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (e.g., '1min', '5min', '15min', '30min', '60min')
            start_date: Start date (optional)
            end_date: End date (optional)
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with intraday data
        """
        try:
            # Determine output size
            outputsize = kwargs.get('outputsize', 'full')
            if start_date and (datetime.now() - start_date).days < 1:
                outputsize = 'compact'
            
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }
            
            # Make the API request
            logger.info(f"Fetching {outputsize} intraday data for {symbol} with interval {interval}...")
            data = self._rate_limited_request(params)
            
            # Check for errors
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching intraday data for {symbol}: {error_msg}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            
            # Map column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            df.index.name = 'datetime'
            
            # Reset index to make datetime a column
            df = df.reset_index()
            
            # Add date column
            df['date'] = df['datetime'].dt.date
            
            # Filter by date range
            if start_date:
                df = df[df['datetime'] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.Timestamp(end_date)]
            
            # Sort by datetime (newest first)
            df = df.sort_values('datetime', ascending=False)
            
            logger.info(f"Successfully fetched {len(df)} rows of intraday data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
