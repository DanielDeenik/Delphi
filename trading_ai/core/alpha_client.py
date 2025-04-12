"""
Alpha Vantage API client for fetching market data.
"""
import os
import logging
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time
from functools import lru_cache
import requests
from ratelimit import limits, sleep_and_retry

from trading_ai.config import config_manager

# Configure logging
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Client for fetching market data from Alpha Vantage API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key. If not provided, will use the one from config.
        """
        self.api_key = api_key or config_manager.system_config.alpha_vantage_api_key
        self.base_url = 'https://www.alphavantage.co/query'

        # Rate limiting settings (5 requests per minute for free tier)
        self.calls_per_minute = 5
        self.request_interval = 60 / self.calls_per_minute  # seconds between requests

    @sleep_and_retry
    @limits(calls=5, period=60)
    def _rate_limited_request(self, params: Dict) -> Dict:
        """Make a rate-limited request to Alpha Vantage API."""
        # Implement rate limiting
        current_time = time.time()
        if hasattr(self, 'last_request_time'):
            elapsed = current_time - self.last_request_time
            if elapsed < self.request_interval:
                sleep_time = self.request_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)

        # Make the request
        response = requests.get(self.base_url, params=params)
        self.last_request_time = time.time()

        # Check for HTTP errors
        response.raise_for_status()

        return response.json()

    async def _async_rate_limited_request(self, session: aiohttp.ClientSession, params: Dict) -> Dict:
        """Make an async rate-limited request to Alpha Vantage API."""
        # Add delay to respect rate limits
        await asyncio.sleep(self.request_interval)
        async with session.get(self.base_url, params=params) as response:
            return await response.json()

    def fetch_daily_data_sync(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """Fetch daily stock data from Alpha Vantage synchronously.

        Args:
            symbol: Stock symbol
            outputsize: 'compact' for last 100 data points, 'full' for all available data

        Returns:
            DataFrame with daily stock data
        """
        try:
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
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

            # Sort by date (newest first)
            df = df.sort_index(ascending=False)

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_daily(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """Fetch daily time series data for a symbol.

        Args:
            symbol: Stock symbol
            outputsize: 'compact' (last 100 data points) or 'full' (all data points)

        Returns:
            DataFrame with daily time series data
        """
        try:
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
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

            # Convert index to datetime and make it a column
            df.index = pd.to_datetime(df.index)
            df = df.reset_index()
            df = df.rename(columns={'index': 'date'})

            # Sort by date (newest first)
            df = df.sort_values('date', ascending=False)

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    async def fetch_daily_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """Fetch daily adjusted time series data for a symbol.

        Args:
            symbol: The stock symbol to fetch data for
            outputsize: 'full' for up to 20 years of data, 'compact' for last 100 days

        Returns:
            DataFrame with daily price and volume data
        """
        try:
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }

            # Make the API request
            logger.info(f"Fetching {outputsize} daily data for {symbol}...")
            async with aiohttp.ClientSession() as session:
                data = await self._async_rate_limited_request(session, params)

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

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add symbol column
            df['symbol'] = symbol

            # Sort by date (newest first)
            df = df.sort_index(ascending=False)

            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def fetch_intraday_data(self, symbol: str, interval: str = '5min', outputsize: str = 'full') -> pd.DataFrame:
        """Fetch intraday time series data for a symbol.

        Args:
            symbol: The stock symbol to fetch data for
            interval: Time interval between data points ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'full' or 'compact'

        Returns:
            DataFrame with intraday price and volume data
        """
        try:
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.api_key
            }

            # Make the API request
            logger.info(f"Fetching {outputsize} intraday data for {symbol} at {interval} interval...")
            async with aiohttp.ClientSession() as session:
                data = await self._async_rate_limited_request(session, params)

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

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add symbol column
            df['symbol'] = symbol

            # Sort by date (newest first)
            df = df.sort_index(ascending=False)

            logger.info(f"Successfully fetched {len(df)} rows of intraday data for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def fetch_company_overview(self, symbol: str) -> Dict:
        """Fetch company overview data for a symbol.

        Args:
            symbol: The stock symbol to fetch data for

        Returns:
            Dictionary with company information
        """
        try:
            # Build request parameters
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }

            # Make the API request
            logger.info(f"Fetching company overview for {symbol}...")
            async with aiohttp.ClientSession() as session:
                data = await self._async_rate_limited_request(session, params)

            # Check for errors
            if not data or 'Symbol' not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching company overview for {symbol}: {error_msg}")
                return {}

            logger.info(f"Successfully fetched company overview for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {str(e)}")
            return {}

    async def fetch_listing_status(self) -> pd.DataFrame:
        """Fetch listing status for all US stocks.

        Returns:
            DataFrame with listing status information
        """
        try:
            # Build request parameters
            params = {
                'function': 'LISTING_STATUS',
                'apikey': self.api_key
            }

            # Make the API request
            logger.info("Fetching listing status...")
            async with aiohttp.ClientSession() as session:
                response = await session.get(self.base_url, params=params)
                if response.status != 200:
                    logger.error(f"Error fetching listing status: {response.status}")
                    return pd.DataFrame()

                csv_data = await response.text()
                df = pd.read_csv(pd.StringIO(csv_data))

            # Add timestamp
            df['timestamp'] = datetime.now()

            logger.info(f"Successfully fetched listing status for {len(df)} symbols")
            return df

        except Exception as e:
            logger.error(f"Error fetching listing status: {str(e)}")
            return pd.DataFrame()

    async def batch_fetch_daily_data(self, symbols: List[str], outputsize: str = 'full') -> Dict[str, pd.DataFrame]:
        """Fetch daily data for multiple symbols with rate limiting.

        Args:
            symbols: List of stock symbols to fetch data for
            outputsize: 'full' or 'compact'

        Returns:
            Dictionary mapping symbols to their respective DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                df = await self.fetch_daily_data(symbol, outputsize)
                if not df.empty:
                    results[symbol] = df

                # Add a delay to respect rate limits
                await asyncio.sleep(self.request_interval)

            except Exception as e:
                logger.error(f"Error in batch fetch for {symbol}: {str(e)}")

        return results
