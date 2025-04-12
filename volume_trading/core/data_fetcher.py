"""
Data fetcher for retrieving stock data from Alpha Vantage.
"""
import os
import logging
import requests
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import time

from volume_trading.config import config

# Configure logging
logger = logging.getLogger(__name__)

class AlphaVantageClient:
    """Client for fetching data from Alpha Vantage API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (optional, will use from config if not provided)
        """
        self.api_key = api_key or config.get("alpha_vantage_api_key")
        self.base_url = "https://www.alphavantage.co/query"
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
    
    def fetch_daily_data(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """Fetch daily stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' for last 100 data points, 'full' for all available data
            
        Returns:
            DataFrame with daily stock data
        """
        try:
            # Build request parameters
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            # Make request
            logger.info(f"Fetching daily data for {symbol}...")
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for errors
            if "Error Message" in data:
                logger.error(f"Error fetching data for {symbol}: {data['Error Message']}")
                return pd.DataFrame()
            
            if "Note" in data:
                logger.warning(f"API limit reached: {data['Note']}")
                return pd.DataFrame()
            
            # Parse data
            if "Time Series (Daily)" not in data:
                logger.error(f"Unexpected response format for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            
            # Rename columns
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. adjusted close": "adjusted_close",
                "6. volume": "volume",
                "7. dividend amount": "dividend",
                "8. split coefficient": "split_coefficient"
            })
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date (newest first)
            df = df.sort_index(ascending=False)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_intraday_data(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> pd.DataFrame:
        """Fetch intraday stock data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' for last 100 data points, 'full' for all available data
            
        Returns:
            DataFrame with intraday stock data
        """
        try:
            # Build request parameters
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": self.api_key
            }
            
            # Make request
            logger.info(f"Fetching intraday data for {symbol} at {interval} interval...")
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for errors
            if "Error Message" in data:
                logger.error(f"Error fetching data for {symbol}: {data['Error Message']}")
                return pd.DataFrame()
            
            if "Note" in data:
                logger.warning(f"API limit reached: {data['Note']}")
                return pd.DataFrame()
            
            # Parse data
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                logger.error(f"Unexpected response format for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
            
            # Rename columns
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date (newest first)
            df = df.sort_index(ascending=False)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: list, function: str = "daily", **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            function: 'daily' or 'intraday'
            **kwargs: Additional arguments for the fetch function
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch data
                if function == "daily":
                    df = self.fetch_daily_data(symbol, **kwargs)
                elif function == "intraday":
                    df = self.fetch_intraday_data(symbol, **kwargs)
                else:
                    logger.error(f"Invalid function: {function}")
                    continue
                
                # Add to results
                if not df.empty:
                    results[symbol] = df
                
                # Respect API rate limits (5 calls per minute for free tier)
                time.sleep(12)  # 60 seconds / 5 calls = 12 seconds per call
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return results
