"""
Alpha Vantage Client

This module provides a client for the Alpha Vantage API.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import functools

from .data_source import DataSource
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class AlphaVantageClient(DataSource):
    """
    Client for the Alpha Vantage API.
    """
    
    def __init__(self, api_key: str = None, rate_limit: int = 5, rate_limit_period: int = 60, cache_size: int = 128):
        """
        Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key (if None, will try to load from environment)
            rate_limit: Number of API calls allowed per rate_limit_period
            rate_limit_period: Period for rate limiting in seconds
            cache_size: Size of the LRU cache for fetch_data
        """
        super().__init__(rate_limit, rate_limit_period, cache_size)
        self.api_key = api_key or self._load_api_key()
        
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        logger.info("Initialized Alpha Vantage client")
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load the API key from environment variables or config.
        
        Returns:
            str: API key if found, None otherwise
        """
        # Try to load from environment variable
        api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        if api_key:
            return api_key
        
        # Try to load from config
        config = get_config()
        if config and 'alpha_vantage' in config and 'api_key' in config['alpha_vantage']:
            return config['alpha_vantage']['api_key']
        
        # Try to load from file
        key_file = os.environ.get('ALPHA_VANTAGE_KEY_FILE')
        if key_file and os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return f.read().strip()
        
        return None
    
    def fetch_data(self, symbol: str, function: str = "TIME_SERIES_DAILY_ADJUSTED", outputsize: str = "full", **kwargs) -> pd.DataFrame:
        """
        Fetch data for a single symbol.
        
        Args:
            symbol: The symbol to fetch data for
            function: Alpha Vantage API function
            outputsize: Output size (compact or full)
            **kwargs: Additional arguments for the API
            
        Returns:
            pd.DataFrame: The fetched data
        """
        self._check_rate_limit()
        
        # Build URL
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
            **kwargs
        }
        
        # Make request
        logger.info(f"Fetching data for {symbol} from Alpha Vantage...")
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for errors
        if "Error Message" in data:
            logger.error(f"Error fetching data for {symbol}: {data['Error Message']}")
            return pd.DataFrame()
        
        # Extract time series data
        if function == "TIME_SERIES_DAILY_ADJUSTED":
            time_series_key = "Time Series (Daily)"
        elif function == "TIME_SERIES_WEEKLY_ADJUSTED":
            time_series_key = "Weekly Adjusted Time Series"
        elif function == "TIME_SERIES_MONTHLY_ADJUSTED":
            time_series_key = "Monthly Adjusted Time Series"
        else:
            time_series_key = next((k for k in data.keys() if k != "Meta Data"), None)
        
        if time_series_key is None or time_series_key not in data:
            logger.error(f"No time series data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        
        # Rename columns
        column_mapping = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adjusted_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
            "8. split coefficient": "split_coefficient"
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Set index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        # Add symbol column
        df["symbol"] = symbol
        
        logger.info(f"Retrieved {len(df)} rows for {symbol}")
        return df
    
    def fetch_batch(self, symbols: List[str], max_workers: int = 1, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with parallel processing.
        
        Args:
            symbols: List of symbols to fetch data for
            max_workers: Maximum number of workers for parallel processing
            **kwargs: Additional arguments for the API
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing if max_workers > 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary mapping futures to symbols
                future_to_symbol = {
                    executor.submit(self.fetch_data, symbol, **kwargs): symbol
                    for symbol in symbols
                }
                
                # Process completed futures
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        df = future.result()
                        if not df.empty:
                            results[symbol] = df
                    except Exception as e:
                        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        else:
            # Use sequential processing
            for symbol in symbols:
                try:
                    df = self.fetch_data(symbol, **kwargs)
                    if not df.empty:
                        results[symbol] = df
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return results
