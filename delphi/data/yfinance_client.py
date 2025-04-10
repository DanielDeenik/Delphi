"""
Yahoo Finance Client

This module provides a client for the Yahoo Finance API using yfinance.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import functools

from .data_source import DataSource

logger = logging.getLogger(__name__)

class YFinanceClient(DataSource):
    """
    Client for the Yahoo Finance API using yfinance.
    """
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the Yahoo Finance client.
        
        Args:
            cache_size: Size of the LRU cache for fetch_data
        """
        super().__init__(rate_limit=0, rate_limit_period=0, cache_size=cache_size)
        
        # Import yfinance here to avoid dependency issues
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            logger.error("yfinance not installed. Please install it with 'pip install yfinance'.")
            raise
        
        logger.info("Initialized Yahoo Finance client")
    
    def fetch_data(self, symbol: str, period: str = "max", interval: str = "1d", **kwargs) -> pd.DataFrame:
        """
        Fetch data for a single symbol.
        
        Args:
            symbol: The symbol to fetch data for
            period: Period to fetch (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Interval between data points (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            **kwargs: Additional arguments for yfinance
            
        Returns:
            pd.DataFrame: The fetched data
        """
        # No need to check rate limit for yfinance
        
        # Make request
        logger.info(f"Fetching data for {symbol} from Yahoo Finance...")
        
        try:
            # Download data
            df = self.yf.download(symbol, period=period, interval=interval, **kwargs)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to match our standard format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adjusted_close',
                'Volume': 'volume'
            })
            
            # Add missing columns
            if 'dividend' not in df.columns:
                df['dividend'] = 0.0
            if 'split_coefficient' not in df.columns:
                df['split_coefficient'] = 1.0
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Set date as index
            df = df.set_index('date')
            
            logger.info(f"Retrieved {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_batch(self, symbols: List[str], max_workers: int = 4, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols with parallel processing.
        
        Args:
            symbols: List of symbols to fetch data for
            max_workers: Maximum number of workers for parallel processing
            **kwargs: Additional arguments for yfinance
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
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
        
        return results
