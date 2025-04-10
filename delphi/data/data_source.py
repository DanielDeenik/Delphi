"""
Data Source Base Class

This module defines the base class for data sources.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional
import functools
import time
import logging

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    All data sources should inherit from this class and implement the required methods.
    """
    
    def __init__(self, rate_limit: int = 5, rate_limit_period: int = 60, cache_size: int = 128):
        """
        Initialize the data source.
        
        Args:
            rate_limit: Number of API calls allowed per rate_limit_period
            rate_limit_period: Period for rate limiting in seconds
            cache_size: Size of the LRU cache for fetch_data
        """
        self.rate_limit = rate_limit
        self.rate_limit_period = rate_limit_period
        self.call_count = 0
        self.last_call_time = 0
        
        # Apply caching to fetch_data
        self.fetch_data = functools.lru_cache(maxsize=cache_size)(self.fetch_data)
    
    def _check_rate_limit(self):
        """
        Check if we've hit the rate limit and wait if necessary.
        """
        self.call_count += 1
        current_time = time.time()
        
        if self.call_count % self.rate_limit == 0:
            elapsed = current_time - self.last_call_time
            if elapsed < self.rate_limit_period:
                wait_time = self.rate_limit_period - elapsed
                logger.info(f"Rate limit reached. Waiting for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        self.last_call_time = time.time()
    
    @abstractmethod
    def fetch_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """
        Fetch data for a single symbol.
        
        Args:
            symbol: The symbol to fetch data for
            **kwargs: Additional arguments for the data source
            
        Returns:
            pd.DataFrame: The fetched data
        """
        pass
    
    def fetch_batch(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols to fetch data for
            **kwargs: Additional arguments for the data source
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_data(symbol, **kwargs)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return results
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: Whether the data is valid
        """
        if df is None or df.empty:
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                return False
        
        return True
