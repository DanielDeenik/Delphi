"""
Base data source module for Delphi.

This module provides the base class for all data sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging
import functools

# Configure logger
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Base class for all data sources."""
    
    def __init__(self, api_key: Optional[str] = None, cache_size: int = 128, **kwargs):
        """Initialize the data source.
        
        Args:
            api_key: API key for the data source
            cache_size: Size of the LRU cache for fetch methods
            **kwargs: Additional arguments
        """
        self.api_key = api_key
        self.cache_size = cache_size
        
        # Validate credentials
        if not self._validate_credentials():
            logger.warning("Invalid or missing credentials")
        
        # Apply caching to fetch methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _validate_credentials(self) -> bool:
        """Validate the credentials.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        return self.api_key is not None
    
    def _apply_caching(self):
        """Apply LRU caching to fetch methods."""
        # Apply caching to fetch_daily
        if hasattr(self, 'fetch_daily'):
            self._fetch_daily_impl = self.fetch_daily
            self.fetch_daily = functools.lru_cache(maxsize=self.cache_size)(self._fetch_daily_impl)
        
        # Apply caching to fetch_intraday
        if hasattr(self, 'fetch_intraday'):
            self._fetch_intraday_impl = self.fetch_intraday
            self.fetch_intraday = functools.lru_cache(maxsize=self.cache_size)(self._fetch_intraday_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for fetch methods."""
        if hasattr(self, 'fetch_daily'):
            self.fetch_daily.cache_clear()
        
        if hasattr(self, 'fetch_intraday'):
            self.fetch_intraday.cache_clear()
        
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    @abstractmethod
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
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        
        # Check for required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for NaN values
        if df[required_columns].isna().any().any():
            logger.warning("DataFrame contains NaN values")
            return False
        
        return True
