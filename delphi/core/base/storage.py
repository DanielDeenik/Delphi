"""
Base storage module for Delphi.

This module provides the base class for all storage services.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging
import functools

# Configure logger
logger = logging.getLogger(__name__)

class StorageService(ABC):
    """Base class for all storage services."""
    
    def __init__(self, cache_size: int = 128, **kwargs):
        """Initialize the storage service.
        
        Args:
            cache_size: Size of the LRU cache for get methods
            **kwargs: Additional arguments
        """
        self.cache_size = cache_size
        
        # Initialize storage
        self._initialize_storage()
        
        # Apply caching to get methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _initialize_storage(self) -> bool:
        """Initialize the storage.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        return True
    
    def _apply_caching(self):
        """Apply LRU caching to get methods."""
        # Apply caching to get_stock_prices
        if hasattr(self, 'get_stock_prices'):
            self._get_stock_prices_impl = self.get_stock_prices
            self.get_stock_prices = functools.lru_cache(maxsize=self.cache_size)(self._get_stock_prices_impl)
        
        # Apply caching to get_volume_analysis
        if hasattr(self, 'get_volume_analysis'):
            self._get_volume_analysis_impl = self.get_volume_analysis
            self.get_volume_analysis = functools.lru_cache(maxsize=self.cache_size)(self._get_volume_analysis_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for get methods."""
        if hasattr(self, 'get_stock_prices'):
            self.get_stock_prices.cache_clear()
        
        if hasattr(self, 'get_volume_analysis'):
            self.get_volume_analysis.cache_clear()
        
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    @abstractmethod
    def store_stock_prices(self, symbol: str, df: pd.DataFrame) -> bool:
        """Store stock price data.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with price data
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stock_prices(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get stock price data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with price data
        """
        pass
