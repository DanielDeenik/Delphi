"""
Base classes for data storage.

This module provides base classes and interfaces for data storage.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)

class StorageService(ABC):
    """Base class for all storage services."""
    
    def __init__(self, **kwargs):
        """Initialize the storage service.
        
        Args:
            **kwargs: Additional arguments for the storage service
        """
        self._initialize_storage()
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _initialize_storage(self) -> bool:
        """Initialize the storage.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        pass
    
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
    def store_volume_analysis(self, symbol: str, df: pd.DataFrame) -> bool:
        """Store volume analysis results.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with volume analysis results
            
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
    
    @abstractmethod
    def get_volume_analysis(self, symbol: str, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get volume analysis results.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with volume analysis results
        """
        pass
    
    def store_batch(self, data: Dict[str, pd.DataFrame], max_workers: int = 4) -> Dict[str, bool]:
        """Store data for multiple symbols in parallel.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Dictionary mapping symbols to success status
        """
        import concurrent.futures
        
        logger.info(f"Storing data for {len(data)} symbols with {max_workers} workers")
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.store_stock_prices, symbol, df): symbol
                for symbol, df in data.items()
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    results[symbol] = success
                    logger.info(f"Successfully stored data for {symbol}")
                except Exception as e:
                    logger.error(f"Error storing data for {symbol}: {str(e)}")
                    results[symbol] = False
        
        return results
