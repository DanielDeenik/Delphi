"""
Base classes for data sources.

This module provides base classes and interfaces for data sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logger
logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Base class for all data sources."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the data source.
        
        Args:
            api_key: API key for the data source
            **kwargs: Additional arguments for the data source
        """
        self.api_key = api_key
        self._validate_credentials()
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def _validate_credentials(self) -> bool:
        """Validate the credentials for the data source.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def fetch_daily(self, symbol: str, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """Fetch daily data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            **kwargs: Additional arguments for the data source
            
        Returns:
            DataFrame with daily data
        """
        pass
    
    @abstractmethod
    def fetch_intraday(self, symbol: str, interval: str = '1min', 
                      start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """Fetch intraday data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval (e.g., '1min', '5min', '15min', '30min', '60min')
            start_date: Start date (optional)
            end_date: End date (optional)
            **kwargs: Additional arguments for the data source
            
        Returns:
            DataFrame with intraday data
        """
        pass
    
    def fetch_batch(self, symbols: List[str], max_workers: int = 4, **kwargs) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            max_workers: Maximum number of workers for parallel processing
            **kwargs: Additional arguments for the data source
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        import concurrent.futures
        
        logger.info(f"Fetching data for {len(symbols)} symbols with {max_workers} workers")
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.fetch_daily, symbol, **kwargs): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    results[symbol] = pd.DataFrame()
        
        return results
    
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
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values
        null_columns = [col for col in required_columns if df[col].isnull().any()]
        if null_columns:
            logger.warning(f"Null values in columns: {null_columns}")
            return False
        
        return True
