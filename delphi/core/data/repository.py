"""
Data repository for accessing market data.

This module provides a repository for accessing market data from various sources and storage services.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging

from delphi.core.data.sources.base import DataSource
from delphi.core.data.storage.base import StorageService

# Configure logger
logger = logging.getLogger(__name__)

class MarketDataRepository:
    """Repository for accessing market data."""
    
    def __init__(self, data_source: DataSource, storage_service: StorageService):
        """Initialize the repository.
        
        Args:
            data_source: Data source for fetching data
            storage_service: Storage service for storing and retrieving data
        """
        self.data_source = data_source
        self.storage_service = storage_service
        logger.info(f"Initialized MarketDataRepository with {data_source.__class__.__name__} and {storage_service.__class__.__name__}")
    
    def get_stock_data(self, symbol: str, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Get stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            force_refresh: Whether to force a refresh from the data source
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Try to get data from storage first
            if not force_refresh:
                df = self.storage_service.get_stock_prices(symbol, start_date, end_date)
                if not df.empty:
                    logger.info(f"Retrieved data for {symbol} from storage")
                    return df
            
            # Fetch data from source
            logger.info(f"Fetching data for {symbol} from source")
            df = self.data_source.fetch_daily(symbol, start_date, end_date)
            
            # Validate data
            if not self.data_source.validate_data(df):
                logger.warning(f"Invalid data for {symbol}")
                return pd.DataFrame()
            
            # Store data
            success = self.storage_service.store_stock_prices(symbol, df)
            if success:
                logger.info(f"Stored data for {symbol}")
            else:
                logger.warning(f"Failed to store data for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_batch_stock_data(self, symbols: List[str], start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None, force_refresh: bool = False,
                            max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """Get stock data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (optional)
            end_date: End date (optional)
            force_refresh: Whether to force a refresh from the data source
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        import concurrent.futures
        
        logger.info(f"Getting data for {len(symbols)} symbols with {max_workers} workers")
        
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, start_date, end_date, force_refresh): symbol
                for symbol in symbols
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                    logger.info(f"Successfully got data for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {str(e)}")
                    results[symbol] = pd.DataFrame()
        
        return results
    
    def get_volume_analysis(self, symbol: str, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get volume analysis results for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with volume analysis results
        """
        try:
            # Get volume analysis from storage
            df = self.storage_service.get_volume_analysis(symbol, start_date, end_date)
            if not df.empty:
                logger.info(f"Retrieved volume analysis for {symbol} from storage")
                return df
            
            logger.warning(f"No volume analysis found for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting volume analysis for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def store_volume_analysis(self, symbol: str, df: pd.DataFrame) -> bool:
        """Store volume analysis results for a symbol.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with volume analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store volume analysis
            success = self.storage_service.store_volume_analysis(symbol, df)
            if success:
                logger.info(f"Stored volume analysis for {symbol}")
            else:
                logger.warning(f"Failed to store volume analysis for {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing volume analysis for {symbol}: {str(e)}")
            return False
