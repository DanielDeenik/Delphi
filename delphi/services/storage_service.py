"""
Storage Service Base Class

This module defines the base class for storage services.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

class StorageService(ABC):
    """
    Abstract base class for storage services.
    
    All storage services should inherit from this class and implement the required methods.
    """
    
    @abstractmethod
    def setup_dataset(self) -> bool:
        """
        Set up the dataset.
        
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def setup_table(self) -> bool:
        """
        Set up the table.
        
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store market data for a symbol.
        
        Args:
            symbol: Symbol to store data for
            data: DataFrame with market data
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            pd.DataFrame: Market data
        """
        pass
    
    @abstractmethod
    def get_latest_market_data(self, symbol: str, days: int = 1) -> pd.DataFrame:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            days: Number of days to get
            
        Returns:
            pd.DataFrame: Latest market data
        """
        pass
