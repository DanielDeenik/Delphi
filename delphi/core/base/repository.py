"""
Base repository module for Delphi.

This module provides the base class for all repositories.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging
import functools

from delphi.core.base.data_source import DataSource
from delphi.core.base.storage import StorageService

# Configure logger
logger = logging.getLogger(__name__)

class Repository(ABC):
    """Base class for all repositories."""
    
    def __init__(self, data_source: Optional[DataSource] = None, 
                storage_service: Optional[StorageService] = None,
                cache_size: int = 128, **kwargs):
        """Initialize the repository.
        
        Args:
            data_source: Data source for fetching data
            storage_service: Storage service for storing and retrieving data
            cache_size: Size of the LRU cache for repository methods
            **kwargs: Additional arguments
        """
        self.data_source = data_source
        self.storage_service = storage_service
        self.cache_size = cache_size
        
        # Apply caching to repository methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _apply_caching(self):
        """Apply LRU caching to repository methods."""
        # Apply caching to get_data
        if hasattr(self, 'get_data'):
            self._get_data_impl = self.get_data
            self.get_data = functools.lru_cache(maxsize=self.cache_size)(self._get_data_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for repository methods."""
        if hasattr(self, 'get_data'):
            self.get_data.cache_clear()
        
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    @abstractmethod
    def get_data(self, **kwargs) -> Any:
        """Get data from the repository.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Data from the repository
        """
        pass
    
    @abstractmethod
    def store_data(self, data: Any, **kwargs) -> bool:
        """Store data in the repository.
        
        Args:
            data: Data to store
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        pass
