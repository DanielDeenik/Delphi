"""
Base service module for Delphi.

This module provides the base class for all services.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
import functools

# Configure logger
logger = logging.getLogger(__name__)

class Service(ABC):
    """Base class for all services."""
    
    def __init__(self, cache_size: int = 128, **kwargs):
        """Initialize the service.
        
        Args:
            cache_size: Size of the LRU cache for service methods
            **kwargs: Additional arguments
        """
        self.cache_size = cache_size
        
        # Apply caching to service methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _apply_caching(self):
        """Apply LRU caching to service methods."""
        # This method should be overridden by subclasses to apply caching to specific methods
        pass
    
    def clear_cache(self):
        """Clear the LRU cache for service methods."""
        # This method should be overridden by subclasses to clear caches for specific methods
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the service.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            True if initialization is successful, False otherwise
        """
        pass
