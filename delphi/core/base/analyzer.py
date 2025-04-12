"""
Base analyzer module for Delphi.

This module provides the base classes for all analyzers and strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Type
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import functools

# Configure logger
logger = logging.getLogger(__name__)

class AnalysisStrategy(ABC):
    """Base class for all analysis strategies."""
    
    def __init__(self, **kwargs):
        """Initialize the analysis strategy.
        
        Args:
            **kwargs: Additional arguments
        """
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze data.
        
        Args:
            data: Data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        pass


class Analyzer(ABC):
    """Base class for all analyzers."""
    
    def __init__(self, strategy: Optional[AnalysisStrategy] = None, cache_size: int = 128, **kwargs):
        """Initialize the analyzer.
        
        Args:
            strategy: Analysis strategy to use
            cache_size: Size of the LRU cache for analyze methods
            **kwargs: Additional arguments
        """
        self.strategy = strategy
        self.cache_size = cache_size
        
        # Apply caching to analyze methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _apply_caching(self):
        """Apply LRU caching to analyze methods."""
        # Apply caching to analyze
        self._analyze_impl = self.analyze
        self.analyze = functools.lru_cache(maxsize=self.cache_size)(self._analyze_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for analyze methods."""
        self.analyze.cache_clear()
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze data.
        
        Args:
            data: Data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate data
            if data.empty:
                logger.warning("DataFrame is empty")
                return {"error": "DataFrame is empty"}
            
            # Use strategy to analyze data
            if self.strategy:
                return self.strategy.analyze(data, **kwargs)
            else:
                return self._analyze(data, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {"error": str(e)}
    
    @abstractmethod
    def _analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze data (implementation).
        
        Args:
            data: Data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        pass
