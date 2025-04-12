"""
Base classes for analysis models.

This module provides base classes and interfaces for analysis models.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
import functools
import hashlib
import json

# Configure logger
logger = logging.getLogger(__name__)

class Analyzer(ABC):
    """Base class for all analyzers."""
    
    def __init__(self, cache_size: int = 128, **kwargs):
        """Initialize the analyzer.
        
        Args:
            cache_size: Size of the LRU cache for analyze methods
            **kwargs: Additional arguments for the analyzer
        """
        self.cache_size = cache_size
        self._initialize_cache()
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def _initialize_cache(self):
        """Initialize the LRU cache."""
        try:
            from functools import lru_cache
            
            # Create a cache key function
            def cache_key(*args, **kwargs):
                # Convert args and kwargs to a string representation
                key_parts = [str(arg) for arg in args]
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key_str = ",".join(key_parts)
                
                # Hash the string representation
                return hashlib.md5(key_str.encode()).hexdigest()
            
            # Apply LRU cache to the analyze method
            self._analyze = lru_cache(maxsize=self.cache_size)(self._analyze)
            logger.info(f"Initialized LRU cache with size {self.cache_size}")
        except Exception as e:
            logger.error(f"Error initializing cache: {str(e)}")
            # Fall back to no caching
            self._analyze = self._analyze
    
    @abstractmethod
    def _analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Internal analyze method.
        
        This method is wrapped with LRU cache.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Analyze data.
        
        This method calls the cached _analyze method.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with analysis results
        """
        try:
            return self._analyze(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear the LRU cache."""
        try:
            self._analyze.cache_clear()
            logger.info("Cleared LRU cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")


class AnalysisStrategy(ABC):
    """Base class for analysis strategies."""
    
    def __init__(self, **kwargs):
        """Initialize the analysis strategy.
        
        Args:
            **kwargs: Additional arguments for the strategy
        """
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze data.
        
        Args:
            data: DataFrame to analyze
            **kwargs: Additional arguments for the analysis
            
        Returns:
            Dictionary with analysis results
        """
        pass


class ModelFactory:
    """Factory for creating analysis models."""
    
    @staticmethod
    def create_analyzer(analyzer_type: str, **kwargs) -> Analyzer:
        """Create an analyzer.
        
        Args:
            analyzer_type: Type of analyzer to create
            **kwargs: Additional arguments for the analyzer
            
        Returns:
            Analyzer instance
        """
        from delphi.core.models.volume.analyzer import VolumeAnalyzer
        from delphi.core.models.correlation.analyzer import CorrelationAnalyzer
        
        if analyzer_type == "volume":
            return VolumeAnalyzer(**kwargs)
        elif analyzer_type == "correlation":
            return CorrelationAnalyzer(**kwargs)
        else:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> AnalysisStrategy:
        """Create an analysis strategy.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional arguments for the strategy
            
        Returns:
            AnalysisStrategy instance
        """
        from delphi.core.models.volume.strategies import SimpleVolumeStrategy, MLVolumeStrategy
        
        if strategy_type == "simple_volume":
            return SimpleVolumeStrategy(**kwargs)
        elif strategy_type == "ml_volume":
            return MLVolumeStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
