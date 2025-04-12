"""
Base model module for Delphi.

This module provides the base classes for all models.
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

class Model(ABC):
    """Base class for all models."""
    
    def __init__(self, cache_size: int = 128, **kwargs):
        """Initialize the model.
        
        Args:
            cache_size: Size of the LRU cache for predict methods
            **kwargs: Additional arguments
        """
        self.cache_size = cache_size
        self.is_trained = False
        
        # Apply caching to predict methods
        self._apply_caching()
        
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    def _apply_caching(self):
        """Apply LRU caching to predict methods."""
        # Apply caching to predict
        if hasattr(self, 'predict'):
            self._predict_impl = self.predict
            self.predict = functools.lru_cache(maxsize=self.cache_size)(self._predict_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for predict methods."""
        if hasattr(self, 'predict'):
            self.predict.cache_clear()
        
        logger.debug(f"Cleared cache for {self.__class__.__name__}")
    
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """Train the model.
        
        Args:
            data: Training data
            **kwargs: Additional arguments
            
        Returns:
            True if training is successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> Any:
        """Make predictions with the model.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Predictions
        """
        pass
    
    def save(self, path: str) -> bool:
        """Save the model to a file.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            
            joblib.dump(self, path)
            logger.info(f"Saved model to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str) -> 'Model':
        """Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        try:
            import joblib
            
            model = joblib.load(path)
            logger.info(f"Loaded model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


class ModelFactory:
    """Factory for creating models."""
    
    _registry = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[Model]):
        """Register a model class.
        
        Args:
            model_type: Type of model
            model_class: Model class
        """
        cls._registry[model_type] = model_class
        logger.debug(f"Registered model type: {model_type}")
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> Model:
        """Create a model.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments for the model
            
        Returns:
            Model instance
        """
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._registry[model_type]
        return model_class(**kwargs)
