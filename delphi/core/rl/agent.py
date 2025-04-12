"""
Reinforcement learning agent module for Delphi.

This module provides the base class for all reinforcement learning agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import torch

from delphi.core.base.model import Model
from delphi.core.rl.environment import TradingEnvironment

# Configure logger
logger = logging.getLogger(__name__)

class RLAgent(Model, ABC):
    """Base class for all reinforcement learning agents."""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu', **kwargs):
        """Initialize the reinforcement learning agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            device: Device to use for training ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        logger.info(f"Initialized {self.__class__.__name__} with state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action based on the current state.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (no exploration)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update the agent's parameters.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary with update metrics
        """
        pass
    
    def train(self, data: pd.DataFrame, **kwargs) -> bool:
        """Train the agent on historical data.
        
        Args:
            data: Training data
            **kwargs: Additional arguments
            
        Returns:
            True if training is successful, False otherwise
        """
        try:
            # Create environment
            env = TradingEnvironment(data, **kwargs)
            
            # Train agent
            self._train_agent(env, **kwargs)
            
            # Set trained flag
            self.is_trained = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error training agent: {str(e)}")
            return False
    
    @abstractmethod
    def _train_agent(self, env: TradingEnvironment, **kwargs) -> None:
        """Train the agent on an environment.
        
        Args:
            env: Training environment
            **kwargs: Additional arguments
        """
        pass
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Make predictions with the agent.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Check if agent is trained
            if not self.is_trained:
                logger.warning("Agent is not trained")
                return {"error": "Agent is not trained"}
            
            # Create environment
            env = TradingEnvironment(data, **kwargs)
            
            # Evaluate agent
            return self._evaluate_agent(env, **kwargs)
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {"error": str(e)}
    
    @abstractmethod
    def _evaluate_agent(self, env: TradingEnvironment, **kwargs) -> Dict[str, Any]:
        """Evaluate the agent on an environment.
        
        Args:
            env: Evaluation environment
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    def save(self, path: str) -> bool:
        """Save the agent to a file.
        
        Args:
            path: Path to save the agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'state_dict': self.state_dict(),
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'is_trained': self.is_trained
            }, path)
            
            logger.info(f"Saved agent to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'RLAgent':
        """Load an agent from a file.
        
        Args:
            path: Path to load the agent from
            **kwargs: Additional arguments
            
        Returns:
            Loaded agent
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location='cpu')
            
            # Create agent
            agent = cls(
                state_dim=checkpoint['state_dim'],
                action_dim=checkpoint['action_dim'],
                **kwargs
            )
            
            # Load state dict
            agent.load_state_dict(checkpoint['state_dict'])
            
            # Set trained flag
            agent.is_trained = checkpoint.get('is_trained', False)
            
            logger.info(f"Loaded agent from {path}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent: {str(e)}")
            return None
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get the agent's state dictionary.
        
        Returns:
            State dictionary
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load a state dictionary into the agent.
        
        Args:
            state_dict: State dictionary
        """
        pass
