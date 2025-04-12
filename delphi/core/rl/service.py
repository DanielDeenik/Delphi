"""
Reinforcement learning service module for Delphi.

This module provides a service for reinforcement learning-based trading.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import torch
import functools

from delphi.core.base.service import Service
from delphi.core.base.storage import StorageService
from delphi.core.rl.agent import RLAgent
from delphi.core.rl.dqn_agent import DQNAgent
from delphi.core.rl.ppo_agent import PPOAgent
from delphi.core.rl.trainer import RLTrainer
from delphi.core.performance.paper_trader import PaperTrader

# Configure logger
logger = logging.getLogger(__name__)

class RLService(Service):
    """Service for reinforcement learning-based trading."""
    
    def __init__(self, storage_service: Optional[StorageService] = None,
                paper_trader: Optional[PaperTrader] = None,
                model_dir: str = 'models/rl',
                cache_size: int = 128, **kwargs):
        """Initialize the reinforcement learning service.
        
        Args:
            storage_service: Storage service for storing and retrieving data
            paper_trader: Paper trader for executing trades
            model_dir: Directory for storing models
            cache_size: Size of the LRU cache for service methods
            **kwargs: Additional arguments
        """
        super().__init__(cache_size=cache_size, **kwargs)
        
        self.storage_service = storage_service
        self.paper_trader = paper_trader
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize agents
        self.agents = {}
        
        logger.info(f"Initialized reinforcement learning service")
    
    def _apply_caching(self):
        """Apply LRU caching to service methods."""
        # Apply caching to get_agent
        self._get_agent_impl = self.get_agent
        self.get_agent = functools.lru_cache(maxsize=self.cache_size)(self._get_agent_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for service methods."""
        self.get_agent.cache_clear()
        logger.debug("Cleared cache for reinforcement learning service")
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the reinforcement learning service.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # Nothing to initialize
        return True
    
    def create_agent(self, agent_type: str, ticker: str, **kwargs) -> RLAgent:
        """Create a reinforcement learning agent.
        
        Args:
            agent_type: Type of agent to create ('dqn' or 'ppo')
            ticker: Stock symbol
            **kwargs: Additional arguments for the agent
            
        Returns:
            Reinforcement learning agent
        """
        try:
            # Create agent
            if agent_type.lower() == 'dqn':
                agent = DQNAgent(**kwargs)
            elif agent_type.lower() == 'ppo':
                agent = PPOAgent(**kwargs)
            else:
                logger.error(f"Unknown agent type: {agent_type}")
                return None
            
            # Store agent
            self.agents[ticker] = agent
            
            logger.info(f"Created {agent_type} agent for {ticker}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            return None
    
    def get_agent(self, ticker: str) -> Optional[RLAgent]:
        """Get a reinforcement learning agent.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Reinforcement learning agent or None if not found
        """
        # Check if agent exists
        if ticker in self.agents:
            return self.agents[ticker]
        
        # Try to load agent from disk
        model_path = os.path.join(self.model_dir, f"{ticker.lower()}_agent.pt")
        
        if os.path.exists(model_path):
            try:
                # Try to load as DQN agent first
                agent = DQNAgent.load(model_path)
                
                if agent is not None:
                    self.agents[ticker] = agent
                    return agent
                
                # Try to load as PPO agent
                agent = PPOAgent.load(model_path)
                
                if agent is not None:
                    self.agents[ticker] = agent
                    return agent
                
            except Exception as e:
                logger.error(f"Error loading agent for {ticker}: {str(e)}")
        
        logger.warning(f"Agent for {ticker} not found")
        return None
    
    def train_agent(self, ticker: str, data: pd.DataFrame, agent_type: str = 'dqn',
                   save_model: bool = True, **kwargs) -> Dict[str, Any]:
        """Train a reinforcement learning agent.
        
        Args:
            ticker: Stock symbol
            data: Training data
            agent_type: Type of agent to train ('dqn' or 'ppo')
            save_model: Whether to save the trained model
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary with training results
        """
        try:
            # Get or create agent
            agent = self.get_agent(ticker)
            
            if agent is None:
                # Create agent
                agent = self.create_agent(agent_type, ticker, **kwargs)
                
                if agent is None:
                    return {"error": f"Failed to create agent for {ticker}"}
            
            # Create trainer
            trainer = RLTrainer(agent)
            
            # Train agent
            results = trainer.train(data, **kwargs)
            
            # Save model if requested
            if save_model:
                model_path = os.path.join(self.model_dir, f"{ticker.lower()}_agent.pt")
                agent.save(model_path)
            
            # Clear cache
            self.clear_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"Error training agent: {str(e)}")
            return {"error": str(e)}
    
    def evaluate_agent(self, ticker: str, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Evaluate a reinforcement learning agent.
        
        Args:
            ticker: Stock symbol
            data: Evaluation data
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get agent
            agent = self.get_agent(ticker)
            
            if agent is None:
                return {"error": f"Agent for {ticker} not found"}
            
            # Create trainer
            trainer = RLTrainer(agent)
            
            # Evaluate agent
            results = trainer.evaluate(data, **kwargs)
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating agent: {str(e)}")
            return {"error": str(e)}
    
    def generate_trading_signals(self, ticker: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate trading signals using a reinforcement learning agent.
        
        Args:
            ticker: Stock symbol
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trading signals
        """
        try:
            # Get agent
            agent = self.get_agent(ticker)
            
            if agent is None:
                logger.warning(f"Agent for {ticker} not found")
                return pd.DataFrame()
            
            # Create trainer
            trainer = RLTrainer(agent)
            
            # Generate signals
            signals = trainer.generate_signals(data, **kwargs)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            return pd.DataFrame()
    
    def execute_signals(self, ticker: str, signals: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Execute trading signals using the paper trader.
        
        Args:
            ticker: Stock symbol
            signals: DataFrame with trading signals
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Check if paper trader is available
            if self.paper_trader is None:
                logger.warning("Paper trader not available")
                return {"error": "Paper trader not available"}
            
            # Execute signals
            results = {
                'trades': [],
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0
            }
            
            # Process each signal
            for _, signal in signals.iterrows():
                # Skip if no action
                if signal['action'] == 0:  # Hold
                    continue
                
                # Determine direction
                direction = 'long' if signal['action'] == 1 else 'short'
                
                # Execute trade
                trade = self.paper_trader.execute_trade(
                    ticker=ticker,
                    direction=direction,
                    entry_price=signal['price'],
                    position_size=None,  # Use default position sizing
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    trigger_reason=f"RL signal ({signal.get('confidence', 0):.2f})",
                    model_confidence=signal.get('confidence')
                )
                
                if trade is not None:
                    results['trades'].append(trade.to_dict())
                    results['successful_trades'] += 1
                else:
                    results['failed_trades'] += 1
                
                results['total_trades'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
            return {"error": str(e)}
