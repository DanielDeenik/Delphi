"""
Trading environment module for Delphi.

This module provides a trading environment for reinforcement learning.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import gym
from gym import spaces

# Configure logger
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, window_size: int = 20, 
                initial_balance: float = 10000.0, commission: float = 0.001,
                reward_scaling: float = 0.01, **kwargs):
        """Initialize the trading environment.
        
        Args:
            data: DataFrame with price data
            window_size: Size of the observation window
            initial_balance: Initial account balance
            commission: Trading commission as a percentage
            reward_scaling: Scaling factor for rewards
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Store parameters
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        
        # Validate data
        self._validate_data()
        
        # Set up environment
        self._setup_environment()
        
        logger.info(f"Initialized trading environment with {len(self.data)} data points")
    
    def _validate_data(self):
        """Validate the input data."""
        # Check if data is empty
        if self.data.empty:
            raise ValueError("Data is empty")
        
        # Check for required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort data by date
        self.data = self.data.sort_values('date')
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
    
    def _setup_environment(self):
        """Set up the environment."""
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space: price data + account info
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, 6),  # OHLCV + position
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment.
        
        Returns:
            Initial observation
        """
        # Reset position
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.data.iloc[self.current_step]['close']
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_commission_paid = 0
        self.total_trades = 0
        
        # Get initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """Get the current observation.
        
        Returns:
            Current observation
        """
        # Get window of price data
        price_data = self.data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Extract OHLCV data
        obs = np.array([
            price_data['open'].values,
            price_data['high'].values,
            price_data['low'].values,
            price_data['close'].values,
            price_data['volume'].values,
            np.array([self.shares_held] * self.window_size)
        ]).T
        
        # Normalize data
        obs[:, 0] = obs[:, 0] / obs[-1, 0] - 1  # open
        obs[:, 1] = obs[:, 1] / obs[-1, 0] - 1  # high
        obs[:, 2] = obs[:, 2] / obs[-1, 0] - 1  # low
        obs[:, 3] = obs[:, 3] / obs[-1, 0] - 1  # close
        obs[:, 4] = obs[:, 4] / obs[:, 4].mean() - 1  # volume
        obs[:, 5] = obs[:, 5] / 100  # position
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current price
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # Initialize reward
        reward = 0
        
        # Execute action
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            if self.balance > 0:
                # Calculate maximum shares that can be bought
                max_shares = self.balance / (self.current_price * (1 + self.commission))
                
                # Buy all possible shares
                shares_bought = max_shares
                cost = shares_bought * self.current_price
                commission = cost * self.commission
                
                # Update state
                self.balance -= (cost + commission)
                self.shares_held += shares_bought
                self.cost_basis = self.current_price
                
                # Update stats
                self.total_shares_bought += shares_bought
                self.total_commission_paid += commission
                self.total_trades += 1
                
                # Calculate reward (negative commission)
                reward = -commission * self.reward_scaling
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Sell all shares
                shares_sold = self.shares_held
                revenue = shares_sold * self.current_price
                commission = revenue * self.commission
                
                # Update state
                self.balance += (revenue - commission)
                
                # Calculate profit
                profit = revenue - (self.cost_basis * shares_sold) - commission
                
                # Update stats
                self.total_shares_sold += shares_sold
                self.total_commission_paid += commission
                self.total_trades += 1
                
                # Calculate reward (profit)
                reward = profit * self.reward_scaling
                
                # Reset position
                self.shares_held = 0
                self.cost_basis = 0
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.shares_held * self.current_price)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get observation
        obs = self._get_observation()
        
        # Prepare info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'current_price': self.current_price,
            'portfolio_value': portfolio_value,
            'total_trades': self.total_trades,
            'total_commission_paid': self.total_commission_paid
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            # Calculate portfolio value
            portfolio_value = self.balance + (self.shares_held * self.current_price)
            
            # Print status
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Current price: ${self.current_price:.2f}")
            print(f"Portfolio value: ${portfolio_value:.2f}")
            print(f"Total trades: {self.total_trades}")
            print(f"Total commission paid: ${self.total_commission_paid:.2f}")
            print("-" * 50)
    
    def close(self):
        """Close the environment."""
        pass
