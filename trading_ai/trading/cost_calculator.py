"""
Trading cost calculator for the Volume Intelligence Trading System.
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class CostCalculator:
    """Calculator for trading costs and fees."""
    
    def __init__(self, broker: str = 'ig_markets'):
        """Initialize the cost calculator.
        
        Args:
            broker: Broker name ('ig_markets', 'interactive_brokers', etc.)
        """
        self.broker = broker
        
        # Default spread settings for different brokers
        self.spread_settings = {
            'ig_markets': {
                'default': 0.1,  # 0.1% default spread
                'stocks': {
                    'AAPL': 0.05,
                    'MSFT': 0.05,
                    'GOOGL': 0.06,
                    'AMZN': 0.06,
                    'TSLA': 0.08,
                    'META': 0.06,
                    'NVDA': 0.07,
                    'ADBE': 0.08,
                    'ORCL': 0.07,
                    'ASML': 0.09,
                    'BIDU': 0.12,
                    'NIO': 0.15,
                    'PINS': 0.12,
                    'SNAP': 0.12,
                    'COIN': 0.15,
                    'PLTR': 0.12,
                    'UBER': 0.10,
                    'LCID': 0.15,
                    'INTC': 0.08,
                    'XPEV': 0.15
                }
            },
            'interactive_brokers': {
                'default': 0.05,  # 0.05% default spread
                'stocks': {}  # Add specific stock spreads if needed
            }
        }
        
        # Default financing rates for different brokers (annual rates)
        self.financing_rates = {
            'ig_markets': {
                'long': 0.0325,  # 3.25% annual rate for long positions
                'short': 0.0425  # 4.25% annual rate for short positions
            },
            'interactive_brokers': {
                'long': 0.0275,  # 2.75% annual rate for long positions
                'short': 0.0375  # 3.75% annual rate for short positions
            }
        }
    
    def calculate_spread_cost(self, ticker: str, price: float, position_size: float) -> float:
        """Calculate the spread cost for a trade.
        
        Args:
            ticker: Stock symbol
            price: Current price
            position_size: Position size (number of shares or contracts)
            
        Returns:
            Spread cost in currency units
        """
        try:
            # Get spread percentage for the ticker
            if ticker in self.spread_settings.get(self.broker, {}).get('stocks', {}):
                spread_pct = self.spread_settings[self.broker]['stocks'][ticker]
            else:
                spread_pct = self.spread_settings.get(self.broker, {}).get('default', 0.1)
            
            # Calculate spread cost
            spread_cost = price * position_size * (spread_pct / 100)
            
            return spread_cost
            
        except Exception as e:
            logger.error(f"Error calculating spread cost: {str(e)}")
            return 0.0
    
    def calculate_financing_cost(self, 
                               ticker: str, 
                               price: float, 
                               position_size: float, 
                               direction: str,
                               days_held: int) -> float:
        """Calculate the financing cost for holding a position.
        
        Args:
            ticker: Stock symbol
            price: Current price
            position_size: Position size (number of shares or contracts)
            direction: Trade direction ('buy' or 'short')
            days_held: Number of days the position is held
            
        Returns:
            Financing cost in currency units
        """
        try:
            # Get annual financing rate based on direction
            if direction.lower() == 'buy':
                annual_rate = self.financing_rates.get(self.broker, {}).get('long', 0.0325)
            else:  # short
                annual_rate = self.financing_rates.get(self.broker, {}).get('short', 0.0425)
            
            # Calculate daily rate
            daily_rate = annual_rate / 365
            
            # Calculate financing cost
            position_value = price * position_size
            financing_cost = position_value * daily_rate * days_held
            
            return financing_cost
            
        except Exception as e:
            logger.error(f"Error calculating financing cost: {str(e)}")
            return 0.0
    
    def calculate_total_cost(self, 
                           ticker: str, 
                           entry_price: float, 
                           exit_price: float, 
                           position_size: float, 
                           direction: str,
                           days_held: int) -> Dict[str, float]:
        """Calculate the total cost for a trade.
        
        Args:
            ticker: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size (number of shares or contracts)
            direction: Trade direction ('buy' or 'short')
            days_held: Number of days the position is held
            
        Returns:
            Dictionary with cost breakdown
        """
        try:
            # Calculate spread costs
            entry_spread = self.calculate_spread_cost(ticker, entry_price, position_size)
            exit_spread = self.calculate_spread_cost(ticker, exit_price, position_size)
            
            # Calculate financing cost
            financing = self.calculate_financing_cost(ticker, entry_price, position_size, direction, days_held)
            
            # Calculate total cost
            total_cost = entry_spread + exit_spread + financing
            
            # Return cost breakdown
            return {
                'entry_spread': entry_spread,
                'exit_spread': exit_spread,
                'financing': financing,
                'total_cost': total_cost
            }
            
        except Exception as e:
            logger.error(f"Error calculating total cost: {str(e)}")
            return {
                'entry_spread': 0.0,
                'exit_spread': 0.0,
                'financing': 0.0,
                'total_cost': 0.0
            }
    
    def calculate_adjusted_pnl(self,
                             ticker: str,
                             entry_price: float,
                             exit_price: float,
                             position_size: float,
                             direction: str,
                             days_held: int) -> Dict[str, float]:
        """Calculate the adjusted P&L for a trade, accounting for costs.
        
        Args:
            ticker: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size (number of shares or contracts)
            direction: Trade direction ('buy' or 'short')
            days_held: Number of days the position is held
            
        Returns:
            Dictionary with P&L breakdown
        """
        try:
            # Calculate raw P&L
            if direction.lower() == 'buy':
                raw_pnl = (exit_price - entry_price) * position_size
            else:  # short
                raw_pnl = (entry_price - exit_price) * position_size
            
            # Calculate costs
            costs = self.calculate_total_cost(ticker, entry_price, exit_price, position_size, direction, days_held)
            
            # Calculate adjusted P&L
            adjusted_pnl = raw_pnl - costs['total_cost']
            
            # Calculate P&L percentage
            position_value = entry_price * position_size
            raw_pnl_pct = (raw_pnl / position_value) * 100
            adjusted_pnl_pct = (adjusted_pnl / position_value) * 100
            
            # Return P&L breakdown
            return {
                'raw_pnl': raw_pnl,
                'raw_pnl_pct': raw_pnl_pct,
                'total_cost': costs['total_cost'],
                'entry_spread': costs['entry_spread'],
                'exit_spread': costs['exit_spread'],
                'financing': costs['financing'],
                'adjusted_pnl': adjusted_pnl,
                'adjusted_pnl_pct': adjusted_pnl_pct
            }
            
        except Exception as e:
            logger.error(f"Error calculating adjusted P&L: {str(e)}")
            return {
                'raw_pnl': 0.0,
                'raw_pnl_pct': 0.0,
                'total_cost': 0.0,
                'entry_spread': 0.0,
                'exit_spread': 0.0,
                'financing': 0.0,
                'adjusted_pnl': 0.0,
                'adjusted_pnl_pct': 0.0
            }
