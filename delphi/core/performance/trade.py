"""
Trade module for Delphi.

This module provides classes for representing trades.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
import logging

# Configure logger
logger = logging.getLogger(__name__)

class TradeStatus:
    """Trade status constants."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

class TradeDirection:
    """Trade direction constants."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Trade:
    """Class for representing a trade."""
    
    # Required fields
    ticker: str
    direction: str
    entry_price: float
    position_size: float
    
    # Optional fields with defaults
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_date: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    pnl_amount: Optional[float] = None
    pnl_percentage: Optional[float] = None
    trade_duration_days: Optional[float] = None
    status: str = TradeStatus.OPEN
    trigger_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    notes: Optional[str] = None
    risk_percentage: Optional[float] = None
    reward_percentage: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    model_confidence: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and initialize the trade."""
        # Validate direction
        if self.direction not in [TradeDirection.LONG, TradeDirection.SHORT]:
            logger.warning(f"Invalid direction: {self.direction}. Using {TradeDirection.LONG} instead.")
            self.direction = TradeDirection.LONG
        
        # Calculate default stop loss if not provided
        if self.stop_loss is None:
            if self.direction == TradeDirection.LONG:
                self.stop_loss = self.entry_price * 0.99
            else:  # short
                self.stop_loss = self.entry_price * 1.01
        
        # Calculate default take profit if not provided
        if self.take_profit is None:
            if self.direction == TradeDirection.LONG:
                self.take_profit = self.entry_price * 1.02
            else:  # short
                self.take_profit = self.entry_price * 0.98
        
        # Calculate risk metrics
        self._calculate_risk_metrics()
    
    def _calculate_risk_metrics(self):
        """Calculate risk metrics for the trade."""
        # Calculate risk per share
        risk_per_share = abs(self.entry_price - self.stop_loss)
        
        # Calculate total risk
        total_risk = risk_per_share * self.position_size
        
        # Calculate reward per share
        reward_per_share = abs(self.take_profit - self.entry_price)
        
        # Calculate total reward
        total_reward = reward_per_share * self.position_size
        
        # Calculate risk-reward ratio
        self.risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    
    def close(self, exit_price: float, exit_date: Optional[datetime] = None, 
             exit_reason: Optional[str] = None, notes: Optional[str] = None) -> bool:
        """Close the trade.
        
        Args:
            exit_price: Exit price
            exit_date: Exit date (defaults to current date)
            exit_reason: Reason for exiting the trade
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if trade is already closed
            if self.status == TradeStatus.CLOSED:
                logger.warning(f"Trade {self.trade_id} is already closed")
                return False
            
            # Set exit date if not provided
            if exit_date is None:
                exit_date = datetime.now()
            
            # Set exit price and date
            self.exit_price = exit_price
            self.exit_date = exit_date
            
            # Set exit reason and notes
            if exit_reason:
                self.exit_reason = exit_reason
            
            if notes:
                self.notes = notes
            
            # Calculate P&L
            if self.direction == TradeDirection.LONG:
                self.pnl_amount = (self.exit_price - self.entry_price) * self.position_size
            else:  # short
                self.pnl_amount = (self.entry_price - self.exit_price) * self.position_size
            
            # Calculate P&L percentage
            position_value = self.entry_price * self.position_size
            self.pnl_percentage = (self.pnl_amount / position_value) * 100
            
            # Calculate trade duration
            self.trade_duration_days = (self.exit_date - self.entry_date).total_seconds() / 86400
            
            # Update status and timestamp
            self.status = TradeStatus.CLOSED
            self.updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trade to a dictionary.
        
        Returns:
            Dictionary representation of the trade
        """
        return {
            'trade_id': self.trade_id,
            'ticker': self.ticker,
            'direction': self.direction,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'pnl_amount': self.pnl_amount,
            'pnl_percentage': self.pnl_percentage,
            'trade_duration_days': self.trade_duration_days,
            'status': self.status,
            'trigger_reason': self.trigger_reason,
            'exit_reason': self.exit_reason,
            'notes': self.notes,
            'risk_percentage': self.risk_percentage,
            'reward_percentage': self.reward_percentage,
            'risk_reward_ratio': self.risk_reward_ratio,
            'model_confidence': self.model_confidence,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create a trade from a dictionary.
        
        Args:
            data: Dictionary with trade data
            
        Returns:
            Trade instance
        """
        return cls(**data)
