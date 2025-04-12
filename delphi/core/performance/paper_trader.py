"""
Paper trading module for Delphi.

This module provides a paper trading system for simulating trades.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from delphi.core.base.service import Service
from delphi.core.performance.trade import Trade, TradeStatus, TradeDirection
from delphi.core.performance.trade_repository import TradeRepository
from delphi.core.performance.performance_metrics import PerformanceMetrics
from delphi.core.performance.risk_analyzer import RiskAnalyzer

# Configure logger
logger = logging.getLogger(__name__)

class PaperTrader(Service):
    """Paper trading system for simulating trades."""
    
    def __init__(self, trade_repository: TradeRepository, 
                performance_metrics: Optional[PerformanceMetrics] = None,
                risk_analyzer: Optional[RiskAnalyzer] = None,
                initial_capital: float = 100000.0, 
                risk_per_trade: float = 0.02,
                **kwargs):
        """Initialize the paper trader.
        
        Args:
            trade_repository: Repository for storing and retrieving trades
            performance_metrics: Service for calculating performance metrics
            risk_analyzer: Service for analyzing risk
            initial_capital: Initial capital for paper trading
            risk_per_trade: Default risk per trade as a percentage of capital (0.02 = 2%)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.trade_repository = trade_repository
        self.performance_metrics = performance_metrics or PerformanceMetrics()
        self.risk_analyzer = risk_analyzer or RiskAnalyzer()
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        logger.info(f"Initialized paper trader with ${initial_capital:.2f} capital and {risk_per_trade*100:.1f}% risk per trade")
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the paper trader.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # Nothing to initialize
        return True
    
    def execute_trade(self,
                     ticker: str,
                     direction: str,
                     entry_price: float,
                     position_size: Optional[float] = None,
                     capital_percentage: Optional[float] = None,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     trigger_reason: Optional[str] = None,
                     notes: Optional[str] = None,
                     model_confidence: Optional[float] = None) -> Optional[Trade]:
        """Execute a paper trade.
        
        Args:
            ticker: Stock symbol
            direction: Trade direction ('long' or 'short')
            entry_price: Entry price
            position_size: Position size (number of shares or contracts)
            capital_percentage: Percentage of capital to allocate (alternative to position_size)
            stop_loss: Stop loss price
            take_profit: Take profit price
            trigger_reason: Reason for entering the trade
            notes: Additional notes
            model_confidence: Confidence score from the model (0-1)
            
        Returns:
            Trade if successful, None otherwise
        """
        try:
            # Validate direction
            if direction not in [TradeDirection.LONG, TradeDirection.SHORT]:
                logger.error(f"Invalid direction: {direction}. Must be '{TradeDirection.LONG}' or '{TradeDirection.SHORT}'")
                return None
            
            # Calculate position size if not provided
            if position_size is None:
                if capital_percentage is not None:
                    # Use specified capital percentage
                    position_value = self.current_capital * capital_percentage
                else:
                    # Use default risk per trade
                    position_value = self.current_capital * self.risk_per_trade
                
                # Calculate number of shares
                position_size = position_value / entry_price
            
            # Calculate position value
            position_value = position_size * entry_price
            
            # Validate position value
            if position_value > self.current_capital:
                logger.warning(f"Position value (${position_value:.2f}) exceeds current capital (${self.current_capital:.2f})")
                position_size = self.current_capital / entry_price
                position_value = position_size * entry_price
                logger.info(f"Adjusted position size to {position_size:.2f} shares (${position_value:.2f})")
            
            # Create trade
            trade = Trade(
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trigger_reason=trigger_reason,
                notes=notes,
                model_confidence=model_confidence
            )
            
            # Store trade
            success = self.trade_repository.store_trade(trade)
            
            if success:
                logger.info(f"Successfully executed {direction} trade for {ticker} with ID {trade.trade_id}")
                logger.info(f"Position: {position_size:.2f} shares at ${entry_price:.2f} (${position_value:.2f})")
                logger.info(f"Risk-Reward Ratio: {trade.risk_reward_ratio:.2f}")
                
                # Update capital (reserve the position value)
                self.current_capital -= position_value
                
                logger.info(f"Available capital: ${self.current_capital:.2f}")
                
                return trade
            else:
                logger.warning(f"Failed to execute trade for {ticker}")
                return None
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None
    
    def close_trade(self,
                   trade_id: str,
                   exit_price: float,
                   exit_reason: Optional[str] = None,
                   notes: Optional[str] = None) -> bool:
        """Close a paper trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exiting the trade
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get trade
            trade = self.trade_repository.get_trade(trade_id)
            
            if trade is None:
                logger.warning(f"Trade {trade_id} not found")
                return False
            
            # Check if trade is already closed
            if trade.status == TradeStatus.CLOSED:
                logger.warning(f"Trade {trade_id} is already closed")
                return False
            
            # Close trade
            success = trade.close(
                exit_price=exit_price,
                exit_reason=exit_reason,
                notes=notes
            )
            
            if not success:
                logger.warning(f"Failed to close trade {trade_id}")
                return False
            
            # Store updated trade
            success = self.trade_repository.store_trade(trade)
            
            if success:
                logger.info(f"Successfully closed trade {trade_id}")
                logger.info(f"P&L: ${trade.pnl_amount:.2f} ({trade.pnl_percentage:.2f}%)")
                logger.info(f"Duration: {trade.trade_duration_days:.2f} days")
                
                # Update capital (return the position value + P&L)
                position_value = trade.entry_price * trade.position_size
                self.current_capital += position_value + trade.pnl_amount
                
                logger.info(f"Current capital: ${self.current_capital:.2f}")
                
                return True
            else:
                logger.warning(f"Failed to store updated trade {trade_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return False
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Trade]:
        """Check open trades for stop loss or take profit triggers.
        
        Args:
            current_prices: Dictionary mapping tickers to current prices
            
        Returns:
            List of trades that were closed
        """
        try:
            # Get open trades
            open_trades_df = self.trade_repository.get_open_trades()
            
            if open_trades_df.empty:
                return []
            
            closed_trades = []
            
            # Check each open trade
            for _, trade_data in open_trades_df.iterrows():
                # Create Trade object
                trade = Trade.from_dict(trade_data)
                
                ticker = trade.ticker
                
                # Skip if we don't have current price for this ticker
                if ticker not in current_prices:
                    continue
                
                current_price = current_prices[ticker]
                
                # Check stop loss
                if trade.direction == TradeDirection.LONG and current_price <= trade.stop_loss:
                    # Close trade at stop loss
                    success = self.close_trade(
                        trade_id=trade.trade_id,
                        exit_price=trade.stop_loss,
                        exit_reason='Stop loss hit',
                        notes=f"Closed at stop loss. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append(trade)
                
                elif trade.direction == TradeDirection.SHORT and current_price >= trade.stop_loss:
                    # Close trade at stop loss
                    success = self.close_trade(
                        trade_id=trade.trade_id,
                        exit_price=trade.stop_loss,
                        exit_reason='Stop loss hit',
                        notes=f"Closed at stop loss. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append(trade)
                
                # Check take profit
                elif trade.direction == TradeDirection.LONG and current_price >= trade.take_profit:
                    # Close trade at take profit
                    success = self.close_trade(
                        trade_id=trade.trade_id,
                        exit_price=trade.take_profit,
                        exit_reason='Take profit hit',
                        notes=f"Closed at take profit. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append(trade)
                
                elif trade.direction == TradeDirection.SHORT and current_price <= trade.take_profit:
                    # Close trade at take profit
                    success = self.close_trade(
                        trade_id=trade.trade_id,
                        exit_price=trade.take_profit,
                        exit_reason='Take profit hit',
                        notes=f"Closed at take profit. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append(trade)
            
            return closed_trades
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return []
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get a summary of the paper trading account.
        
        Returns:
            Dictionary with account summary
        """
        try:
            # Get open trades
            open_trades_df = self.trade_repository.get_open_trades()
            
            # Get trade history
            trade_history_df = self.trade_repository.get_trade_history(status=TradeStatus.CLOSED)
            
            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            
            # Calculate performance metrics
            performance_metrics = self.performance_metrics.calculate_metrics(trade_history_df)
            
            # Calculate account value
            account_value = self.current_capital + unrealized_pnl
            
            # Calculate return metrics
            total_return = account_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Calculate risk metrics
            risk_metrics = self.risk_analyzer.calculate_risk_metrics(trade_history_df)
            
            # Prepare summary
            summary = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'unrealized_pnl': unrealized_pnl,
                'account_value': account_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'open_trades_count': len(open_trades_df),
                'closed_trades_count': len(trade_history_df),
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_metrics
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {}
