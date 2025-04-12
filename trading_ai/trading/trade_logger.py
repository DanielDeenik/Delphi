"""
Trade logger for the Volume Intelligence Trading System.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import uuid

from trading_ai.config import config_manager
from trading_ai.core.bigquery_io import BigQueryStorage

# Configure logging
logger = logging.getLogger(__name__)

class TradeLogger:
    """Logger for trades and performance tracking."""
    
    def __init__(self):
        """Initialize the trade logger."""
        self.bigquery_storage = BigQueryStorage()
    
    def log_trade(self, 
                 ticker: str, 
                 direction: str, 
                 entry_price: float, 
                 position_size: float,
                 stop_loss: float,
                 take_profit: float,
                 entry_date: Optional[datetime] = None,
                 trigger_reason: Optional[str] = None,
                 notes: Optional[str] = None) -> str:
        """Log a new trade.
        
        Args:
            ticker: Stock symbol
            direction: Trade direction ('buy' or 'short')
            entry_price: Entry price
            position_size: Position size (number of shares or contracts)
            stop_loss: Stop loss price
            take_profit: Take profit price
            entry_date: Entry date (defaults to current date)
            trigger_reason: Reason for entering the trade
            notes: Additional notes
            
        Returns:
            Trade ID if successful, empty string otherwise
        """
        try:
            # Generate trade ID
            trade_id = str(uuid.uuid4())
            
            # Set entry date if not provided
            if entry_date is None:
                entry_date = datetime.now()
            
            # Create trade data
            trade_data = {
                'trade_id': trade_id,
                'symbol': ticker,
                'direction': direction,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'exit_date': None,
                'exit_price': None,
                'pnl_amount': None,
                'pnl_percent': None,
                'trade_duration_days': None,
                'status': 'open',
                'trigger_reason': trigger_reason,
                'exit_reason': None,
                'notes': notes,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Log trade to BigQuery
            success = self.bigquery_storage.log_trade(trade_data)
            
            if success:
                logger.info(f"Successfully logged trade {trade_id} for {ticker}")
                return trade_id
            else:
                logger.warning(f"Failed to log trade for {ticker}")
                return ""
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            return ""
    
    def update_trade(self,
                    trade_id: str,
                    exit_price: Optional[float] = None,
                    exit_date: Optional[datetime] = None,
                    exit_reason: Optional[str] = None,
                    status: Optional[str] = None,
                    notes: Optional[str] = None) -> bool:
        """Update an existing trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_date: Exit date
            exit_reason: Reason for exiting the trade
            status: Trade status ('open', 'closed', 'cancelled')
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing trade
            trades_df = self.bigquery_storage.get_trade_logs()
            trade_row = trades_df[trades_df['trade_id'] == trade_id]
            
            if trade_row.empty:
                logger.warning(f"Trade {trade_id} not found")
                return False
            
            # Extract existing trade data
            trade_data = trade_row.iloc[0].to_dict()
            
            # Update fields
            if exit_price is not None:
                trade_data['exit_price'] = exit_price
            
            if exit_date is not None:
                trade_data['exit_date'] = exit_date
            elif exit_price is not None and trade_data['exit_date'] is None:
                trade_data['exit_date'] = datetime.now()
            
            if exit_reason is not None:
                trade_data['exit_reason'] = exit_reason
            
            if status is not None:
                trade_data['status'] = status
            elif exit_price is not None and trade_data['status'] == 'open':
                trade_data['status'] = 'closed'
            
            if notes is not None:
                trade_data['notes'] = notes
            
            # Calculate P&L if trade is closed
            if trade_data['status'] == 'closed' and trade_data['exit_price'] is not None:
                # Calculate P&L amount
                if trade_data['direction'] == 'buy':
                    pnl_amount = (trade_data['exit_price'] - trade_data['entry_price']) * trade_data['position_size']
                else:  # short
                    pnl_amount = (trade_data['entry_price'] - trade_data['exit_price']) * trade_data['position_size']
                
                trade_data['pnl_amount'] = pnl_amount
                
                # Calculate P&L percentage
                pnl_percent = (pnl_amount / (trade_data['entry_price'] * trade_data['position_size'])) * 100
                trade_data['pnl_percent'] = pnl_percent
                
                # Calculate trade duration
                if trade_data['entry_date'] is not None and trade_data['exit_date'] is not None:
                    entry_date = pd.to_datetime(trade_data['entry_date'])
                    exit_date = pd.to_datetime(trade_data['exit_date'])
                    trade_data['trade_duration_days'] = (exit_date - entry_date).days
            
            # Update timestamp
            trade_data['updated_at'] = datetime.now()
            
            # Log updated trade to BigQuery
            success = self.bigquery_storage.log_trade(trade_data)
            
            if success:
                logger.info(f"Successfully updated trade {trade_id}")
                return True
            else:
                logger.warning(f"Failed to update trade {trade_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error updating trade: {str(e)}")
            return False
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades.
        
        Returns:
            DataFrame with open trades
        """
        try:
            # Get trades with status 'open'
            trades_df = self.bigquery_storage.get_trade_logs(status='open')
            
            logger.info(f"Retrieved {len(trades_df)} open trades")
            return trades_df
            
        except Exception as e:
            logger.error(f"Error getting open trades: {str(e)}")
            return pd.DataFrame()
    
    def get_trade_history(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Get trade history.
        
        Args:
            ticker: Filter by stock symbol (None for all symbols)
            
        Returns:
            DataFrame with trade history
        """
        try:
            # Get all trades
            trades_df = self.bigquery_storage.get_trade_logs(symbol=ticker)
            
            logger.info(f"Retrieved {len(trades_df)} trades")
            return trades_df
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get all closed trades
            trades_df = self.bigquery_storage.get_trade_logs(status='closed')
            
            if trades_df.empty:
                logger.warning("No closed trades found")
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_pnl': 0,
                    'avg_trade_duration': 0
                }
            
            # Calculate metrics
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['pnl_amount'] > 0]
            losing_trades = trades_df[trades_df['pnl_amount'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = win_count / total_trades if total_trades > 0 else 0
            
            avg_profit = winning_trades['pnl_amount'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl_amount'].mean() if not losing_trades.empty else 0
            
            total_profit = winning_trades['pnl_amount'].sum() if not winning_trades.empty else 0
            total_loss = abs(losing_trades['pnl_amount'].sum()) if not losing_trades.empty else 0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            total_pnl = trades_df['pnl_amount'].sum()
            
            avg_trade_duration = trades_df['trade_duration_days'].mean()
            
            # Return metrics
            metrics = {
                'total_trades': total_trades,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': total_pnl,
                'avg_trade_duration': avg_trade_duration
            }
            
            logger.info(f"Calculated performance metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
