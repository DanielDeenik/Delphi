"""
Paper trading engine for the Volume Intelligence Trading System.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import uuid

from trading_ai.config import config_manager
from trading_ai.core.bigquery_io import BigQueryStorage
from trading_ai.trading.trade_logger import TradeLogger
from trading_ai.trading.cost_calculator import CostCalculator

# Configure logging
logger = logging.getLogger(__name__)

class PaperTrader:
    """Paper trading engine for simulating trades."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize the paper trader.
        
        Args:
            initial_capital: Initial capital for paper trading
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trade_logger = TradeLogger()
        self.cost_calculator = CostCalculator()
        self.bigquery_storage = BigQueryStorage()
    
    def execute_trade(self,
                     ticker: str,
                     direction: str,
                     entry_price: float,
                     position_size: Optional[float] = None,
                     capital_percentage: Optional[float] = None,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     trigger_reason: Optional[str] = None,
                     notes: Optional[str] = None) -> str:
        """Execute a paper trade.
        
        Args:
            ticker: Stock symbol
            direction: Trade direction ('buy' or 'short')
            entry_price: Entry price
            position_size: Position size (number of shares or contracts)
            capital_percentage: Percentage of capital to allocate (alternative to position_size)
            stop_loss: Stop loss price
            take_profit: Take profit price
            trigger_reason: Reason for entering the trade
            notes: Additional notes
            
        Returns:
            Trade ID if successful, empty string otherwise
        """
        try:
            # Validate direction
            if direction.lower() not in ['buy', 'short']:
                logger.error(f"Invalid direction: {direction}. Must be 'buy' or 'short'.")
                return ""
            
            # Standardize direction
            direction = direction.lower()
            
            # Calculate position size if not provided
            if position_size is None:
                if capital_percentage is None:
                    # Default to 5% of capital
                    capital_percentage = 5.0
                
                # Calculate position size based on capital percentage
                position_value = self.current_capital * (capital_percentage / 100)
                position_size = position_value / entry_price
            
            # Calculate default stop loss and take profit if not provided
            if stop_loss is None or take_profit is None:
                # Get price data
                price_df = self.bigquery_storage.get_stock_prices(ticker, days=30)
                
                if not price_df.empty:
                    # Calculate ATR (Average True Range)
                    price_df['high_low'] = price_df['high'] - price_df['low']
                    price_df['high_close'] = abs(price_df['high'] - price_df['close'].shift(1))
                    price_df['low_close'] = abs(price_df['low'] - price_df['close'].shift(1))
                    price_df['tr'] = price_df[['high_low', 'high_close', 'low_close']].max(axis=1)
                    atr = price_df['tr'].rolling(window=14).mean().iloc[-1]
                    
                    # Set stop loss and take profit based on ATR
                    if stop_loss is None:
                        if direction == 'buy':
                            stop_loss = entry_price - (2 * atr)
                        else:  # short
                            stop_loss = entry_price + (2 * atr)
                    
                    if take_profit is None:
                        if direction == 'buy':
                            take_profit = entry_price + (3 * atr)
                        else:  # short
                            take_profit = entry_price - (3 * atr)
                else:
                    # Fallback to percentage-based levels
                    if stop_loss is None:
                        if direction == 'buy':
                            stop_loss = entry_price * 0.95  # 5% below entry
                        else:  # short
                            stop_loss = entry_price * 1.05  # 5% above entry
                    
                    if take_profit is None:
                        if direction == 'buy':
                            take_profit = entry_price * 1.10  # 10% above entry
                        else:  # short
                            take_profit = entry_price * 0.90  # 10% below entry
            
            # Log the trade
            trade_id = self.trade_logger.log_trade(
                ticker=ticker,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_date=datetime.now(),
                trigger_reason=trigger_reason,
                notes=notes
            )
            
            if trade_id:
                logger.info(f"Successfully executed {direction} trade for {ticker} with ID {trade_id}")
                
                # Update capital (account for spread cost)
                spread_cost = self.cost_calculator.calculate_spread_cost(ticker, entry_price, position_size)
                self.current_capital -= spread_cost
                
                logger.info(f"Current capital: ${self.current_capital:.2f}")
            else:
                logger.warning(f"Failed to execute trade for {ticker}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return ""
    
    def close_trade(self,
                   trade_id: str,
                   exit_price: float,
                   exit_reason: Optional[str] = None,
                   notes: Optional[str] = None) -> bool:
        """Close an open paper trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exiting the trade
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get trade details
            trades_df = self.trade_logger.get_open_trades()
            trade_row = trades_df[trades_df['trade_id'] == trade_id]
            
            if trade_row.empty:
                logger.warning(f"Trade {trade_id} not found or already closed")
                return False
            
            # Extract trade data
            trade = trade_row.iloc[0]
            
            # Calculate days held
            entry_date = pd.to_datetime(trade['entry_date'])
            exit_date = datetime.now()
            days_held = (exit_date - entry_date).days
            
            # Calculate P&L with costs
            pnl_info = self.cost_calculator.calculate_adjusted_pnl(
                ticker=trade['symbol'],
                entry_price=trade['entry_price'],
                exit_price=exit_price,
                position_size=trade['position_size'],
                direction=trade['direction'],
                days_held=days_held
            )
            
            # Update trade
            success = self.trade_logger.update_trade(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_date=exit_date,
                exit_reason=exit_reason,
                status='closed',
                notes=f"{notes}\n\nP&L: ${pnl_info['adjusted_pnl']:.2f} ({pnl_info['adjusted_pnl_pct']:.2f}%)" if notes else f"P&L: ${pnl_info['adjusted_pnl']:.2f} ({pnl_info['adjusted_pnl_pct']:.2f}%)"
            )
            
            if success:
                logger.info(f"Successfully closed trade {trade_id} with P&L: ${pnl_info['adjusted_pnl']:.2f}")
                
                # Update capital
                self.current_capital += pnl_info['adjusted_pnl']
                
                logger.info(f"Current capital: ${self.current_capital:.2f}")
            else:
                logger.warning(f"Failed to close trade {trade_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            return False
    
    def check_stop_loss_take_profit(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check open trades for stop loss or take profit triggers.
        
        Args:
            current_prices: Dictionary mapping tickers to current prices
            
        Returns:
            List of trades that were closed
        """
        try:
            # Get open trades
            open_trades = self.trade_logger.get_open_trades()
            
            if open_trades.empty:
                return []
            
            closed_trades = []
            
            # Check each open trade
            for _, trade in open_trades.iterrows():
                ticker = trade['symbol']
                
                # Skip if we don't have current price for this ticker
                if ticker not in current_prices:
                    continue
                
                current_price = current_prices[ticker]
                
                # Check stop loss
                if trade['direction'] == 'buy' and current_price <= trade['stop_loss']:
                    # Close trade at stop loss
                    success = self.close_trade(
                        trade_id=trade['trade_id'],
                        exit_price=trade['stop_loss'],
                        exit_reason='Stop loss hit',
                        notes=f"Closed at stop loss. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append({
                            'trade_id': trade['trade_id'],
                            'ticker': ticker,
                            'direction': trade['direction'],
                            'reason': 'Stop loss hit',
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['stop_loss']
                        })
                
                elif trade['direction'] == 'short' and current_price >= trade['stop_loss']:
                    # Close trade at stop loss
                    success = self.close_trade(
                        trade_id=trade['trade_id'],
                        exit_price=trade['stop_loss'],
                        exit_reason='Stop loss hit',
                        notes=f"Closed at stop loss. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append({
                            'trade_id': trade['trade_id'],
                            'ticker': ticker,
                            'direction': trade['direction'],
                            'reason': 'Stop loss hit',
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['stop_loss']
                        })
                
                # Check take profit
                elif trade['direction'] == 'buy' and current_price >= trade['take_profit']:
                    # Close trade at take profit
                    success = self.close_trade(
                        trade_id=trade['trade_id'],
                        exit_price=trade['take_profit'],
                        exit_reason='Take profit hit',
                        notes=f"Closed at take profit. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append({
                            'trade_id': trade['trade_id'],
                            'ticker': ticker,
                            'direction': trade['direction'],
                            'reason': 'Take profit hit',
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['take_profit']
                        })
                
                elif trade['direction'] == 'short' and current_price <= trade['take_profit']:
                    # Close trade at take profit
                    success = self.close_trade(
                        trade_id=trade['trade_id'],
                        exit_price=trade['take_profit'],
                        exit_reason='Take profit hit',
                        notes=f"Closed at take profit. Current price: ${current_price:.2f}"
                    )
                    
                    if success:
                        closed_trades.append({
                            'trade_id': trade['trade_id'],
                            'ticker': ticker,
                            'direction': trade['direction'],
                            'reason': 'Take profit hit',
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['take_profit']
                        })
            
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
            open_trades = self.trade_logger.get_open_trades()
            
            # Get trade history
            trade_history = self.trade_logger.get_trade_history()
            closed_trades = trade_history[trade_history['status'] == 'closed']
            
            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            
            # Calculate performance metrics
            performance_metrics = self.trade_logger.calculate_performance_metrics()
            
            # Calculate account value
            account_value = self.current_capital + unrealized_pnl
            
            # Calculate return metrics
            total_return = account_value - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            
            # Return account summary
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'unrealized_pnl': unrealized_pnl,
                'account_value': account_value,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'open_trades_count': len(open_trades),
                'closed_trades_count': len(closed_trades),
                'win_rate': performance_metrics.get('win_rate', 0),
                'profit_factor': performance_metrics.get('profit_factor', 0),
                'avg_profit': performance_metrics.get('avg_profit', 0),
                'avg_loss': performance_metrics.get('avg_loss', 0),
                'total_pnl': performance_metrics.get('total_pnl', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {}
