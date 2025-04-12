"""
Trade repository module for Delphi.

This module provides a repository for storing and retrieving trades.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging
import functools

from delphi.core.base.repository import Repository
from delphi.core.base.storage import StorageService
from delphi.core.performance.trade import Trade, TradeStatus

# Configure logger
logger = logging.getLogger(__name__)

class TradeRepository(Repository):
    """Repository for storing and retrieving trades."""
    
    def __init__(self, storage_service: StorageService, cache_size: int = 128, **kwargs):
        """Initialize the trade repository.
        
        Args:
            storage_service: Storage service for storing and retrieving trades
            cache_size: Size of the LRU cache for repository methods
            **kwargs: Additional arguments
        """
        super().__init__(data_source=None, storage_service=storage_service, cache_size=cache_size, **kwargs)
        
        # Initialize tables
        self._initialize_tables()
        
        logger.info("Initialized trade repository")
    
    def _initialize_tables(self) -> bool:
        """Initialize the trade tables.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return False
            
            # Create trade log table
            query = f"""
            CREATE TABLE IF NOT EXISTS `{self.storage_service.project_id}.{self.storage_service.dataset_id}.trade_logs` (
                trade_id STRING NOT NULL,
                ticker STRING NOT NULL,
                direction STRING NOT NULL,
                entry_date TIMESTAMP NOT NULL,
                entry_price FLOAT64 NOT NULL,
                position_size FLOAT64 NOT NULL,
                stop_loss FLOAT64,
                take_profit FLOAT64,
                exit_date TIMESTAMP,
                exit_price FLOAT64,
                pnl_amount FLOAT64,
                pnl_percentage FLOAT64,
                trade_duration_days FLOAT64,
                status STRING NOT NULL,
                trigger_reason STRING,
                exit_reason STRING,
                notes STRING,
                risk_percentage FLOAT64,
                reward_percentage FLOAT64,
                risk_reward_ratio FLOAT64,
                model_confidence FLOAT64,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            PARTITION BY DATE(entry_date)
            CLUSTER BY ticker, status
            """
            
            self.storage_service.execute_query(query)
            
            # Create performance metrics table
            query = f"""
            CREATE TABLE IF NOT EXISTS `{self.storage_service.project_id}.{self.storage_service.dataset_id}.performance_metrics` (
                calculation_date TIMESTAMP NOT NULL,
                total_trades INT64 NOT NULL,
                win_count INT64 NOT NULL,
                loss_count INT64 NOT NULL,
                win_rate FLOAT64 NOT NULL,
                avg_profit FLOAT64 NOT NULL,
                avg_loss FLOAT64 NOT NULL,
                profit_factor FLOAT64 NOT NULL,
                total_pnl FLOAT64 NOT NULL,
                avg_trade_duration FLOAT64 NOT NULL,
                sharpe_ratio FLOAT64,
                sortino_ratio FLOAT64,
                max_drawdown FLOAT64,
                max_drawdown_percentage FLOAT64,
                created_at TIMESTAMP NOT NULL
            )
            PARTITION BY DATE(calculation_date)
            """
            
            self.storage_service.execute_query(query)
            
            logger.info("Successfully initialized trade tables")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trade tables: {str(e)}")
            return False
    
    def _apply_caching(self):
        """Apply LRU caching to repository methods."""
        # Apply caching to get_open_trades
        self._get_open_trades_impl = self.get_open_trades
        self.get_open_trades = functools.lru_cache(maxsize=self.cache_size)(self._get_open_trades_impl)
        
        # Apply caching to get_trade_history
        self._get_trade_history_impl = self.get_trade_history
        self.get_trade_history = functools.lru_cache(maxsize=self.cache_size)(self._get_trade_history_impl)
        
        # Apply caching to get_trade
        self._get_trade_impl = self.get_trade
        self.get_trade = functools.lru_cache(maxsize=self.cache_size)(self._get_trade_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for repository methods."""
        self.get_open_trades.cache_clear()
        self.get_trade_history.cache_clear()
        self.get_trade.cache_clear()
        
        logger.debug("Cleared cache for trade repository")
    
    def get_data(self, **kwargs) -> pd.DataFrame:
        """Get trade data.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with trade data
        """
        return self.get_trade_history(**kwargs)
    
    def store_data(self, data: Union[Trade, Dict[str, Any]], **kwargs) -> bool:
        """Store trade data.
        
        Args:
            data: Trade data to store
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(data, Trade):
            return self.store_trade(data)
        elif isinstance(data, dict):
            return self.store_trade(Trade.from_dict(data))
        else:
            logger.error(f"Invalid data type: {type(data)}")
            return False
    
    def store_trade(self, trade: Trade) -> bool:
        """Store a trade.
        
        Args:
            trade: Trade to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return False
            
            # Convert trade to dictionary
            trade_data = trade.to_dict()
            
            # Insert or update trade
            query = f"""
            MERGE `{self.storage_service.project_id}.{self.storage_service.dataset_id}.trade_logs` AS target
            USING (SELECT @trade_id AS trade_id) AS source
            ON target.trade_id = source.trade_id
            WHEN MATCHED THEN
                UPDATE SET
                    ticker = @ticker,
                    direction = @direction,
                    entry_date = @entry_date,
                    entry_price = @entry_price,
                    position_size = @position_size,
                    stop_loss = @stop_loss,
                    take_profit = @take_profit,
                    exit_date = @exit_date,
                    exit_price = @exit_price,
                    pnl_amount = @pnl_amount,
                    pnl_percentage = @pnl_percentage,
                    trade_duration_days = @trade_duration_days,
                    status = @status,
                    trigger_reason = @trigger_reason,
                    exit_reason = @exit_reason,
                    notes = @notes,
                    risk_percentage = @risk_percentage,
                    reward_percentage = @reward_percentage,
                    risk_reward_ratio = @risk_reward_ratio,
                    model_confidence = @model_confidence,
                    updated_at = @updated_at
            WHEN NOT MATCHED THEN
                INSERT (
                    trade_id, ticker, direction, entry_date, entry_price, position_size, 
                    stop_loss, take_profit, exit_date, exit_price, pnl_amount, pnl_percentage, 
                    trade_duration_days, status, trigger_reason, exit_reason, notes, 
                    risk_percentage, reward_percentage, risk_reward_ratio, model_confidence,
                    created_at, updated_at
                )
                VALUES (
                    @trade_id, @ticker, @direction, @entry_date, @entry_price, @position_size, 
                    @stop_loss, @take_profit, @exit_date, @exit_price, @pnl_amount, @pnl_percentage, 
                    @trade_duration_days, @status, @trigger_reason, @exit_reason, @notes, 
                    @risk_percentage, @reward_percentage, @risk_reward_ratio, @model_confidence,
                    @created_at, @updated_at
                )
            """
            
            # Execute query
            self.storage_service.execute_query(query, trade_data)
            
            # Clear cache
            self.clear_cache()
            
            logger.info(f"Successfully stored trade {trade.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing trade: {str(e)}")
            return False
    
    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade if found, None otherwise
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return None
            
            # Build query
            query = f"""
            SELECT * FROM `{self.storage_service.project_id}.{self.storage_service.dataset_id}.trade_logs`
            WHERE trade_id = @trade_id
            """
            
            params = {'trade_id': trade_id}
            
            # Execute query
            trade_df = self.storage_service.execute_query(query, params)
            
            if trade_df.empty:
                logger.warning(f"Trade {trade_id} not found")
                return None
            
            # Convert to Trade object
            trade_data = trade_df.iloc[0].to_dict()
            trade = Trade.from_dict(trade_data)
            
            logger.debug(f"Retrieved trade {trade_id}")
            return trade
            
        except Exception as e:
            logger.error(f"Error getting trade: {str(e)}")
            return None
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades.
        
        Returns:
            DataFrame with open trades
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return pd.DataFrame()
            
            # Build query
            query = f"""
            SELECT * FROM `{self.storage_service.project_id}.{self.storage_service.dataset_id}.trade_logs`
            WHERE status = '{TradeStatus.OPEN}'
            ORDER BY entry_date DESC
            """
            
            # Execute query
            trades_df = self.storage_service.execute_query(query)
            
            logger.info(f"Retrieved {len(trades_df)} open trades")
            return trades_df
            
        except Exception as e:
            logger.error(f"Error getting open trades: {str(e)}")
            return pd.DataFrame()
    
    def get_trade_history(self, ticker: Optional[str] = None, status: Optional[str] = None, 
                         limit: int = 1000) -> pd.DataFrame:
        """Get trade history.
        
        Args:
            ticker: Filter by stock symbol (None for all symbols)
            status: Filter by status (None for all statuses)
            limit: Maximum number of trades to return
            
        Returns:
            DataFrame with trade history
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return pd.DataFrame()
            
            # Build query
            query = f"""
            SELECT * FROM `{self.storage_service.project_id}.{self.storage_service.dataset_id}.trade_logs`
            """
            
            conditions = []
            params = {}
            
            if ticker is not None:
                conditions.append("ticker = @ticker")
                params['ticker'] = ticker
            
            if status is not None:
                conditions.append("status = @status")
                params['status'] = status
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY entry_date DESC LIMIT @limit"
            params['limit'] = limit
            
            # Execute query
            trades_df = self.storage_service.execute_query(query, params)
            
            logger.info(f"Retrieved {len(trades_df)} trades")
            return trades_df
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return pd.DataFrame()
    
    def store_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store performance metrics.
        
        Args:
            metrics: Dictionary with performance metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return False
            
            # Add calculation date and timestamp if not present
            if 'calculation_date' not in metrics:
                metrics['calculation_date'] = datetime.now()
            
            if 'created_at' not in metrics:
                metrics['created_at'] = datetime.now()
            
            # Insert into BigQuery
            query = f"""
            INSERT INTO `{self.storage_service.project_id}.{self.storage_service.dataset_id}.performance_metrics` (
                calculation_date, total_trades, win_count, loss_count, win_rate, 
                avg_profit, avg_loss, profit_factor, total_pnl, avg_trade_duration,
                sharpe_ratio, sortino_ratio, max_drawdown, max_drawdown_percentage, created_at
            ) VALUES (
                @calculation_date, @total_trades, @win_count, @loss_count, @win_rate, 
                @avg_profit, @avg_loss, @profit_factor, @total_pnl, @avg_trade_duration,
                @sharpe_ratio, @sortino_ratio, @max_drawdown, @max_drawdown_percentage, @created_at
            )
            """
            
            # Execute query
            self.storage_service.execute_query(query, metrics)
            
            logger.info("Successfully stored performance metrics")
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {str(e)}")
            return False
    
    def get_performance_metrics(self, limit: int = 10) -> pd.DataFrame:
        """Get performance metrics.
        
        Args:
            limit: Maximum number of metrics to return
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            # Check if storage service has execute_query method
            if not hasattr(self.storage_service, 'execute_query'):
                logger.warning("Storage service does not have execute_query method")
                return pd.DataFrame()
            
            # Build query
            query = f"""
            SELECT * FROM `{self.storage_service.project_id}.{self.storage_service.dataset_id}.performance_metrics`
            ORDER BY calculation_date DESC
            LIMIT @limit
            """
            
            params = {'limit': limit}
            
            # Execute query
            metrics_df = self.storage_service.execute_query(query, params)
            
            logger.info(f"Retrieved {len(metrics_df)} performance metrics")
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return pd.DataFrame()
