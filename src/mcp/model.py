"""
Model component of the MCP architecture for Oracle of Delphi.

This module handles data storage and retrieval.
"""

import os
import sqlite3
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MarketDataModel:
    """Model for market data storage and retrieval."""
    
    def __init__(self):
        self.db_dir = os.path.join('data')
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, 'market_data.db')
        self._ensure_database_setup()
    
    def _ensure_database_setup(self):
        """Ensure the SQLite database is set up correctly."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create market_data table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                data JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
            ''')
            
            # Create index on symbol and date
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
            ON market_data (symbol, date)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error setting up SQLite database: {str(e)}")
            raise
    
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve market data from SQLite.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            start_date: Start date for data retrieval (default: 90 days ago)
            end_date: End date for data retrieval (default: now)
        
        Returns:
            pd.DataFrame: DataFrame with time series data
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query data
            cursor.execute('''
            SELECT date, data
            FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date DESC
            ''', (symbol, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_list = []
            for date_str, data_json in rows:
                row_data = json.loads(data_json)
                row_data['date'] = pd.to_datetime(date_str)
                data_list.append(row_data)
            
            df = pd.DataFrame(data_list)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data from SQLite: {str(e)}")
            return pd.DataFrame()
    
    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store market data in SQLite.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data: DataFrame with time series data
        
        Returns:
            bool: Success status
        """
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    data.index = pd.to_datetime(data.index)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert data
            rows_inserted = 0
            for timestamp, row in data.iterrows():
                # Convert row to JSON
                row_dict = row.to_dict()
                row_json = json.dumps(row_dict)
                
                # Insert or replace data
                try:
                    cursor.execute('''
                    INSERT OR REPLACE INTO market_data (symbol, date, data)
                    VALUES (?, ?, ?)
                    ''', (symbol, timestamp.isoformat(), row_json))
                    rows_inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting row for {symbol} at {timestamp}: {str(e)}")
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully stored {rows_inserted} data points for {symbol} in SQLite")
            return True
        except Exception as e:
            logger.error(f"Error storing market data in SQLite: {str(e)}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of all available symbols in the database.
        
        Returns:
            List[str]: List of available symbols
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT symbol FROM market_data')
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
