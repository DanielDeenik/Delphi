"""Time series data storage service using SQLite."""
import os
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import json

logger = logging.getLogger(__name__)

class SQLiteStorageService:
    """Manages time series data storage in SQLite with path to BigQuery migration."""

    def __init__(self):
        # SQLite database path
        self.db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, 'market_data.db')
        
        # Initialize database
        self._ensure_storage_setup()

    def _ensure_storage_setup(self):
        """Ensure required SQLite database and tables exist."""
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

    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store market data in SQLite.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data: DataFrame with time series data
                Expected columns: Open, High, Low, Close, Volume, etc.
                Index should be datetime
        
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

    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve market data from SQLite.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            start_date: Start date for data retrieval (default: 30 days ago)
            end_date: End date for data retrieval (default: now)
        
        Returns:
            pd.DataFrame: DataFrame with time series data
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query data
            cursor.execute('''
            SELECT date, data
            FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
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
    
    def export_to_bigquery(self, symbol: str, project_id: str, dataset_id: str = "market_data",
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> bool:
        """Export data from SQLite to BigQuery for long-term storage.
        
        This provides the migration path to BigQuery when needed.
        
        Args:
            symbol: Trading symbol to export
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            start_date: Start date for data export
            end_date: End date for data export
            
        Returns:
            bool: Success status
        """
        try:
            # Get data from SQLite
            df = self.get_market_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data to export for {symbol}")
                return False
            
            # Export to BigQuery
            table_id = f"{project_id}.{dataset_id}.{symbol.lower()}_daily"
            df.to_gbq(table_id, project_id=project_id, if_exists='append')
            
            logger.info(f"Successfully exported {len(df)} rows for {symbol} to BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to BigQuery: {str(e)}")
            return False
