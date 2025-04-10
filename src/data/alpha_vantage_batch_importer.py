"""
AlphaVantage Batch Importer

This module handles batch importing of stock data from AlphaVantage API
while respecting rate limits.
"""

import os
import time
import logging
import json
import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AlphaVantageBatchImporter:
    """
    Manages batch importing of stock data from AlphaVantage API.
    Handles rate limiting, error recovery, and data storage.
    """
    
    def __init__(self, api_key: str, db_path: str = None):
        """
        Initialize the batch importer.
        
        Args:
            api_key: AlphaVantage API key
            db_path: Path to SQLite database (default: data/market_data.db)
        """
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
        
        # Set up database
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, 'market_data.db')
        else:
            self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Rate limiting settings
        self.calls_per_minute = 5  # Free tier limit
        self.min_call_interval = 60 / self.calls_per_minute  # Seconds between calls
        self.last_call_time = 0
        
        # Batch processing settings
        self.batch_size = 5  # Process 5 tickers at a time
        self.max_retries = 3
        self.retry_delay = 5  # Initial retry delay in seconds
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
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
            
            # Create ticker_status table to track import status
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticker_status (
                symbol TEXT PRIMARY KEY,
                last_updated TIMESTAMP,
                status TEXT,
                priority INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_error TEXT,
                metadata JSON
            )
            ''')
            
            # Create api_usage table to track API usage
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN,
                response_time REAL,
                error TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to comply with API rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        if elapsed < self.min_call_interval:
            sleep_time = self.min_call_interval - elapsed
            logger.debug(f"Rate limit: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_api_call(self, params: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Make an API call to AlphaVantage with rate limiting.
        
        Args:
            params: API parameters
            
        Returns:
            Tuple of (success, response_data)
        """
        self._wait_for_rate_limit()
        
        start_time = time.time()
        success = False
        error = None
        
        try:
            response = requests.get(self.base_url, params=params)
            response_data = response.json()
            
            # Check for error messages
            if 'Error Message' in response_data:
                error = response_data['Error Message']
                logger.error(f"API error: {error}")
                return False, {'error': error}
            
            # Check for rate limit messages
            if 'Note' in response_data and 'call frequency' in response_data['Note']:
                error = response_data['Note']
                logger.warning(f"Rate limit warning: {error}")
                return False, {'error': error}
            
            success = True
            return True, response_data
            
        except Exception as e:
            error = str(e)
            logger.error(f"API call error: {error}")
            return False, {'error': error}
        finally:
            # Record API usage
            response_time = time.time() - start_time
            self._record_api_usage(params.get('function', 'unknown'), success, response_time, error)
    
    def _record_api_usage(self, endpoint: str, success: bool, response_time: float, error: Optional[str] = None):
        """Record API usage for monitoring and optimization."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO api_usage (endpoint, timestamp, success, response_time, error)
            VALUES (?, ?, ?, ?, ?)
            ''', (endpoint, datetime.now().isoformat(), success, response_time, error))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error recording API usage: {str(e)}")
    
    def _update_ticker_status(self, symbol: str, status: str, error: Optional[str] = None):
        """Update the status of a ticker in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if ticker exists
            cursor.execute('SELECT symbol FROM ticker_status WHERE symbol = ?', (symbol,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing record
                if error:
                    cursor.execute('''
                    UPDATE ticker_status 
                    SET status = ?, last_updated = ?, last_error = ?, error_count = error_count + 1
                    WHERE symbol = ?
                    ''', (status, datetime.now().isoformat(), error, symbol))
                else:
                    cursor.execute('''
                    UPDATE ticker_status 
                    SET status = ?, last_updated = ?, error_count = 0, last_error = NULL
                    WHERE symbol = ?
                    ''', (status, datetime.now().isoformat(), symbol))
            else:
                # Insert new record
                cursor.execute('''
                INSERT INTO ticker_status (symbol, status, last_updated, last_error, error_count)
                VALUES (?, ?, ?, ?, ?)
                ''', (symbol, status, datetime.now().isoformat(), error, 1 if error else 0))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating ticker status: {str(e)}")
    
    def _store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store market data in SQLite.
        
        Args:
            symbol: Trading symbol
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
            
            logger.info(f"Successfully stored {rows_inserted} data points for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
            return False
    
    def _fetch_daily_adjusted(self, symbol: str, retries: int = 0) -> Optional[pd.DataFrame]:
        """
        Fetch daily adjusted time series for a symbol.
        
        Args:
            symbol: Trading symbol
            retries: Current retry count
            
        Returns:
            DataFrame with time series data or None on failure
        """
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        success, response_data = self._make_api_call(params)
        
        if not success:
            error = response_data.get('error', 'Unknown error')
            
            # Handle rate limiting
            if 'call frequency' in error:
                # Wait longer than the standard rate limit
                logger.warning(f"Rate limit exceeded. Waiting for 60 seconds.")
                time.sleep(60)
                
                # Retry if under max retries
                if retries < self.max_retries:
                    logger.info(f"Retrying {symbol} (attempt {retries + 1}/{self.max_retries})")
                    return self._fetch_daily_adjusted(symbol, retries + 1)
            
            # Handle other errors
            elif retries < self.max_retries:
                # Exponential backoff
                wait_time = self.retry_delay * (2 ** retries)
                logger.info(f"Retrying {symbol} in {wait_time} seconds (attempt {retries + 1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._fetch_daily_adjusted(symbol, retries + 1)
            
            # Update ticker status with error
            self._update_ticker_status(symbol, 'failed', error)
            return None
        
        # Process successful response
        try:
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in response_data:
                error = f"Missing time series data in response for {symbol}"
                logger.error(error)
                self._update_ticker_status(symbol, 'failed', error)
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(response_data[time_series_key], orient='index')
            
            # Rename columns
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adjusted_Close',
                '6. volume': 'Volume',
                '7. dividend amount': 'Dividend',
                '8. split coefficient': 'Split_Coefficient'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Update ticker status
            self._update_ticker_status(symbol, 'success')
            
            return df
            
        except Exception as e:
            error = f"Error processing data for {symbol}: {str(e)}"
            logger.error(error)
            self._update_ticker_status(symbol, 'failed', error)
            return None
    
    def add_tickers(self, symbols: List[str], priority: int = 0):
        """
        Add tickers to the import queue.
        
        Args:
            symbols: List of ticker symbols
            priority: Priority level (higher = more important)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for symbol in symbols:
                # Check if ticker exists
                cursor.execute('SELECT symbol FROM ticker_status WHERE symbol = ?', (symbol,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update priority if higher
                    cursor.execute('''
                    UPDATE ticker_status 
                    SET priority = MAX(priority, ?)
                    WHERE symbol = ?
                    ''', (priority, symbol))
                else:
                    # Insert new record
                    cursor.execute('''
                    INSERT INTO ticker_status (symbol, status, priority, last_updated)
                    VALUES (?, ?, ?, ?)
                    ''', (symbol, 'pending', priority, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added {len(symbols)} tickers to the import queue")
        except Exception as e:
            logger.error(f"Error adding tickers: {str(e)}")
    
    def get_pending_tickers(self, limit: int = 100) -> List[str]:
        """
        Get a list of pending tickers to import.
        
        Args:
            limit: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get tickers ordered by priority (high to low) and then by last updated (oldest first)
            cursor.execute('''
            SELECT symbol FROM ticker_status
            WHERE status = 'pending' OR (status = 'failed' AND error_count < ?)
            ORDER BY priority DESC, last_updated ASC
            LIMIT ?
            ''', (self.max_retries, limit))
            
            symbols = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return symbols
        except Exception as e:
            logger.error(f"Error getting pending tickers: {str(e)}")
            return []
    
    def import_ticker(self, symbol: str) -> bool:
        """
        Import data for a single ticker.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            bool: Success status
        """
        logger.info(f"Importing data for {symbol}")
        
        # Fetch data
        df = self._fetch_daily_adjusted(symbol)
        
        if df is None:
            return False
        
        # Store data
        success = self._store_market_data(symbol, df)
        
        return success
    
    def import_batch(self, batch_size: Optional[int] = None) -> Dict[str, bool]:
        """
        Import a batch of tickers.
        
        Args:
            batch_size: Number of tickers to process (default: self.batch_size)
            
        Returns:
            Dict mapping symbols to success status
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Get pending tickers
        symbols = self.get_pending_tickers(batch_size)
        
        if not symbols:
            logger.info("No pending tickers to import")
            return {}
        
        logger.info(f"Importing batch of {len(symbols)} tickers")
        
        results = {}
        for symbol in symbols:
            success = self.import_ticker(symbol)
            results[symbol] = success
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Batch import completed. Success: {success_count}/{len(results)}")
        
        return results
    
    def import_all(self, max_batches: Optional[int] = None) -> Dict[str, bool]:
        """
        Import all pending tickers in batches.
        
        Args:
            max_batches: Maximum number of batches to process (default: unlimited)
            
        Returns:
            Dict mapping symbols to success status
        """
        batch_count = 0
        all_results = {}
        
        while True:
            # Check if we've reached the maximum number of batches
            if max_batches is not None and batch_count >= max_batches:
                logger.info(f"Reached maximum of {max_batches} batches")
                break
            
            # Import a batch
            batch_results = self.import_batch()
            
            # Update overall results
            all_results.update(batch_results)
            
            # Stop if no more pending tickers
            if not batch_results:
                break
            
            batch_count += 1
        
        # Log summary
        success_count = sum(1 for success in all_results.values() if success)
        logger.info(f"Import completed. Success: {success_count}/{len(all_results)}")
        
        return all_results
    
    def get_import_status(self) -> Dict[str, Any]:
        """
        Get the current import status.
        
        Returns:
            Dict with import statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts by status
            cursor.execute('''
            SELECT status, COUNT(*) FROM ticker_status
            GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())
            
            # Get total ticker count
            cursor.execute('SELECT COUNT(*) FROM ticker_status')
            total_tickers = cursor.fetchone()[0]
            
            # Get API usage statistics
            cursor.execute('''
            SELECT 
                COUNT(*) as total_calls,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_calls,
                AVG(response_time) as avg_response_time
            FROM api_usage
            WHERE timestamp > ?
            ''', ((datetime.now() - timedelta(days=1)).isoformat(),))
            
            api_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_tickers': total_tickers,
                'status_counts': status_counts,
                'api_usage_24h': {
                    'total_calls': api_stats[0],
                    'successful_calls': api_stats[1],
                    'success_rate': api_stats[1] / api_stats[0] if api_stats[0] > 0 else 0,
                    'avg_response_time': api_stats[2]
                }
            }
        except Exception as e:
            logger.error(f"Error getting import status: {str(e)}")
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get API key from environment
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
        exit(1)
    
    # Initialize importer
    importer = AlphaVantageBatchImporter(api_key)
    
    # Add some tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
    importer.add_tickers(test_tickers)
    
    # Import a batch
    results = importer.import_batch(3)  # Import 3 tickers
    
    # Print results
    for symbol, success in results.items():
        print(f"{symbol}: {'Success' if success else 'Failed'}")
    
    # Print status
    status = importer.get_import_status()
    print(f"Import status: {status}")
