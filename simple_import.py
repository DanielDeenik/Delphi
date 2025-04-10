#!/usr/bin/env python
"""
Simple script to import market data from Alpha Vantage into SQLite.

This script is self-contained and doesn't rely on the project structure.
"""

import os
import sys
import logging
import sqlite3
import json
import requests
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAlphaVantageClient:
    """Simple client for fetching market data from Alpha Vantage API."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'

    def fetch_daily_adjusted(self, symbol):
        """Fetch daily adjusted time series data."""
        try:
            # Fetch from Alpha Vantage
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.api_key
            }

            logger.info(f"Fetching data for {symbol} from Alpha Vantage...")
            response = requests.get(self.base_url, params=params)
            data = response.json()

            if 'Time Series (Daily)' not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching data for {symbol}: {error_msg}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

            # Map column names
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

            # Check for missing columns
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns in API response for {symbol}: {missing_columns}")
                # Continue anyway, just map the columns that exist
                column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}

            # Rename columns
            df = df.rename(columns=column_mapping)

            # Convert index to datetime
            df.index = pd.to_datetime(df.index)

            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Forward fill any NaN values
            df = df.ffill()

            logger.info(f"Successfully fetched data for {symbol}. Shape: {df.shape}")
            
            # Store in SQLite
            self.store_in_sqlite(symbol, df)

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def store_in_sqlite(self, symbol, df):
        """Store market data in SQLite."""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Connect to SQLite database
            db_path = os.path.join('data', 'market_data.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
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
            
            # Create index if it doesn't exist
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
            ON market_data (symbol, date)
            ''')
            
            # Insert data
            rows_inserted = 0
            for timestamp, row in df.iterrows():
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
            
            logger.info(f"Successfully stored {rows_inserted} data points for {symbol} in SQLite at {db_path}")
            return True
        except Exception as e:
            logger.error(f"Error storing market data in SQLite: {str(e)}")
            return False

def main():
    """Main entry point."""
    # Check if API key is set
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        # Try to read from .env file
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if line.startswith('ALPHA_VANTAGE_API_KEY='):
                        os.environ['ALPHA_VANTAGE_API_KEY'] = line.strip().split('=', 1)[1]
                        api_key = os.environ['ALPHA_VANTAGE_API_KEY']
                        break
    
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY not set. Please set it in your environment or .env file.")
        return
    
    # Initialize the client
    client = SimpleAlphaVantageClient(api_key)
    
    # Default symbols to import
    default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get symbols from command line arguments or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else default_symbols
    
    logger.info(f"Importing data for {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Import data for each symbol
    success_count = 0
    for symbol in symbols:
        data = client.fetch_daily_adjusted(symbol)
        if data is not None and not data.empty:
            success_count += 1
            # Alpha Vantage has a rate limit of 5 calls per minute for free API keys
            if symbol != symbols[-1]:
                logger.info("Waiting 15 seconds due to Alpha Vantage rate limits...")
                import time
                time.sleep(15)
    
    logger.info(f"Import completed. Successfully imported {success_count}/{len(symbols)} symbols.")
    logger.info(f"Data stored in SQLite database: {os.path.join(os.getcwd(), 'data', 'market_data.db')}")

if __name__ == "__main__":
    main()
