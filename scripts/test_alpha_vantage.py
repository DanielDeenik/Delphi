#!/usr/bin/env python
"""
Test script for Alpha Vantage client with SQLite storage.

This script demonstrates how to use the Alpha Vantage client
to fetch and store market data in SQLite.

Usage:
    python scripts/test_alpha_vantage.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    try:
        # Import the Alpha Vantage client
        from src.data.alpha_vantage_client import AlphaVantageClient
        
        # Check if API key is set
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
            logger.info("Please set your Alpha Vantage API key:")
            logger.info("export ALPHA_VANTAGE_API_KEY=your_api_key")
            return
        
        # Initialize the client
        logger.info("Initializing Alpha Vantage client")
        client = AlphaVantageClient()
        
        # Fetch data for a symbol
        symbol = 'AAPL'
        logger.info(f"Fetching data for {symbol}")
        data = client.fetch_daily_adjusted(symbol)
        
        if data is None:
            logger.error(f"Failed to fetch data for {symbol}")
            return
        
        # Display the data
        logger.info(f"Successfully fetched data for {symbol}")
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Latest data point: {data.iloc[0].to_dict()}")
        
        # Check if data was stored in SQLite
        logger.info("Checking if data was stored in SQLite")
        from src.services.sqlite_storage_service import SQLiteStorageService
        
        sqlite_service = SQLiteStorageService()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        stored_data = sqlite_service.get_market_data(symbol, start_date, end_date)
        
        if stored_data.empty:
            logger.warning(f"No data found in SQLite for {symbol}")
        else:
            logger.info(f"Successfully retrieved {len(stored_data)} rows from SQLite")
            logger.info(f"SQLite data columns: {stored_data.columns.tolist()}")
            logger.info(f"Latest SQLite data point: {stored_data.iloc[0].to_dict()}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
