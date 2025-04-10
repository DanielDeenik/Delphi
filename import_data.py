#!/usr/bin/env python
"""
Import market data from Alpha Vantage into SQLite.

This script fetches data for specified symbols and stores it in SQLite.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the required modules
try:
    from src.data.alpha_vantage_client import AlphaVantageClient
    from src.services.sqlite_storage_service import SQLiteStorageService
except ImportError:
    # If imports fail, add the project root to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from src.data.alpha_vantage_client import AlphaVantageClient
        from src.services.sqlite_storage_service import SQLiteStorageService
    except ImportError:
        logger.error("Could not import required modules. Make sure you're running this script from the project root.")
        sys.exit(1)

def import_symbol(symbol, client):
    """Import data for a single symbol."""
    logger.info(f"Fetching data for {symbol}...")
    try:
        # Fetch data from Alpha Vantage
        data = client.fetch_daily_adjusted(symbol)
        
        if data is None or data.empty:
            logger.error(f"Failed to fetch data for {symbol}")
            return False
        
        logger.info(f"Successfully fetched data for {symbol}. Shape: {data.shape}")
        return True
    except Exception as e:
        logger.error(f"Error importing {symbol}: {str(e)}")
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
    logger.info("Initializing Alpha Vantage client...")
    client = AlphaVantageClient()
    
    # Default symbols to import
    default_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get symbols from command line arguments or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else default_symbols
    
    logger.info(f"Importing data for {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Import data for each symbol
    success_count = 0
    for symbol in symbols:
        if import_symbol(symbol, client):
            success_count += 1
    
    logger.info(f"Import completed. Successfully imported {success_count}/{len(symbols)} symbols.")
    
    # Check the SQLite database
    sqlite_service = SQLiteStorageService()
    db_path = sqlite_service.db_path
    logger.info(f"Data stored in SQLite database: {db_path}")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

if __name__ == "__main__":
    main()
