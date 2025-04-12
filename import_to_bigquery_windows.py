#!/usr/bin/env python3
"""
Import stock data to BigQuery using the trading_ai codebase.
Modified for Windows compatibility.
"""
import os
import json
import asyncio
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime

# Fix for Windows event loop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_config():
    """Set up configuration files if they don't exist."""
    try:
        # Create config directory
        config_dir = Path("config")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create system config if it doesn't exist
        system_config_path = config_dir / "system_config.json"
        if not system_config_path.exists():
            # Get API key from environment or input
            alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if not alpha_vantage_key:
                alpha_vantage_key = input("Enter your Alpha Vantage API key: ")
            
            # Get Google Cloud project ID
            google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT", "delphi-449908")
            
            # Create system config
            system_config = {
                "google_cloud_project": google_cloud_project,
                "bigquery_dataset": "trading_insights",
                "alpha_vantage_api_key": alpha_vantage_key,
                "trading_days_lookback": 252,
                "volume_ma_periods": [5, 20, 50]
            }
            
            # Save system config
            with open(system_config_path, "w") as f:
                json.dump(system_config, f, indent=2)
            
            logger.info(f"Created system config at {system_config_path}")
        
        # Create tracked stocks config if it doesn't exist
        stocks_config_path = config_dir / "tracked_stocks.json"
        if not stocks_config_path.exists():
            # Default tracked stocks
            tracked_stocks = {
                "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", 
                       "TSLA", "META", "ADBE", "ORCL", "ASML"],
                "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", 
                         "PLTR", "UBER", "LCID", "INTC", "XPEV"]
            }
            
            # Save tracked stocks
            with open(stocks_config_path, "w") as f:
                json.dump(tracked_stocks, f, indent=2)
            
            logger.info(f"Created tracked stocks config at {stocks_config_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up configuration: {str(e)}")
        return False

def import_data_to_bigquery(force_full=False, batch_size=5):
    """
    Import stock data to BigQuery using synchronous methods.
    
    Args:
        force_full: Whether to force a full data refresh
        batch_size: Number of stocks to process in each batch
    """
    try:
        # Import required modules
        from trading_ai.config import config_manager
        from trading_ai.core.alpha_client import AlphaVantageClient
        from trading_ai.core.bigquery_io import BigQueryStorage
        from trading_ai.core.volume_footprint import VolumeAnalyzer
        
        # Initialize BigQuery storage
        bigquery_storage = BigQueryStorage()
        
        # Initialize tables
        logger.info("Initializing BigQuery tables...")
        bigquery_storage.initialize_tables()
        
        # Initialize clients
        alpha_client = AlphaVantageClient()
        volume_analyzer = VolumeAnalyzer()
        
        # Get all tracked tickers
        all_tickers = config_manager.get_all_tickers()
        logger.info(f"Found {len(all_tickers)} tracked tickers: {', '.join(all_tickers)}")
        
        # Split tickers into batches
        batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]
        logger.info(f"Processing {len(batches)} batches of {batch_size} tickers each")
        
        # Process each batch
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {', '.join(batch)}")
            
            # Process each ticker in the batch
            for ticker in batch:
                # Determine output size based on existing data
                outputsize = 'full' if force_full else 'compact'
                
                # Fetch daily data (using synchronous method)
                logger.info(f"Fetching {outputsize} data for {ticker}...")
                df = alpha_client.fetch_daily_data_sync(ticker, outputsize)
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue
                
                # Store in BigQuery
                logger.info(f"Storing data for {ticker} in BigQuery...")
                success = bigquery_storage.store_stock_prices(ticker, df)
                
                if success:
                    logger.info(f"Successfully stored data for {ticker}")
                    
                    # Process volume analysis
                    logger.info(f"Processing volume analysis for {ticker}...")
                    analysis_df = volume_analyzer.analyze(df)
                    
                    # Store volume analysis
                    logger.info(f"Storing volume analysis for {ticker}...")
                    bigquery_storage.store_volume_analysis(ticker, analysis_df)
                else:
                    logger.warning(f"Failed to store data for {ticker}")
            
            # Wait between batches (except for the last batch)
            if i < len(batches) - 1:
                logger.info("Waiting 60 seconds before processing next batch...")
                time.sleep(60)
        
        # Generate master summary
        logger.info("Generating master summary...")
        # Get summaries for each ticker
        summaries = []
        for ticker in all_tickers:
            # Get volume analysis
            analysis_df = bigquery_storage.get_volume_analysis(ticker, days=30)
            
            if not analysis_df.empty:
                # Get latest data point
                latest = analysis_df.iloc[0]  # Assuming sorted by date desc
                
                # Create summary entry
                summary_entry = {
                    'date': latest['date'],
                    'symbol': ticker,
                    'direction': config_manager.get_ticker_direction(ticker),
                    'close': latest.get('close', 0),
                    'volume': latest.get('volume', 0),
                    'relative_volume': latest.get('relative_volume_20d', 0),
                    'volume_z_score': latest.get('volume_z_score', 0),
                    'is_volume_spike': latest.get('is_volume_spike', False),
                    'spike_strength': latest.get('spike_strength', 0),
                    'price_change_pct': latest.get('price_change_pct', 0),
                    'signal': latest.get('signal', 'NEUTRAL'),
                    'confidence': latest.get('signal_strength', 0),
                    'notes': latest.get('notes', ''),
                    'notebook_url': f"https://colab.research.google.com/drive/your-notebook-id-for-{ticker}",
                    'timestamp': datetime.now()
                }
                
                summaries.append(summary_entry)
        
        # Create DataFrame
        import pandas as pd
        summary_df = pd.DataFrame(summaries)
        
        # Store in BigQuery
        if not summary_df.empty:
            bigquery_storage.store_master_summary(summary_df)
        
        logger.info("Data import completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error importing data to BigQuery: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Import stock data to BigQuery")
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of stocks to process in each batch")
    args = parser.parse_args()
    
    # Set up configuration
    if not setup_config():
        logger.error("Failed to set up configuration")
        return False
    
    # Import data to BigQuery
    success = import_data_to_bigquery(args.force_full, args.batch_size)
    
    if success:
        print("\nData import completed successfully!")
    else:
        print("\nData import failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    # Add missing imports for synchronous version
    import time
    
    # Add synchronous fetch method to AlphaVantageClient
    from trading_ai.core.alpha_client import AlphaVantageClient
    
    # Add synchronous fetch method if it doesn't exist
    if not hasattr(AlphaVantageClient, 'fetch_daily_data_sync'):
        def fetch_daily_data_sync(self, symbol, outputsize='full'):
            """Synchronous version of fetch_daily_data."""
            import pandas as pd
            import requests
            
            # Build request parameters
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.api_key
            }
            
            # Make the API request
            logger.info(f"Fetching {outputsize} daily data for {symbol}...")
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for errors
            if 'Time Series (Daily)' not in data:
                error_msg = data.get('Note', data.get('Error Message', 'Unknown error'))
                logger.error(f"Error fetching data for {symbol}: {error_msg}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            
            # Map column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split_coefficient'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Sort by date (newest first)
            df = df.sort_index(ascending=False)
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            return df
        
        # Add method to AlphaVantageClient class
        AlphaVantageClient.fetch_daily_data_sync = fetch_daily_data_sync
    
    # Run the main function
    main()
