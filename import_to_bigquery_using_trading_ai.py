#!/usr/bin/env python3
"""
Import stock data to BigQuery using the trading_ai codebase.
"""
import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime

from trading_ai.config import config_manager
from trading_ai.core.data_ingestion import DataIngestionManager
from trading_ai.core.bigquery_io import BigQueryStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def import_data_to_bigquery(force_full=False, batch_size=5):
    """
    Import stock data to BigQuery.
    
    Args:
        force_full: Whether to force a full data refresh
        batch_size: Number of stocks to process in each batch
    """
    try:
        # Initialize BigQuery storage
        bigquery_storage = BigQueryStorage()
        
        # Initialize tables
        logger.info("Initializing BigQuery tables...")
        bigquery_storage.initialize_tables()
        
        # Initialize data ingestion manager
        ingestion_manager = DataIngestionManager()
        
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
                # Ingest stock data
                logger.info(f"Ingesting data for {ticker}...")
                success = await ingestion_manager.ingest_stock_data(ticker, force_full)
                
                if success:
                    # Process volume analysis
                    logger.info(f"Processing volume analysis for {ticker}...")
                    await ingestion_manager.process_volume_analysis(ticker)
                else:
                    logger.warning(f"Failed to ingest data for {ticker}, skipping volume analysis")
            
            # Wait between batches (except for the last batch)
            if i < len(batches) - 1:
                logger.info("Waiting 60 seconds before processing next batch...")
                await asyncio.sleep(60)
        
        # Generate master summary
        logger.info("Generating master summary...")
        await ingestion_manager.generate_master_summary()
        
        logger.info("Data import completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error importing data to BigQuery: {str(e)}")
        return False

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
        
        # Reload configuration
        config_manager.load_config()
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up configuration: {str(e)}")
        return False

async def main():
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
    success = await import_data_to_bigquery(args.force_full, args.batch_size)
    
    if success:
        print("\nData import completed successfully!")
    else:
        print("\nData import failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
