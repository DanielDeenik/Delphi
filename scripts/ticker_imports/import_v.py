#!/usr/bin/env python3
"""
Import script for V data.

This script imports time series data for V from Alpha Vantage to BigQuery.
It can be used to update the data for the V notebook.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from trading_ai module
from trading_ai.config import config_manager
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage
from import_time_series import TimeSeriesImporter

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"import_V_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Import V data from Alpha Vantage to BigQuery")
    
    # Import options
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--days", type=int, default=365, help="Number of days of data to import")
    parser.add_argument("--repair-missing", action="store_true", help="Check and repair missing dates")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing dates without importing")
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        # Initialize importer
        importer = TimeSeriesImporter()
        
        # Define ticker
        ticker = "V"
        
        logger.info(f"Starting import for {ticker}")
        
        # Check for missing dates if requested
        if args.check_missing:
            missing_info = importer.check_missing_dates(ticker, args.days)
            if missing_info["missing_count"] > 0:
                logger.warning(f"Found {missing_info['missing_count']} missing dates for {ticker}")
                for date in missing_info["missing_dates"]:
                    logger.warning(f"  Missing: {date}")
            else:
                logger.info(f"No missing dates found for {ticker}")
            return 0
        
        # Repair missing data if requested
        if args.repair_missing:
            success = importer.repair_missing_data(ticker, args.days)
            if success:
                logger.info(f"Successfully repaired missing data for {ticker}")
            else:
                logger.error(f"Failed to repair missing data for {ticker}")
            return 0 if success else 1
        
        # Import data
        success = importer.import_ticker_data(
            ticker,
            force_full=args.force_full,
            repair_missing=args.repair_missing,
            days=args.days
        )
        
        if success:
            logger.info(f"Successfully imported data for {ticker}")
        else:
            logger.error(f"Failed to import data for {ticker}")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
