#!/usr/bin/env python3
"""
Command-line interface for importing time series data.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from trading_ai module
from trading_ai.config import config_manager
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Import time series data from Alpha Vantage to BigQuery")
    
    # Ticker options
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker", type=str, help="Import a specific ticker")
    ticker_group.add_argument("--tickers", type=str, nargs="+", help="Import multiple tickers")
    ticker_group.add_argument("--tickers-file", type=str, help="File with tickers to import (one per line)")
    
    # Import options
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--days", type=int, default=365, help="Number of days of data to import")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of tickers to process in each batch")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of worker threads")
    
    # Repair options
    parser.add_argument("--repair-missing", action="store_true", help="Check and repair missing dates")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed imports")
    
    # Report options
    parser.add_argument("--report", action="store_true", help="Generate import report")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing dates")
    
    return parser.parse_args()

def get_tickers(args) -> List[str]:
    """Get list of tickers to import."""
    if args.ticker:
        return [args.ticker]
    elif args.tickers:
        return args.tickers
    elif args.tickers_file:
        with open(args.tickers_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return config_manager.get_all_tickers()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Import the TimeSeriesImporter class
        from run_time_series_import import TimeSeriesImporter
        
        # Initialize importer
        importer = TimeSeriesImporter()
        
        # Generate report if requested
        if args.report:
            report = importer.generate_import_report()
            print(json.dumps(report, indent=2))
            return 0
        
        # Get tickers to import
        tickers = get_tickers(args)
        
        if not tickers:
            logger.error("No tickers specified")
            return 1
        
        logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
        
        # Check for missing dates if requested
        if args.check_missing:
            for ticker in tickers:
                missing_info = importer.check_missing_dates(ticker, args.days)
                if missing_info["status"] == "incomplete":
                    print(f"{ticker}: {missing_info['missing_count']} missing dates")
                    print(f"  Missing dates: {', '.join(missing_info['missing_dates'][:5])}...")
                elif missing_info["status"] == "error":
                    print(f"{ticker}: Error - {missing_info.get('message')}")
                else:
                    print(f"{ticker}: No missing dates")
            return 0
        
        # Retry failed imports if requested
        if args.retry_failed:
            results = importer.retry_failed_imports(args.batch_size, args.days)
            success_count = sum(1 for status in results.values() if status)
            logger.info(f"Retry summary: {success_count}/{len(results)} tickers imported successfully")
            return 0
        
        # Repair missing data if requested
        if args.repair_missing:
            results = {}
            for ticker in tickers:
                results[ticker] = importer.repair_missing_data(ticker, args.days)
            
            success_count = sum(1 for status in results.values() if status)
            logger.info(f"Repair summary: {success_count}/{len(results)} tickers repaired successfully")
            return 0
        
        # Import data for tickers
        results = importer.import_tickers(
            tickers, 
            args.batch_size, 
            args.force_full, 
            args.repair_missing,
            args.days,
            args.max_workers
        )
        
        success_count = sum(1 for status in results.values() if status)
        logger.info(f"Import summary: {success_count}/{len(results)} tickers imported successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
