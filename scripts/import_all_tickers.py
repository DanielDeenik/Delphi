#!/usr/bin/env python
"""
Import All Tickers Script

This script imports data for all tickers from AlphaVantage in batches,
respecting API rate limits.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alpha_vantage_batch_importer import AlphaVantageBatchImporter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"import_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Import all tickers from AlphaVantage')
    
    parser.add_argument('--api-key', help='AlphaVantage API key (default: from environment)')
    parser.add_argument('--ticker-file', help='File containing ticker symbols (one per line)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of tickers per batch (default: 5)')
    parser.add_argument('--max-batches', type=int, help='Maximum number of batches to process')
    parser.add_argument('--delay', type=int, default=0, help='Additional delay between batches in seconds (default: 0)')
    
    return parser.parse_args()

def load_tickers_from_file(file_path):
    """Load ticker symbols from a file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_default_tickers():
    """Load a default list of popular tickers."""
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CSCO',
        # Financial
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'PYPL', 'AXP',
        # Healthcare
        'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'CVS', 'AMGN', 'GILD', 'BIIB',
        # Consumer
        'WMT', 'TGT', 'COST', 'HD', 'MCD', 'SBUX', 'NKE', 'DIS', 'NFLX', 'CMCSA',
        # Industrial
        'GE', 'BA', 'CAT', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'GD',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'VLO', 'MPC', 'KMI',
        # Telecom
        'T', 'VZ', 'TMUS', 'CHTR', 'LUMN', 'DISH', 'ATUS', 'CCOI', 'ZAYO', 'CTL',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'ED', 'PEG', 'FE',
        # Real Estate
        'AMT', 'PLD', 'CCI', 'PSA', 'EQIX', 'DLR', 'O', 'AVB', 'EQR', 'SPG',
        # Materials
        'LIN', 'APD', 'ECL', 'SHW', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM', 'DOW'
    ]

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        logger.error("AlphaVantage API key not provided. Use --api-key or set ALPHA_VANTAGE_API_KEY environment variable.")
        return 1
    
    # Initialize importer
    importer = AlphaVantageBatchImporter(api_key)
    importer.batch_size = args.batch_size
    
    # Load tickers
    if args.ticker_file:
        try:
            tickers = load_tickers_from_file(args.ticker_file)
            logger.info(f"Loaded {len(tickers)} tickers from {args.ticker_file}")
        except Exception as e:
            logger.error(f"Error loading tickers from file: {str(e)}")
            return 1
    else:
        tickers = load_default_tickers()
        logger.info(f"Using default list of {len(tickers)} popular tickers")
    
    # Add tickers to the import queue
    importer.add_tickers(tickers)
    
    # Import in batches
    batch_count = 0
    total_success = 0
    total_failed = 0
    
    logger.info(f"Starting import of {len(tickers)} tickers in batches of {args.batch_size}")
    start_time = time.time()
    
    while True:
        # Check if we've reached the maximum number of batches
        if args.max_batches is not None and batch_count >= args.max_batches:
            logger.info(f"Reached maximum of {args.max_batches} batches")
            break
        
        # Import a batch
        logger.info(f"Processing batch {batch_count + 1}")
        batch_results = importer.import_batch()
        
        # Stop if no more pending tickers
        if not batch_results:
            logger.info("No more pending tickers")
            break
        
        # Update counts
        batch_success = sum(1 for success in batch_results.values() if success)
        batch_failed = len(batch_results) - batch_success
        
        total_success += batch_success
        total_failed += batch_failed
        
        logger.info(f"Batch {batch_count + 1} completed: {batch_success} succeeded, {batch_failed} failed")
        
        # Add additional delay if specified
        if args.delay > 0:
            logger.info(f"Waiting {args.delay} seconds before next batch")
            time.sleep(args.delay)
        
        batch_count += 1
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Log summary
    logger.info(f"Import completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Total batches: {batch_count}")
    logger.info(f"Total tickers: {total_success + total_failed}")
    logger.info(f"Successful imports: {total_success}")
    logger.info(f"Failed imports: {total_failed}")
    
    # Get final status
    status = importer.get_import_status()
    logger.info(f"Final import status: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
