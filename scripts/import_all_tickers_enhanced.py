#!/usr/bin/env python
"""
Enhanced Import All Tickers Script

This script imports data for all tickers from AlphaVantage in batches,
with intelligent prioritization and rate limiting.
"""

import os
import sys
import logging
import argparse
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alpha_vantage_batch_importer import AlphaVantageBatchImporter
from src.data.ticker_universe import TickerUniverse

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
    parser = argparse.ArgumentParser(description='Import all tickers from AlphaVantage with enhanced features')
    
    parser.add_argument('--api-key', help='AlphaVantage API key (default: from environment)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of tickers per batch (default: 5)')
    parser.add_argument('--max-batches', type=int, help='Maximum number of batches to process')
    parser.add_argument('--delay', type=int, default=0, help='Additional delay between batches in seconds (default: 0)')
    parser.add_argument('--priority-file', help='JSON file with ticker priorities')
    parser.add_argument('--exchange', help='Filter tickers by exchange (NYSE, NASDAQ)')
    parser.add_argument('--sector', help='Filter tickers by sector')
    parser.add_argument('--min-market-cap', type=float, help='Minimum market cap in billions')
    parser.add_argument('--refresh-tickers', action='store_true', help='Refresh ticker list from sources')
    parser.add_argument('--status-interval', type=int, default=5, help='Interval (in batches) to print status updates')
    
    return parser.parse_args()

def load_ticker_priorities(file_path: str) -> Dict[str, int]:
    """Load ticker priorities from a JSON file."""
    priorities = {}
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                priorities = json.load(f)
            logger.info(f"Loaded priorities for {len(priorities)} tickers from {file_path}")
        else:
            logger.warning(f"Priority file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading ticker priorities: {str(e)}")
    
    return priorities

def save_import_status(importer: AlphaVantageBatchImporter, file_path: str):
    """Save the current import status to a JSON file."""
    try:
        status = importer.get_import_status()
        
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2)
            
        logger.info(f"Saved import status to {file_path}")
    except Exception as e:
        logger.error(f"Error saving import status: {str(e)}")

def prioritize_tickers(tickers: List[Dict], priorities: Dict[str, int]) -> List[Dict]:
    """
    Prioritize tickers based on provided priorities and market cap.
    
    Args:
        tickers: List of ticker dictionaries
        priorities: Dictionary mapping symbols to priority values
        
    Returns:
        List of ticker dictionaries with 'priority' field added
    """
    for ticker in tickers:
        symbol = ticker.get('symbol', '')
        
        # Start with base priority
        priority = 0
        
        # Add priority from priorities file if available
        if symbol in priorities:
            priority += priorities[symbol]
        
        # Add priority based on market cap
        market_cap = ticker.get('market_cap', 0)
        if isinstance(market_cap, str):
            try:
                market_cap = float(market_cap.replace(',', ''))
            except:
                market_cap = 0
        
        # Convert to billions and add priority (higher market cap = higher priority)
        market_cap_billions = market_cap / 1_000_000_000
        if market_cap_billions > 100:
            priority += 30  # Mega cap
        elif market_cap_billions > 10:
            priority += 20  # Large cap
        elif market_cap_billions > 2:
            priority += 10  # Mid cap
        
        # Add priority field
        ticker['priority'] = priority
    
    # Sort by priority (highest first)
    return sorted(tickers, key=lambda x: x.get('priority', 0), reverse=True)

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
    
    # Initialize ticker universe
    universe = TickerUniverse()
    
    # Load ticker priorities if provided
    priorities = {}
    if args.priority_file:
        priorities = load_ticker_priorities(args.priority_file)
    
    # Get tickers based on filters
    if args.exchange or args.sector or args.min_market_cap is not None:
        tickers = universe.filter_tickers(
            exchange=args.exchange,
            sector=args.sector,
            min_market_cap=args.min_market_cap
        )
        logger.info(f"Filtered to {len(tickers)} tickers based on criteria")
    else:
        tickers = universe.get_all_tickers(refresh=args.refresh_tickers)
        logger.info(f"Using all {len(tickers)} available tickers")
    
    if not tickers:
        logger.error("No tickers to import")
        return 1
    
    # Prioritize tickers
    prioritized_tickers = prioritize_tickers(tickers, priorities)
    
    # Extract symbols and priorities
    symbols_with_priorities = [(t['symbol'], t.get('priority', 0)) for t in prioritized_tickers]
    
    # Add tickers to the import queue with priorities
    for symbol, priority in symbols_with_priorities:
        importer.add_tickers([symbol], priority=priority)
    
    # Import in batches
    batch_count = 0
    total_success = 0
    total_failed = 0
    
    logger.info(f"Starting import of {len(symbols_with_priorities)} tickers in batches of {args.batch_size}")
    start_time = time.time()
    
    # Create status file path
    status_file = f"import_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
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
            
            # Save status at intervals
            if batch_count % args.status_interval == 0:
                save_import_status(importer, status_file)
            
            # Add additional delay if specified
            if args.delay > 0:
                logger.info(f"Waiting {args.delay} seconds before next batch")
                time.sleep(args.delay)
            
            batch_count += 1
            
    except KeyboardInterrupt:
        logger.info("Import interrupted by user")
    except Exception as e:
        logger.error(f"Error during import: {str(e)}")
    finally:
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Log summary
        logger.info(f"Import completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info(f"Total batches: {batch_count}")
        logger.info(f"Total tickers processed: {total_success + total_failed}")
        logger.info(f"Successful imports: {total_success}")
        logger.info(f"Failed imports: {total_failed}")
        
        # Save final status
        save_import_status(importer, status_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
