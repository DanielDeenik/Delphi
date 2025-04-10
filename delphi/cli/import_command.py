"""
Import Command

This module provides a command-line interface for importing data.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional

from ..data import AlphaVantageClient, YFinanceClient
from ..data import BigQueryImporter
from ..utils.config import get_config, load_env
from ..utils.logger import get_logger

logger = logging.getLogger(__name__)

def import_data():
    """
    Import data from a data source to BigQuery.
    """
    # Load environment variables
    load_env()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Import data to BigQuery")
    parser.add_argument(
        "--source",
        choices=["alpha_vantage", "yfinance"],
        default="yfinance",
        help="Data source (default: yfinance)"
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Period to import (default: 2y)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to import (default: from config)"
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    parser.add_argument(
        "--table-id",
        help="BigQuery table ID (default: from config)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Log file (default: None)"
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = get_logger(__name__, getattr(logging, args.log_level), args.log_file)
    
    # Get config
    config = get_config()
    
    # Get symbols
    symbols = args.symbols
    if not symbols:
        symbols = config['symbols']['default']
    
    # Get symbol names
    symbol_names = config['symbols']['names']
    
    # Create data source
    if args.source == "alpha_vantage":
        # Get API key
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            api_key = input("Enter your Alpha Vantage API key: ")
            if not api_key:
                logger.error("Alpha Vantage API key is required")
                return 1
        
        # Create client
        data_source = AlphaVantageClient(api_key=api_key)
    else:
        # Create client
        data_source = YFinanceClient()
    
    # Create importer
    importer = BigQueryImporter(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        table_id=args.table_id
    )
    
    # Import data
    results = importer.import_data(
        data_source=data_source,
        symbols=symbols,
        symbol_names=symbol_names,
        max_workers=args.max_workers,
        period=args.period
    )
    
    # Print results
    success_count = 0
    for symbol, result in results.items():
        if result['success']:
            logger.info(f"{symbol}: {result['message']}")
            success_count += 1
        else:
            logger.error(f"{symbol}: {result['message']}")
    
    # Print summary
    logger.info(f"Import completed: {success_count}/{len(symbols)} symbols imported successfully")
    
    return 0 if success_count == len(symbols) else 1

if __name__ == "__main__":
    sys.exit(import_data())
