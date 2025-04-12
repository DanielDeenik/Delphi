#!/usr/bin/env python
"""
Migration script for importing data from Alpha Vantage directly to BigQuery.

This script helps import historical market data from Alpha Vantage to BigQuery
for scalable time series storage and analysis.

Usage:
    python migrate_to_bigquery.py --symbols AAPL,MSFT,GOOG
"""

import os
import sys
import argparse
import logging
import pandas as pd
from google.cloud import bigquery

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from trading_ai module
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Migrate data from SQLite to BigQuery')
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols to migrate')
    parser.add_argument('--project-id', help='Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)')
    parser.add_argument('--dataset', default='market_data', help='BigQuery dataset name (default: market_data)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of data to migrate (default: 365)')
    return parser.parse_args()

def ensure_bigquery_setup(project_id, dataset_id):
    """Ensure BigQuery dataset exists."""
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)

    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_id} already exists")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        client.create_dataset(dataset)
        logger.info(f"Created BigQuery dataset: {dataset_id}")

def migrate_symbol(symbol, alpha_client, bigquery_storage, days):
    """Import data for a symbol from Alpha Vantage to BigQuery."""
    try:
        # Get data from Alpha Vantage
        logger.info(f"Fetching data for {symbol} from Alpha Vantage")

        # Determine output size based on days
        outputsize = 'full' if days > 100 else 'compact'

        # Fetch data
        df = alpha_client.fetch_daily(symbol, outputsize=outputsize)

        if df.empty:
            logger.warning(f"No data found for {symbol} from Alpha Vantage")
            return False

        # Filter to requested date range if needed
        if days < 365 and len(df) > days:
            df = df.iloc[:days]

        logger.info(f"Retrieved {len(df)} rows for {symbol}")

        # Prepare data for BigQuery
        if 'date' in df.columns:
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
        else:
            logger.warning(f"No date column found in data for {symbol}")
            return False

        # Add symbol column if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol

        # Upload to BigQuery using the storage service
        success = bigquery_storage.store_stock_prices(symbol, df)

        if success:
            logger.info(f"Successfully imported {len(df)} rows for {symbol} to BigQuery")
            return True
        else:
            logger.error(f"Failed to import data for {symbol} to BigQuery")
            return False

    except Exception as e:
        logger.error(f"Error migrating {symbol}: {str(e)}")
        return False

def main():
    """Main entry point."""
    args = parse_args()

    # Get project ID from args or environment
    project_id = args.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        logger.error("Google Cloud project ID not specified. Use --project-id or set GOOGLE_CLOUD_PROJECT environment variable")
        sys.exit(1)

    # Initialize services
    alpha_client = AlphaVantageClient(api_key=os.getenv('ALPHA_VANTAGE_API_KEY'))
    bigquery_storage = BigQueryStorage(project_id=project_id, dataset_id=args.dataset)

    # Ensure BigQuery dataset exists
    ensure_bigquery_setup(project_id, args.dataset)

    # Process each symbol
    symbols = [s.strip() for s in args.symbols.split(',')]
    logger.info(f"Importing data for {len(symbols)} symbols: {', '.join(symbols)}")

    success_count = 0
    for symbol in symbols:
        if migrate_symbol(symbol, alpha_client, bigquery_storage, args.days):
            success_count += 1

    logger.info(f"Import completed. Successfully imported {success_count}/{len(symbols)} symbols")

if __name__ == "__main__":
    main()
