#!/usr/bin/env python
"""
Migration script for moving data from SQLite to BigQuery.

This script helps migrate time series data from SQLite to BigQuery
when you're ready to transition to a more scalable solution.

Usage:
    python migrate_to_bigquery.py --symbols AAPL,MSFT,GOOG
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from pandas_gbq import to_gbq

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.sqlite_storage_service import SQLiteStorageService

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

def migrate_symbol(symbol, sqlite_service, project_id, dataset_id, days):
    """Migrate data for a single symbol from SQLite to BigQuery."""
    try:
        # Get data from SQLite
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
        df = sqlite_service.get_market_data(symbol, start_date, end_date)

        if df.empty:
            logger.warning(f"No data found for {symbol} in SQLite")
            return False

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

        # Export to BigQuery
        table_id = f"{symbol.lower()}_daily"
        logger.info(f"Exporting {len(df)} rows to BigQuery table {project_id}.{dataset_id}.{table_id}")

        to_gbq(
            df,
            f"{dataset_id}.{table_id}",
            project_id=project_id,
            if_exists='append'
        )

        logger.info(f"Successfully migrated {len(df)} rows for {symbol} to BigQuery")
        return True

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
    sqlite_service = SQLiteStorageService()

    # Ensure BigQuery dataset exists
    ensure_bigquery_setup(project_id, args.dataset)

    # Process each symbol
    symbols = [s.strip() for s in args.symbols.split(',')]
    logger.info(f"Migrating data for {len(symbols)} symbols: {', '.join(symbols)}")

    success_count = 0
    for symbol in symbols:
        if migrate_symbol(symbol, sqlite_service, project_id, args.dataset, args.days):
            success_count += 1

    logger.info(f"Migration completed. Successfully migrated {success_count}/{len(symbols)} symbols")

if __name__ == "__main__":
    main()
