#!/usr/bin/env python3
"""
Unified import module for the Delphi trading intelligence system.

This module provides a unified interface for importing time series data from Alpha Vantage
to BigQuery, with robust error handling, data validation, and reconciliation capabilities.
It leverages the existing trading_ai modules for Alpha Vantage API access and BigQuery storage.
"""
import os
import sys
import logging
import argparse
import time
import json
import platform
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix for Windows event loop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"unified_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import from trading_ai module
from trading_ai.config import config_manager
from trading_ai.core.alpha_client import AlphaVantageClient
from trading_ai.core.bigquery_io import BigQueryStorage

class UnifiedImporter:
    """Unified importer for time series data from Alpha Vantage to BigQuery."""
    
    def __init__(self, alpha_client: Optional[AlphaVantageClient] = None, 
                bigquery_storage: Optional[BigQueryStorage] = None):
        """Initialize the unified importer.
        
        Args:
            alpha_client: Alpha Vantage client. If not provided, a new one will be created.
            bigquery_storage: BigQuery storage. If not provided, a new one will be created.
        """
        # Initialize clients
        self.alpha_client = alpha_client or AlphaVantageClient()
        self.bigquery_storage = bigquery_storage or BigQueryStorage()
        
        # Create directory for status tracking
        self.status_dir = Path("status")
        self.status_dir.mkdir(exist_ok=True)
        
        # Initialize status tracking
        self.import_status = self._load_status()
        
        logger.info("Unified importer initialized")
    
    def _load_status(self) -> Dict[str, Dict[str, Any]]:
        """Load import status from file.
        
        Returns:
            Dictionary with import status for each ticker
        """
        status_file = self.status_dir / "import_status.json"
        if status_file.exists():
            try:
                with open(status_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading status file: {str(e)}")
        
        return {}
    
    def _save_status(self):
        """Save import status to file."""
        status_file = self.status_dir / "import_status.json"
        try:
            with open(status_file, "w") as f:
                json.dump(self.import_status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving status file: {str(e)}")
    
    def _update_ticker_status(self, ticker: str, status: str, details: Optional[Dict[str, Any]] = None):
        """Update status for a ticker.
        
        Args:
            ticker: Stock symbol
            status: Status (success, failed, pending)
            details: Additional details
        """
        self.import_status[ticker] = {
            "status": status,
            "last_update": datetime.now().isoformat(),
            **(details or {})
        }
        self._save_status()
    
    def validate_data(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate data before import.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required columns
        required_columns = ["date", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing")
        
        # Check for null values in key columns
        if not errors:
            for col in required_columns:
                if col in df.columns and df[col].isnull().any():
                    null_count = df[col].isnull().sum()
                    errors.append(f"Column '{col}' contains {null_count} null values")
        
        # Check for data type issues
        if "date" in df.columns:
            if not pd.api.types.is_datetime64_dtype(df["date"]):
                errors.append("Column 'date' is not datetime type")
            
            # Check for negative values in numeric columns
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        errors.append(f"Column '{col}' contains {negative_count} negative values")
        
        return len(errors) == 0, errors
    
    def import_ticker_data(self, ticker: str, force_full: bool = False, 
                          repair_missing: bool = False, days: int = 365) -> bool:
        """Import data for a ticker.
        
        Args:
            ticker: Stock symbol
            force_full: Whether to force a full data refresh
            repair_missing: Whether to check and repair missing dates
            days: Number of days of data to import
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Importing data for {ticker}")
        
        # Update status to pending
        self._update_ticker_status(ticker, "pending")
        
        try:
            # Determine output size based on days and force_full
            outputsize = "full" if force_full or days > 100 else "compact"
            
            # Fetch price data
            logger.info(f"Fetching price data for {ticker} (outputsize={outputsize})")
            price_df = self.alpha_client.fetch_daily(ticker, outputsize=outputsize)
            
            if price_df.empty:
                logger.error(f"No price data found for {ticker}")
                self._update_ticker_status(ticker, "failed", {"error": "No price data found"})
                return False
            
            # Filter to requested date range if needed
            if days < 365 and len(price_df) > days:
                cutoff_date = datetime.now() - timedelta(days=days)
                price_df = price_df[price_df["date"] >= cutoff_date]
            
            # Validate price data
            is_valid, errors = self.validate_data(price_df)
            if not is_valid:
                logger.error(f"Price data validation failed for {ticker}: {errors}")
                self._update_ticker_status(ticker, "failed", {"error": f"Price data validation failed: {errors}"})
                return False
            
            # Store price data in BigQuery
            logger.info(f"Storing {len(price_df)} price records for {ticker}")
            price_success = self.bigquery_storage.store_stock_prices(ticker, price_df)
            
            if not price_success:
                logger.error(f"Failed to store price data for {ticker}")
                self._update_ticker_status(ticker, "failed", {"error": "Failed to store price data"})
                return False
            
            # Update status to success
            self._update_ticker_status(ticker, "success", {
                "price_records": len(price_df),
                "date_range": {
                    "start": price_df["date"].min().isoformat(),
                    "end": price_df["date"].max().isoformat()
                }
            })
            
            logger.info(f"Successfully imported data for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing data for {ticker}: {str(e)}")
            self._update_ticker_status(ticker, "failed", {"error": str(e)})
            return False
    
    def import_tickers(self, tickers: List[str], batch_size: int = 5, 
                      force_full: bool = False, repair_missing: bool = False,
                      days: int = 365, max_workers: int = 3) -> Dict[str, bool]:
        """Import data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            batch_size: Number of tickers to process in each batch
            force_full: Whether to force a full data refresh
            repair_missing: Whether to check and repair missing dates
            days: Number of days of data to import
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping tickers to success status
        """
        logger.info(f"Importing data for {len(tickers)} tickers")
        
        results = {}
        
        # Process tickers in batches to avoid rate limiting
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {', '.join(batch)}")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks
                future_to_ticker = {
                    executor.submit(self.import_ticker_data, ticker, force_full, repair_missing, days): ticker
                    for ticker in batch
                }
                
                # Process results as they complete
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        results[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {str(e)}")
                        results[ticker] = False
            
            # Wait between batches to avoid rate limiting
            if i + batch_size < len(tickers):
                wait_time = 60  # 60 seconds
                logger.info(f"Waiting {wait_time} seconds before processing next batch...")
                time.sleep(wait_time)
        
        # Generate summary
        success_count = sum(1 for status in results.values() if status)
        logger.info(f"Import summary: {success_count}/{len(tickers)} tickers imported successfully")
        
        # List failed tickers
        failed_tickers = [ticker for ticker, status in results.items() if not status]
        if failed_tickers:
            logger.warning(f"Failed tickers: {', '.join(failed_tickers)}")
        
        return results
    
    def get_failed_tickers(self) -> List[str]:
        """Get list of tickers that failed to import.
        
        Returns:
            List of ticker symbols
        """
        return [ticker for ticker, status in self.import_status.items()
                if status.get("status") == "failed"]
    
    def retry_failed_imports(self, batch_size: int = 3, days: int = 365) -> Dict[str, bool]:
        """Retry importing data for tickers that previously failed.
        
        Args:
            batch_size: Number of tickers to process in each batch
            days: Number of days of data to import
            
        Returns:
            Dictionary of ticker to success status
        """
        failed_tickers = self.get_failed_tickers()
        if not failed_tickers:
            logger.info("No failed imports to retry")
            return {}
        
        logger.info(f"Retrying import for {len(failed_tickers)} tickers: {', '.join(failed_tickers)}")
        return self.import_tickers(failed_tickers, batch_size, force_full=True, days=days)
    
    def check_missing_dates(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """Check for missing dates in time series data.
        
        Args:
            ticker: Stock symbol
            days: Number of days to check
            
        Returns:
            Dictionary with missing dates information
        """
        logger.info(f"Checking missing dates for {ticker} (last {days} days)")
        
        try:
            # Get date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get data from BigQuery
            df = self.bigquery_storage.get_stock_prices(ticker, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker} in the specified date range")
                return {
                    "ticker": ticker,
                    "status": "error",
                    "message": "No data found",
                    "missing_dates": [],
                    "missing_count": 0
                }
            
            # Get all trading days in the range
            # This is a simplified approach - in a real system, you would use a proper trading calendar
            all_days = set()
            current_date = start_date
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                    all_days.add(current_date)
                current_date += timedelta(days=1)
            
            # Get days with data
            data_days = set(pd.to_datetime(df["date"]).dt.date)
            
            # Find missing days
            missing_days = all_days - data_days
            
            # Create result
            result = {
                "ticker": ticker,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "status": "success" if not missing_days else "incomplete",
                "total_trading_days": len(all_days),
                "available_days": len(data_days),
                "missing_dates": sorted([d.isoformat() for d in missing_days]),
                "missing_count": len(missing_days)
            }
            
            if missing_days:
                logger.warning(f"Found {len(missing_days)} missing dates for {ticker}")
            else:
                logger.info(f"No missing dates found for {ticker}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking missing dates for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "status": "error",
                "message": str(e),
                "missing_dates": [],
                "missing_count": 0
            }
    
    def repair_missing_data(self, ticker: str, days: int = 90) -> bool:
        """Repair missing data for a ticker.
        
        Args:
            ticker: Stock symbol
            days: Number of days to check
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Repairing missing data for {ticker}")
        
        # Check for missing dates
        missing_info = self.check_missing_dates(ticker, days)
        
        if missing_info["status"] == "error":
            logger.error(f"Error checking missing dates for {ticker}: {missing_info.get('message')}")
            return False
        
        if missing_info["status"] == "success":
            logger.info(f"No missing dates found for {ticker}")
            return True
        
        # If there are missing dates, reimport data
        logger.info(f"Found {missing_info['missing_count']} missing dates for {ticker}, reimporting data")
        return self.import_ticker_data(ticker, force_full=False, repair_missing=True, days=days)
    
    def generate_import_report(self) -> Dict[str, Any]:
        """Generate a report of import status.
        
        Returns:
            Dictionary with import report
        """
        logger.info("Generating import report")
        
        # Count statuses
        status_counts = {
            'success': 0,
            'failed': 0,
            'pending': 0
        }
        
        for ticker, status in self.import_status.items():
            if status.get('status') == 'success':
                status_counts['success'] += 1
            elif status.get('status') == 'failed':
                status_counts['failed'] += 1
            else:
                status_counts['pending'] += 1
        
        # Get latest import date
        latest_import = None
        for ticker, status in self.import_status.items():
            if status.get('last_update'):
                import_date = datetime.fromisoformat(status['last_update'])
                if latest_import is None or import_date > latest_import:
                    latest_import = import_date
        
        # Create report
        report = {
            'generated_at': datetime.now().isoformat(),
            'latest_import': latest_import.isoformat() if latest_import else None,
            'status_counts': status_counts,
            'tickers': {
                ticker: {
                    'status': status.get('status'),
                    'last_update': status.get('last_update'),
                    'error': status.get('error')
                }
                for ticker, status in self.import_status.items()
            }
        }
        
        return report

def setup_environment() -> bool:
    """Set up the environment for import.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if Google Cloud credentials are available
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
            # Check if credentials file exists in default location
            if platform.system() == 'Windows':
                default_creds = Path(os.environ.get('APPDATA', '')) / 'gcloud' / 'application_default_credentials.json'
            else:
                default_creds = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
            
            if default_creds.exists():
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(default_creds)
                logger.info(f"Using default credentials from {default_creds}")
            else:
                logger.warning("Google Cloud credentials not found. Please run 'gcloud auth application-default login'")
                return False
        
        # Check if Alpha Vantage API key is available
        api_key = config_manager.system_config.alpha_vantage_api_key
        if not api_key:
            logger.warning("Alpha Vantage API key not found in configuration")
            return False
        
        # Check if Google Cloud project ID is available
        project_id = config_manager.system_config.google_cloud_project
        if not project_id:
            logger.warning("Google Cloud project ID not found in configuration")
            return False
        
        logger.info("Environment setup successful")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Unified import for time series data")
    
    # Ticker options
    parser.add_argument("--ticker", type=str, help="Import a specific ticker")
    parser.add_argument("--tickers-file", type=str, help="File with tickers to import (one per line)")
    
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

def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        return 1
    
    try:
        # Initialize importer
        importer = UnifiedImporter()
        
        # Generate report if requested
        if args.report:
            report = importer.generate_import_report()
            print(json.dumps(report, indent=2))
            return 0
        
        # Get tickers to import
        tickers = []
        if args.ticker:
            tickers = [args.ticker]
        elif args.tickers_file:
            with open(args.tickers_file, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]
        else:
            # Get all tickers from config
            tickers = config_manager.get_all_tickers()
        
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
