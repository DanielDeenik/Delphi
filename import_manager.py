"""
Import Manager for handling data imports to BigQuery with robust error handling and retry logic.
"""
import os
import logging
import time
import json
import hashlib
import platform
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Fix for Windows event loop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("import_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImportManager:
    """Manager for handling data imports with robust error handling and retry logic."""

    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """Initialize the import manager.

        Args:
            max_retries: Maximum number of retries for failed imports
            retry_delay: Delay between retries in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.import_status = {}
        self.status_file = Path("import_status.json")
        self._load_status()

        # Create status directory if it doesn't exist
        os.makedirs("status", exist_ok=True)

    def _load_status(self):
        """Load import status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.import_status = json.load(f)
                logger.info(f"Loaded import status for {len(self.import_status)} tickers")
            except Exception as e:
                logger.error(f"Error loading import status: {str(e)}")
                self.import_status = {}

    def _save_status(self):
        """Save import status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.import_status, f, indent=2)
            logger.info(f"Saved import status for {len(self.import_status)} tickers")
        except Exception as e:
            logger.error(f"Error saving import status: {str(e)}")

    def _get_ticker_status_file(self, ticker: str) -> Path:
        """Get the status file path for a ticker."""
        return Path(f"status/{ticker.lower()}_status.json")

    def _save_ticker_status(self, ticker: str, status: Dict):
        """Save status for a specific ticker."""
        try:
            status_file = self._get_ticker_status_file(ticker)
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status for {ticker}: {str(e)}")

    def _load_ticker_status(self, ticker: str) -> Dict:
        """Load status for a specific ticker."""
        status_file = self._get_ticker_status_file(ticker)
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading status for {ticker}: {str(e)}")
        return {}

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the DataFrame to detect changes."""
        if df.empty:
            return ""
        # Convert to string and hash
        df_str = df.to_json()
        return hashlib.md5(df_str.encode()).hexdigest()

    def _validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
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
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")

        # Check for null values in critical columns
        for col in ['date', 'close', 'volume']:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                errors.append(f"Column '{col}' contains {null_count} null values")

        # Check for duplicate dates
        if 'date' in df.columns:
            duplicate_dates = df['date'].duplicated().sum()
            if duplicate_dates > 0:
                errors.append(f"Found {duplicate_dates} duplicate dates")

        # Check for future dates
        if 'date' in df.columns:
            today = pd.Timestamp.now().normalize()
            future_dates = (df['date'] > today).sum()
            if future_dates > 0:
                errors.append(f"Found {future_dates} future dates")

        # Check for negative values in numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and (df[col] < 0).any():
                negative_count = (df[col] < 0).sum()
                errors.append(f"Column '{col}' contains {negative_count} negative values")

        return len(errors) == 0, errors

    def import_ticker_data(self, ticker: str, force_full: bool = False, repair_missing: bool = False) -> bool:
        """Import data for a ticker with retry logic.

        Args:
            ticker: Stock symbol
            force_full: Whether to force a full data refresh
            repair_missing: Whether to check and repair missing dates

        Returns:
            True if successful, False otherwise
        """
        # Import required modules
        from trading_ai.config import config_manager
        from trading_ai.core.alpha_client import AlphaVantageClient
        from trading_ai.core.bigquery_io import BigQueryStorage
        from trading_ai.core.volume_footprint import VolumeAnalyzer

        # Initialize clients
        alpha_client = AlphaVantageClient()
        bigquery_storage = BigQueryStorage()
        volume_analyzer = VolumeAnalyzer()

        # Initialize status tracking
        self._ensure_ticker_status(ticker)

        # Load ticker status
        ticker_status = self._load_ticker_status(ticker)
        last_import_date = ticker_status.get('last_import_date')
        last_data_hash = ticker_status.get('data_hash', '')

        # Determine if we need a full import
        need_full_import = force_full or not last_import_date
        outputsize = 'full' if need_full_import else 'compact'

        logger.info(f"Importing data for {ticker} (outputsize={outputsize})")

        # Try to import with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                # Fetch data
                logger.info(f"Attempt {attempt}/{self.max_retries}: Fetching {outputsize} data for {ticker}")
                df = alpha_client.fetch_daily_data_sync(ticker, outputsize)

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")

                    # Update status
                    ticker_status.update({
                        'last_attempt_date': datetime.now().isoformat(),
                        'last_attempt_status': 'failed',
                        'last_error': 'No data returned'
                    })
                    self._save_ticker_status(ticker, ticker_status)

                    # Try again with full data if we were using compact
                    if outputsize == 'compact' and attempt == self.max_retries:
                        logger.info(f"Retrying with full data for {ticker}")
                        return self.import_ticker_data(ticker, force_full=True)

                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    return False

                # Validate data
                is_valid, errors = self._validate_data(df)
                if not is_valid:
                    logger.warning(f"Data validation failed for {ticker}: {', '.join(errors)}")

                    # Update status
                    ticker_status.update({
                        'last_attempt_date': datetime.now().isoformat(),
                        'last_attempt_status': 'failed',
                        'last_error': f"Validation failed: {', '.join(errors)}"
                    })
                    self._save_ticker_status(ticker, ticker_status)

                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    return False

                # Compute data hash to check for changes
                data_hash = self._compute_data_hash(df)
                if data_hash == last_data_hash and not force_full:
                    logger.info(f"No new data for {ticker} since last import")

                    # Update status
                    ticker_status.update({
                        'last_attempt_date': datetime.now().isoformat(),
                        'last_attempt_status': 'skipped',
                        'last_error': None,
                        'last_check_date': datetime.now().isoformat()
                    })
                    self._save_ticker_status(ticker, ticker_status)

                    # Still return success
                    return True

                # Store in BigQuery
                logger.info(f"Storing data for {ticker} in BigQuery...")
                success = bigquery_storage.store_stock_prices(ticker, df)

                if not success:
                    logger.warning(f"Failed to store data for {ticker}")

                    # Update status
                    ticker_status.update({
                        'last_attempt_date': datetime.now().isoformat(),
                        'last_attempt_status': 'failed',
                        'last_error': 'Failed to store data in BigQuery'
                    })
                    self._save_ticker_status(ticker, ticker_status)

                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    return False

                # Process volume analysis
                logger.info(f"Processing volume analysis for {ticker}...")

                try:
                    # Calculate volume metrics
                    analysis_df = volume_analyzer.calculate_volume_metrics(df)

                    # Detect volume inefficiencies
                    analysis_df = volume_analyzer.detect_volume_inefficiencies(analysis_df)

                    # Store volume analysis
                    logger.info(f"Storing volume analysis for {ticker}...")
                    analysis_success = bigquery_storage.store_volume_analysis(ticker, analysis_df)

                    if not analysis_success:
                        logger.warning(f"Failed to store volume analysis for {ticker}")
                except Exception as e:
                    logger.error(f"Error processing volume analysis for {ticker}: {str(e)}")
                    analysis_success = False

                # Update status
                ticker_status.update({
                    'last_import_date': datetime.now().isoformat(),
                    'last_attempt_date': datetime.now().isoformat(),
                    'last_attempt_status': 'success',
                    'last_error': None,
                    'data_hash': data_hash,
                    'row_count': len(df),
                    'date_range': {
                        'start': df['date'].min().isoformat() if isinstance(df['date'].min(), pd.Timestamp) else str(df['date'].min()),
                        'end': df['date'].max().isoformat() if isinstance(df['date'].max(), pd.Timestamp) else str(df['date'].max())
                    },
                    'volume_analysis_success': analysis_success
                })
                self._save_ticker_status(ticker, ticker_status)

                # Update global status
                self.import_status[ticker] = {
                    'last_import_date': datetime.now().isoformat(),
                    'status': 'success',
                    'row_count': len(df)
                }
                self._save_status()

                logger.info(f"Successfully imported data for {ticker}")
                return True

            except Exception as e:
                logger.error(f"Error importing data for {ticker} (attempt {attempt}/{self.max_retries}): {str(e)}")

                # Update status
                ticker_status.update({
                    'last_attempt_date': datetime.now().isoformat(),
                    'last_attempt_status': 'failed',
                    'last_error': str(e)
                })
                self._save_ticker_status(ticker, ticker_status)

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    # Update global status
                    self.import_status[ticker] = {
                        'last_attempt_date': datetime.now().isoformat(),
                        'status': 'failed',
                        'error': str(e)
                    }
                    self._save_status()
                    return False

        return False

    def import_all_tickers(self, tickers: List[str], batch_size: int = 3, force_full: bool = False) -> Dict[str, bool]:
        """Import data for multiple tickers in batches.

        Args:
            tickers: List of stock symbols
            batch_size: Number of tickers to process in each batch
            force_full: Whether to force a full data refresh

        Returns:
            Dictionary of ticker to success status
        """
        results = {}

        # Split tickers into batches
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        logger.info(f"Processing {len(batches)} batches of {batch_size} tickers each")

        # Process each batch
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {', '.join(batch)}")

            # Process each ticker in the batch
            for ticker in batch:
                success = self.import_ticker_data(ticker, force_full)
                results[ticker] = success

            # Wait between batches (except for the last batch)
            if i < len(batches) - 1:
                logger.info(f"Waiting 60 seconds before processing next batch...")
                time.sleep(60)

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
                if status.get('status') == 'failed']

    def retry_failed_imports(self, batch_size: int = 3) -> Dict[str, bool]:
        """Retry importing data for tickers that previously failed.

        Args:
            batch_size: Number of tickers to process in each batch

        Returns:
            Dictionary of ticker to success status
        """
        failed_tickers = self.get_failed_tickers()
        if not failed_tickers:
            logger.info("No failed imports to retry")
            return {}

        logger.info(f"Retrying import for {len(failed_tickers)} tickers: {', '.join(failed_tickers)}")
        return self.import_all_tickers(failed_tickers, batch_size, force_full=True)

    def reimport_tickers_with_missing_data(self, tickers: List[str] = None, batch_size: int = 5) -> Dict[str, bool]:
        """Reimport data for tickers with missing dates.

        Args:
            tickers: List of tickers to check (default: all tickers with status)
            batch_size: Number of tickers to process in each batch

        Returns:
            Dictionary mapping tickers to success status
        """
        # Import data reconciliation tool
        try:
            from data_reconciliation import DataReconciliationTool
        except ImportError:
            logger.error("Could not import DataReconciliationTool")
            return {}

        # Initialize reconciliation tool
        try:
            reconciliation_tool = DataReconciliationTool()
        except Exception as e:
            logger.error(f"Error initializing DataReconciliationTool: {str(e)}")
            return {}

        # Get tickers to check
        if tickers is None:
            tickers = list(self.import_status.keys())

        if not tickers:
            logger.info("No tickers to check for missing data")
            return {}

        logger.info(f"Checking {len(tickers)} tickers for missing data")

        # Check each ticker for missing dates
        tickers_to_reimport = []
        for ticker in tickers:
            try:
                # Check for missing dates
                result = reconciliation_tool.check_missing_dates(ticker)

                # Check if ticker needs reimport
                if result["status"] == "incomplete" and result.get("missing_count", 0) > 0:
                    tickers_to_reimport.append(ticker)
                    logger.info(f"{ticker} needs reimport due to {result['missing_count']} missing dates")
            except Exception as e:
                logger.error(f"Error checking missing dates for {ticker}: {str(e)}")

        if not tickers_to_reimport:
            logger.info("No tickers need reimport for missing data")
            return {}

        logger.info(f"Reimporting {len(tickers_to_reimport)} tickers with missing data")

        # Import tickers in batches
        return self.import_all_tickers(tickers_to_reimport, batch_size, force_full=True)

    def generate_import_report(self) -> Dict:
        """Generate a report of import status.

        Returns:
            Dictionary with import statistics
        """
        # Count by status
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
            if status.get('last_import_date'):
                import_date = datetime.fromisoformat(status['last_import_date'])
                if latest_import is None or import_date > latest_import:
                    latest_import = import_date

        # Generate report
        report = {
            'total_tickers': len(self.import_status),
            'status_counts': status_counts,
            'latest_import_date': latest_import.isoformat() if latest_import else None,
            'failed_tickers': self.get_failed_tickers()
        }

        return report

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Import stock data to BigQuery with robust error handling")
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of stocks to process in each batch")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed imports")
    parser.add_argument("--report", action="store_true", help="Generate import report")
    parser.add_argument("--ticker", type=str, help="Import a specific ticker")
    args = parser.parse_args()

    # Initialize import manager
    import_manager = ImportManager()

    # Import data
    if args.retry_failed:
        import_manager.retry_failed_imports(args.batch_size)
    elif args.report:
        report = import_manager.generate_import_report()
        print(json.dumps(report, indent=2))
    elif args.ticker:
        import_manager.import_ticker_data(args.ticker, args.force_full)
    else:
        # Import all tickers
        from trading_ai.config import config_manager
        all_tickers = config_manager.get_all_tickers()
        import_manager.import_all_tickers(all_tickers, args.batch_size, args.force_full)

if __name__ == "__main__":
    main()
