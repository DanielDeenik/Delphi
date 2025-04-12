#!/usr/bin/env python3
"""
Run optimized data import to BigQuery with robust error handling and storage optimization.
"""
import os
import sys
import logging
import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

# Fix for Windows event loop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment for import."""
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

        # Import required modules to check if they're available
        try:
            from trading_ai.config import config_manager
            from trading_ai.core.alpha_client import AlphaVantageClient
            from trading_ai.core.bigquery_io import BigQueryStorage
            from trading_ai.core.volume_footprint import VolumeAnalyzer

            # Check if Alpha Vantage API key is available
            api_key = config_manager.system_config.alpha_vantage_api_key
            if not api_key:
                logger.warning("Alpha Vantage API key not found in configuration")
                return False

            # Check if Google Cloud project is available
            project_id = config_manager.system_config.google_cloud_project
            if not project_id:
                logger.warning("Google Cloud project ID not found in configuration")
                return False

            logger.info("Environment setup successful")
            return True
        except ImportError as e:
            logger.error(f"Required module not found: {str(e)}")
            return False

    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

def run_import(args):
    """Run the import process."""
    try:
        from import_manager import ImportManager

        # Initialize import manager
        import_manager = ImportManager(max_retries=args.retries, retry_delay=args.retry_delay)

        # Get tickers to import
        if args.ticker:
            tickers = [args.ticker]
        else:
            from trading_ai.config import config_manager
            tickers = config_manager.get_all_tickers()

        # Import data
        logger.info(f"Starting import for {len(tickers)} tickers")
        results = import_manager.import_all_tickers(tickers, args.batch_size, args.force_full)

        # Check for failed imports
        failed_tickers = [ticker for ticker, success in results.items() if not success]
        if failed_tickers:
            logger.warning(f"{len(failed_tickers)} tickers failed to import: {', '.join(failed_tickers)}")

            if args.retry_failed:
                logger.info("Retrying failed imports...")
                retry_results = import_manager.retry_failed_imports(args.batch_size)

                # Update results
                results.update(retry_results)

                # Check for still failed imports
                still_failed = [ticker for ticker, success in retry_results.items() if not success]
                if still_failed:
                    logger.warning(f"{len(still_failed)} tickers still failed after retry: {', '.join(still_failed)}")

        # Generate report
        report = import_manager.generate_import_report()
        logger.info(f"Import report: {report['status_counts']['success']}/{report['total_tickers']} tickers imported successfully")

        return len(failed_tickers) == 0

    except Exception as e:
        logger.error(f"Error running import: {str(e)}")
        return False

def run_data_repair(args):
    """Run data repair process."""
    try:
        from data_repair import DataRepairTool

        # Initialize repair tool
        repair_tool = DataRepairTool()

        # Get tickers to repair
        if args.ticker:
            tickers = [args.ticker]
        else:
            from trading_ai.config import config_manager
            tickers = config_manager.get_all_tickers()

        # Repair data
        logger.info(f"Starting data repair for {len(tickers)} tickers")
        results = repair_tool.repair_all_tickers(tickers)

        # Check for failed repairs
        failed_tickers = [ticker for ticker, success in results.items() if not success]
        if failed_tickers:
            logger.warning(f"{len(failed_tickers)} tickers failed to repair: {', '.join(failed_tickers)}")

        return len(failed_tickers) == 0

    except Exception as e:
        logger.error(f"Error running data repair: {str(e)}")
        return False

def run_data_reconciliation(args):
    """Run data reconciliation process."""
    try:
        from data_reconciliation import DataReconciliationTool

        # Initialize reconciliation tool
        reconciliation_tool = DataReconciliationTool()

        # Get tickers to reconcile
        if args.ticker:
            tickers = [args.ticker]
        else:
            from trading_ai.config import config_manager
            tickers = config_manager.get_all_tickers()

        # Generate reconciliation report
        logger.info(f"Generating reconciliation report for {len(tickers)} tickers")
        report = reconciliation_tool.generate_reconciliation_report(tickers)

        # Log summary
        logger.info(f"Reconciliation summary:")
        logger.info(f"  Complete: {report['summary']['complete']}")
        logger.info(f"  Incomplete: {report['summary']['incomplete']}")
        logger.info(f"  Error: {report['summary']['error']}")
        logger.info(f"  Missing dates total: {report['summary']['missing_dates_total']}")
        logger.info(f"  Quality issues total: {report['summary']['quality_issues_total']}")

        # Get tickers needing reimport
        tickers_to_reimport = reconciliation_tool.get_tickers_needing_reimport(report)

        if tickers_to_reimport:
            logger.warning(f"{len(tickers_to_reimport)} tickers need reimport: {', '.join(tickers_to_reimport)}")

            # Save list of tickers to reimport
            reimport_file = Path("reimport_tickers.txt")
            with open(reimport_file, 'w') as f:
                f.write('\n'.join(tickers_to_reimport))

            logger.info(f"Saved list of tickers to reimport to {reimport_file}")

            # Reimport if requested
            if args.reimport_missing:
                logger.info(f"Reimporting {len(tickers_to_reimport)} tickers with missing data")
                from import_manager import ImportManager
                import_manager = ImportManager()
                import_manager.reimport_tickers_with_missing_data(tickers_to_reimport)
        else:
            logger.info("No tickers need reimport")

        return True

    except Exception as e:
        logger.error(f"Error running data reconciliation: {str(e)}")
        return False

def run_storage_optimization(args):
    """Run storage optimization process."""
    try:
        from storage_optimizer import StorageOptimizer

        # Initialize optimizer
        optimizer = StorageOptimizer()

        # Get storage usage before optimization
        before_usage = optimizer.get_storage_usage()
        logger.info(f"Storage usage before optimization: {before_usage['total_size_mb']:.2f} MB")

        # Optimize tables
        if args.optimize_schema:
            logger.info("Optimizing table schemas...")
            optimizer.optimize_all_tables()

        # Compress historical data
        if args.compress_history:
            logger.info(f"Compressing historical data older than {args.compress_days} days...")
            optimizer.compress_all_historical_data(args.compress_days)

        # Get storage usage after optimization
        after_usage = optimizer.get_storage_usage()
        logger.info(f"Storage usage after optimization: {after_usage['total_size_mb']:.2f} MB")

        # Calculate savings
        savings = before_usage['total_size_mb'] - after_usage['total_size_mb']
        savings_percent = (savings / before_usage['total_size_mb'] * 100) if before_usage['total_size_mb'] > 0 else 0
        logger.info(f"Storage savings: {savings:.2f} MB ({savings_percent:.2f}%)")

        return True

    except Exception as e:
        logger.error(f"Error running storage optimization: {str(e)}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run optimized data import to BigQuery")

    # Import options
    parser.add_argument("--ticker", type=str, help="Import a specific ticker")
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of stocks to process in each batch")
    parser.add_argument("--retries", type=int, default=3, help="Maximum number of retries for failed imports")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries in seconds")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed imports")

    # Data repair options
    parser.add_argument("--repair", action="store_true", help="Repair data inconsistencies")
    parser.add_argument("--reconcile", action="store_true", help="Reconcile data and check for missing dates")
    parser.add_argument("--reimport-missing", action="store_true", help="Reimport data for tickers with missing dates")

    # Storage optimization options
    parser.add_argument("--optimize", action="store_true", help="Optimize storage")
    parser.add_argument("--optimize-schema", action="store_true", help="Optimize table schemas")
    parser.add_argument("--compress-history", action="store_true", help="Compress historical data")
    parser.add_argument("--compress-days", type=int, default=365, help="Compress data older than this many days")

    # Parse arguments
    args = parser.parse_args()

    # Set default actions if none specified
    if not any([args.ticker, args.force_full, args.retry_failed, args.repair, args.reconcile, args.reimport_missing,
                args.optimize, args.optimize_schema, args.compress_history]):
        args.force_full = False
        args.retry_failed = True
        args.repair = True
        args.reconcile = True
        args.optimize_schema = True
        args.compress_history = True

    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        return 1

    # Run import
    if args.ticker or args.force_full or args.retry_failed or not any([args.repair, args.optimize,
                                                                      args.optimize_schema, args.compress_history]):
        logger.info("Running data import...")
        if not run_import(args):
            logger.error("Data import failed")
            return 1

    # Run data repair
    if args.repair:
        logger.info("Running data repair...")
        if not run_data_repair(args):
            logger.warning("Data repair had some issues")

    # Run data reconciliation
    if args.reconcile:
        logger.info("Running data reconciliation...")
        if not run_data_reconciliation(args):
            logger.warning("Data reconciliation had some issues")

    # Run storage optimization
    if args.optimize or args.optimize_schema or args.compress_history:
        logger.info("Running storage optimization...")
        if not run_storage_optimization(args):
            logger.warning("Storage optimization had some issues")

    logger.info("All processes completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
