"""
Script to run the entire Volume Trading System process.
"""
import logging
import argparse
from datetime import datetime

from volume_trading.import_all_data import import_all_data
from volume_trading.generate_notebooks import generate_notebooks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"volume_trading_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def run_all(force_refresh=False, batch_size=5, delay_between_batches=60, use_async=False, tickers=None):
    """Run the entire process - import data and generate notebooks.

    Args:
        force_refresh: Whether to force a full refresh of data
        batch_size: Number of stocks to process in each batch
        delay_between_batches: Delay in seconds between batches
        use_async: Whether to use asynchronous downloading
        tickers: List of specific tickers to process (None for all tracked tickers)
    """
    logger.info("Starting Volume Trading System process...")

    # Step 1: Import data
    logger.info("Step 1: Importing data...")
    import_all_data(
        force_refresh=force_refresh,
        batch_size=batch_size,
        delay_between_batches=delay_between_batches,
        use_async=use_async,
        tickers=tickers
    )

    # Step 2: Generate notebooks
    logger.info("Step 2: Generating notebooks...")
    generate_notebooks()

    logger.info("Process completed successfully")

    # Print final instructions
    print("\nVolume Trading System is ready!")
    print("To open the master dashboard, run:")
    print("jupyter notebook notebooks/master_dashboard.ipynb")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the Volume Trading System process")
    parser.add_argument("--force-refresh", action="store_true", help="Force a full refresh of data")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of stocks to process in each batch")
    parser.add_argument("--delay", type=int, default=60, help="Delay in seconds between batches")
    parser.add_argument("--async", action="store_true", dest="use_async", help="Use asynchronous downloading")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to process (space-separated)")
    args = parser.parse_args()

    # Run the process
    run_all(
        force_refresh=args.force_refresh,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        use_async=args.use_async,
        tickers=args.tickers
    )
