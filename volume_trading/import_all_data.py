"""
Script to import all data for tracked stocks.
"""
import logging
from datetime import datetime
import asyncio
import argparse

from volume_trading.config import config
from volume_trading.core.batch_downloader import BatchDownloader
from volume_trading.core.volume_analyzer import VolumeAnalyzer
from volume_trading.core.data_storage import DataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"data_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def import_all_data(force_refresh=True, batch_size=5, delay_between_batches=60, use_async=False, tickers=None):
    """Import all data for tracked stocks.

    Args:
        force_refresh: Whether to force a full refresh of data
        batch_size: Number of stocks to process in each batch
        delay_between_batches: Delay in seconds between batches
        use_async: Whether to use asynchronous downloading
        tickers: List of specific tickers to process (None for all tracked tickers)
    """
    logger.info("Starting data import for all tracked stocks...")

    # Get tickers to process
    if tickers is None:
        tickers = config.get_all_tickers()
    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")

    # Initialize clients
    downloader = BatchDownloader(batch_size=batch_size, delay_between_batches=delay_between_batches)
    storage = DataStorage()
    analyzer = VolumeAnalyzer()

    # Download data in batches
    logger.info(f"Downloading data in batches of {batch_size}...")
    if use_async:
        # Use asyncio to run the async download function
        download_results = asyncio.run(downloader.download_all_stocks_async(tickers, force_refresh))
    else:
        download_results = downloader.download_all_stocks(tickers, force_refresh)

    # Process downloaded data
    logger.info("Processing downloaded data...")
    success_count = 0
    summaries = []

    for ticker, success in download_results.items():
        if success:
            try:
                # Load price data
                df = storage.load_price_data(ticker)

                if df.empty:
                    logger.warning(f"No data loaded for {ticker}")
                    continue

                # Analyze data
                logger.info(f"Analyzing data for {ticker}...")
                analysis_df = analyzer.analyze(df)

                # Save analysis results
                logger.info(f"Saving analysis results for {ticker}...")
                storage.save_analysis_results(ticker, analysis_df)

                # Get and save summary
                logger.info(f"Generating summary for {ticker}...")
                summary = analyzer.get_summary(analysis_df)
                storage.save_summary(ticker, summary)

                summaries.append(summary)
                success_count += 1
                logger.info(f"Completed processing for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")

    # Generate master summary
    logger.info("Generating master summary...")
    storage.save_master_summary(summaries)

    logger.info(f"Data import completed for {success_count}/{len(tickers)} tickers")

    # Print summary
    print("\nImport Summary:")
    print(f"Processed {success_count}/{len(tickers)} tickers")
    print("\nTop Signals:")

    # Sort summaries by signal strength
    sorted_summaries = sorted(summaries, key=lambda x: x.get("signal_strength", 0), reverse=True)

    # Print top signals
    for summary in sorted_summaries[:5]:
        if summary.get("latest_signal") != "NEUTRAL":
            print(f"  {summary.get('ticker')}: {summary.get('latest_signal')} (Strength: {summary.get('signal_strength', 0):.2f})")
            print(f"    {summary.get('notes', '')}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Import data for tracked stocks")
    parser.add_argument("--force-refresh", action="store_true", help="Force a full refresh of data")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of stocks to process in each batch")
    parser.add_argument("--delay", type=int, default=60, help="Delay in seconds between batches")
    parser.add_argument("--async", action="store_true", dest="use_async", help="Use asynchronous downloading")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to process (space-separated)")

    args = parser.parse_args()

    # Import data with specified options
    import_all_data(
        force_refresh=args.force_refresh,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        use_async=args.use_async,
        tickers=args.tickers
    )
