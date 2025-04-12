"""
Direct import script for BigQuery using the trading_ai codebase.
"""
import logging
import platform
import sys
import time
import pandas as pd
from datetime import datetime

# Fix for Windows event loop
if platform.system() == 'Windows':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def import_to_bigquery(batch_size=3, force_full=False):
    """Import data to BigQuery directly."""
    try:
        # Import required modules
        from trading_ai.config import config_manager
        from trading_ai.core.alpha_client import AlphaVantageClient
        from trading_ai.core.bigquery_io import BigQueryStorage
        from trading_ai.core.volume_footprint import VolumeAnalyzer

        # Initialize BigQuery storage
        logger.info("Initializing BigQuery storage...")
        bigquery_storage = BigQueryStorage()

        # Initialize tables
        logger.info("Initializing BigQuery tables...")
        bigquery_storage.initialize_tables()

        # Initialize clients
        logger.info("Initializing Alpha Vantage client and Volume Analyzer...")
        alpha_client = AlphaVantageClient()
        volume_analyzer = VolumeAnalyzer()

        # Get all tracked tickers
        all_tickers = config_manager.get_all_tickers()
        logger.info(f"Found {len(all_tickers)} tracked tickers: {', '.join(all_tickers)}")

        # Split tickers into batches
        batches = [all_tickers[i:i+batch_size] for i in range(0, len(all_tickers), batch_size)]
        logger.info(f"Processing {len(batches)} batches of {batch_size} tickers each")

        # Process each batch
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {', '.join(batch)}")

            # Process each ticker in the batch
            for ticker in batch:
                # Determine output size based on existing data
                outputsize = 'full' if force_full else 'compact'

                # Fetch daily data (using synchronous method)
                logger.info(f"Fetching {outputsize} data for {ticker}...")
                df = alpha_client.fetch_daily_data_sync(ticker, outputsize)

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                # Store in BigQuery
                logger.info(f"Storing data for {ticker} in BigQuery...")
                success = bigquery_storage.store_stock_prices(ticker, df)

                if success:
                    logger.info(f"Successfully stored data for {ticker}")

                    # Process volume analysis
                    logger.info(f"Processing volume analysis for {ticker}...")

                    # Calculate volume metrics
                    analysis_df = volume_analyzer.calculate_volume_metrics(df)

                    # Detect volume inefficiencies
                    analysis_df = volume_analyzer.detect_volume_inefficiencies(analysis_df)

                    # Store volume analysis
                    logger.info(f"Storing volume analysis for {ticker}...")
                    bigquery_storage.store_volume_analysis(ticker, analysis_df)
                else:
                    logger.warning(f"Failed to store data for {ticker}")

            # Wait between batches (except for the last batch)
            if i < len(batches) - 1:
                logger.info("Waiting 60 seconds before processing next batch...")
                time.sleep(60)

        # Generate master summary
        logger.info("Generating master summary...")
        # Get summaries for each ticker
        summaries = []
        for ticker in all_tickers:
            # Get volume analysis
            analysis_df = bigquery_storage.get_volume_analysis(ticker, days=30)

            if not analysis_df.empty:
                # Get latest data point
                latest = analysis_df.iloc[0]  # Assuming sorted by date desc

                # Create summary entry
                summary_entry = {
                    'date': latest['date'],
                    'symbol': ticker,
                    'direction': config_manager.get_ticker_direction(ticker),
                    'close': latest.get('close', 0),
                    'volume': latest.get('volume', 0),
                    'relative_volume': latest.get('relative_volume_20d', 0),
                    'volume_z_score': latest.get('volume_z_score', 0),
                    'is_volume_spike': latest.get('is_volume_spike', False),
                    'spike_strength': latest.get('spike_strength', 0),
                    'price_change_pct': latest.get('price_change_pct', 0),
                    'signal': latest.get('signal', 'NEUTRAL'),
                    'confidence': latest.get('signal_strength', 0),
                    'notes': latest.get('notes', ''),
                    'notebook_url': f"https://colab.research.google.com/drive/your-notebook-id-for-{ticker}",
                    'timestamp': datetime.now()
                }

                summaries.append(summary_entry)

        # Create DataFrame
        summary_df = pd.DataFrame(summaries)

        # Store in BigQuery
        if not summary_df.empty:
            bigquery_storage.store_master_summary(summary_df)
            logger.info(f"Successfully stored master summary for {len(summary_df)} stocks")
        else:
            logger.warning("No data for master summary")

        logger.info("Data import completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error importing data to BigQuery: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting direct import to BigQuery...")
    success = import_to_bigquery(batch_size=3, force_full=False)

    if success:
        print("\nData import completed successfully!")
    else:
        print("\nData import failed. Check the logs for details.")
