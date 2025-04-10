"""
Script to import stock data from Alpha Vantage to BigQuery.
"""
import json
import os
import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google.cloud import bigquery
from src.data.unified_alpha_vantage_client import UnifiedAlphaVantageClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        with open('config/tracked_stocks.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tracked stocks: {str(e)}")
        # Default stocks if config file is not found
        return {
            "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "ORCL", "ASML"],
            "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", "PLTR", "UBER", "LCID", "INTC", "XPEV"]
        }

async def import_stock_data():
    """Import stock data from Alpha Vantage to BigQuery."""
    # Get project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return False
    
    # Initialize Alpha Vantage client
    av_client = UnifiedAlphaVantageClient()
    
    # Initialize BigQuery client
    bq_client = bigquery.Client(project=project_id)
    
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    all_stocks = tracked_stocks["buy"] + tracked_stocks["short"]
    
    # Import data for each stock
    for ticker in all_stocks:
        logger.info(f"Importing data for {ticker}...")
        
        try:
            # Fetch daily data from Alpha Vantage
            df = await av_client.fetch_time_series(ticker, 'daily', incremental=True)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {ticker}")
                continue
            
            # Add symbol column
            df['symbol'] = ticker
            
            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'index': 'date'})
            
            # Rename columns to match BigQuery schema
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adjusted_Close': 'adjusted_close'
            })
            
            # Select only the columns we need
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'symbol']
            df = df[columns]
            
            # Convert date to string (YYYY-MM-DD format)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Upload to BigQuery
            table_id = f"{project_id}.trading_insights.stock_{ticker}_prices"
            
            # Use pandas_gbq to upload data
            df.to_gbq(
                destination_table=f"trading_insights.stock_{ticker}_prices",
                project_id=project_id,
                if_exists='append'
            )
            
            logger.info(f"Successfully imported {len(df)} rows for {ticker}")
            
            # Add a delay to respect Alpha Vantage rate limits
            await asyncio.sleep(12)  # 5 requests per minute = 12 seconds between requests
            
        except Exception as e:
            logger.error(f"Error importing data for {ticker}: {str(e)}")
    
    logger.info("Stock data import completed")
    return True

if __name__ == "__main__":
    asyncio.run(import_stock_data())
