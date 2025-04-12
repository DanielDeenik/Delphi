"""
Script to import stock data from Alpha Vantage to BigQuery.
This script can be scheduled as a cron job to run daily.

Example cron job (runs daily at 6:00 AM):
0 6 * * * python /path/to/scheduled_data_import.py

Required environment variables:
- GOOGLE_CLOUD_PROJECT: Google Cloud project ID
- ALPHA_VANTAGE_API_KEY: Alpha Vantage API key
"""
import json
import os
import logging
import asyncio
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        config_path = os.getenv("CONFIG_PATH", "config/tracked_stocks.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tracked stocks: {str(e)}")
        # Default stocks if config file is not found
        return {
            "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "ORCL", "ASML"],
            "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", "PLTR", "UBER", "LCID", "INTC", "XPEV"]
        }

def archive_old_ticker_data(ticker, project_id, dataset="trading_insights"):
    """Archive data for a ticker that is no longer tracked."""
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        
        # Check if the table exists
        table_id = f"{project_id}.{dataset}.stock_{ticker}_prices"
        try:
            client.get_table(table_id)
        except Exception:
            logger.info(f"Table {table_id} does not exist, no need to archive")
            return True
        
        # Create archive table if it doesn't exist
        archive_table_id = f"{project_id}.{dataset}.archived_stock_{ticker}_prices"
        
        # Copy data to archive table
        job_config = bigquery.QueryJobConfig()
        job_config.destination = archive_table_id
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        
        sql = f"""
        SELECT * FROM `{table_id}`
        """
        
        query_job = client.query(sql, job_config=job_config)
        query_job.result()  # Wait for the job to complete
        
        logger.info(f"Archived data for {ticker} to {archive_table_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error archiving data for {ticker}: {str(e)}")
        return False

def import_stock_data():
    """Import stock data from Alpha Vantage to BigQuery."""
    # Get project ID and API key from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        return False
    
    if not api_key:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set")
        return False
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    all_stocks = []
    for direction in tracked_stocks:
        all_stocks.extend(tracked_stocks[direction])
    
    # Check for stocks that are no longer tracked
    dataset_id = "trading_insights"
    
    # List existing tables
    tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
    existing_stock_tables = [table.table_id for table in tables if table.table_id.startswith("stock_") and table.table_id.endswith("_prices")]
    
    # Extract tickers from table names
    existing_tickers = [table_id.replace("stock_", "").replace("_prices", "") for table_id in existing_stock_tables]
    
    # Find tickers that are no longer tracked
    removed_tickers = [ticker for ticker in existing_tickers if ticker not in all_stocks]
    
    # Archive data for removed tickers
    for ticker in removed_tickers:
        logger.info(f"Archiving data for removed ticker: {ticker}")
        archive_old_ticker_data(ticker, project_id, dataset_id)
    
    # Import data for each stock
    for i, ticker in enumerate(all_stocks):
        logger.info(f"Importing data for {ticker} ({i+1}/{len(all_stocks)})...")
        
        try:
            # Fetch daily data from Alpha Vantage
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={api_key}"
            response = requests.get(url)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.warning(f"Error fetching data for {ticker}: {data.get('Note', 'Unknown error')}")
                # Wait for API rate limit
                time.sleep(12)  # 5 requests per minute = 12 seconds between requests
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            
            # Map column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split_coefficient'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add symbol column
            df['symbol'] = ticker
            
            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'index': 'date'})
            
            # Select only the columns we need
            columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close', 'symbol']
            df = df[columns]
            
            # Convert date to string (YYYY-MM-DD format)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Upload to BigQuery
            table_id = f"{dataset_id}.stock_{ticker}_prices"
            
            # Use pandas_gbq to upload data
            df.to_gbq(
                destination_table=table_id,
                project_id=project_id,
                if_exists='replace'
            )
            
            logger.info(f"Successfully imported {len(df)} rows for {ticker}")
            
            # Wait for API rate limit
            time.sleep(12)  # 5 requests per minute = 12 seconds between requests
            
        except Exception as e:
            logger.error(f"Error importing data for {ticker}: {str(e)}")
    
    logger.info("Stock data import completed")
    return True

if __name__ == "__main__":
    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting data import at {start_time}")
    
    # Import stock data
    success = import_stock_data()
    
    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time
    
    if success:
        logger.info(f"Data import completed successfully in {duration}")
    else:
        logger.error(f"Data import failed after {duration}")
