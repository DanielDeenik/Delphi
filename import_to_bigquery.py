#!/usr/bin/env python3
"""
Import VIX, SPY, and PLTR data to BigQuery.
Based on working code from GitHub repository.
"""

import os
import pandas as pd
import requests
import time
import datetime
import logging
from google.cloud import bigquery

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Alpha Vantage API key - replace with your key
ALPHA_VANTAGE_KEY = "YOUR_ALPHA_VANTAGE_KEY"

# Google Cloud project ID - replace with your project ID
PROJECT_ID = "delphi-449908"

# BigQuery dataset and table
DATASET_ID = "market_data"
TABLE_ID = "time_series"

# Dictionary of symbols with their names
SYMBOLS_DICT = {
    '^VIX': 'CBOE Volatility Index',
    'SPY': 'SPDR S&P 500 ETF Trust',
    'PLTR': 'Palantir Technologies Inc.'
}

def setup_bigquery():
    """Set up BigQuery dataset and table."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        
        # Create dataset if it doesn't exist
        dataset_ref = client.dataset(DATASET_ID)
        try:
            client.get_dataset(dataset_ref)
            logger.info(f"Dataset {DATASET_ID} already exists")
        except Exception:
            # Create dataset
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            dataset = client.create_dataset(dataset)
            logger.info(f"Created dataset {DATASET_ID}")
        
        # Create table if it doesn't exist
        table_ref = dataset_ref.table(TABLE_ID)
        try:
            client.get_table(table_ref)
            logger.info(f"Table {TABLE_ID} already exists")
        except Exception:
            # Create table
            schema = [
                bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("symbol_name", "STRING"),
                bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("open", "FLOAT"),
                bigquery.SchemaField("high", "FLOAT"),
                bigquery.SchemaField("low", "FLOAT"),
                bigquery.SchemaField("close", "FLOAT"),
                bigquery.SchemaField("adjusted_close", "FLOAT"),
                bigquery.SchemaField("volume", "INTEGER"),
                bigquery.SchemaField("dividend", "FLOAT"),
                bigquery.SchemaField("split_coefficient", "FLOAT"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
            ]
            
            table = bigquery.Table(table_ref, schema=schema)
            # Add clustering by symbol and date
            table.clustering_fields = ["symbol", "date"]
            # Add time partitioning by date
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            )
            
            table = client.create_table(table)
            logger.info(f"Created table {TABLE_ID}")
        
        return client
        
    except Exception as e:
        logger.error(f"Error setting up BigQuery: {str(e)}")
        raise

def pull_time_series_stock_data(from_date=None, to_date=None):
    """
    Pull time series daily adjusted stock data using Alpha Vantage API.
    
    Args:
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: Stock data
    """
    # Dictionary to store raw data
    stock_data = {}
    
    # Counter to track calls
    counter = 1
    
    # Iterate through each symbol
    for symbol in SYMBOLS_DICT.keys():
        # Use try to avoid unanticipated API errors
        try:
            if counter % 6 != 0:
                # Update URL
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}&outputsize=full'
                response = requests.get(url)
                
                if 'Time Series (Daily)' in response.json():
                    stock_data[symbol] = response.json()['Time Series (Daily)']
                    logger.info(f"Successfully pulled data for {symbol}")
                else:
                    logger.error(f"Error in API response for {symbol}: {response.json()}")
                
                counter += 1
            else:
                # Pause after 5 calls
                logger.info('Reached max 5 API calls per minute. Pausing for 60s...')
                time.sleep(62)
                counter += 1
        except Exception as e:
            # Print exceptions for respective stocks/symbols
            logger.error(f'Error: Unable to pull data for {symbol} - {str(e)}')
            counter += 1
    
    # Prepare dataframe
    df_stock = []
    df_day = []
    df_open = []
    df_high = []
    df_low = []
    df_close = []
    df_adjusted_close = []
    df_volume = []
    df_dividend = []
    df_split = []
    
    for stock in stock_data.keys():
        for day in stock_data[stock].keys():
            df_stock.append(stock)
            df_day.append(day)
            df_open.append(float(stock_data[stock][day]['1. open']))
            df_high.append(float(stock_data[stock][day]['2. high']))
            df_low.append(float(stock_data[stock][day]['3. low']))
            df_close.append(float(stock_data[stock][day]['4. close']))
            df_adjusted_close.append(float(stock_data[stock][day]['5. adjusted close']))
            df_volume.append(int(stock_data[stock][day]['6. volume']))
            df_dividend.append(float(stock_data[stock][day]['7. dividend amount']))
            df_split.append(float(stock_data[stock][day]['8. split coefficient']))
    
    # Create dataframe
    stock_df = pd.DataFrame({
        'symbol': df_stock,
        'symbol_name': [SYMBOLS_DICT[i] for i in df_stock],
        'date': df_day,
        'open': df_open,
        'high': df_high,
        'low': df_low,
        'close': df_close,
        'adjusted_close': df_adjusted_close,
        'volume': df_volume,
        'dividend': df_dividend,
        'split_coefficient': df_split,
        'created_at': datetime.datetime.now()
    })
    
    # Apply filters to only capture data as per from and to dates that were passed
    if from_date:
        stock_df = stock_df[stock_df['date'] >= from_date]
    
    if to_date:
        stock_df = stock_df[stock_df['date'] < to_date]
    
    # Save data locally for reference
    stock_df.to_csv('stocks_df.csv', index=False)
    
    logger.info(f"Created DataFrame with {len(stock_df)} rows")
    
    return stock_df

def load_to_bigquery(df):
    """
    Load DataFrame to BigQuery.
    
    Args:
        df: DataFrame to load
        
    Returns:
        bool: Success status
    """
    try:
        if df.empty:
            logger.warning("DataFrame is empty. Nothing to load to BigQuery.")
            return False
        
        # Initialize BigQuery client
        client = setup_bigquery()
        
        # Configure job
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        
        # Load data
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for job to complete
        
        logger.info(f"Loaded {len(df)} rows to {table_ref}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data to BigQuery: {str(e)}")
        return False

def main():
    """Main function."""
    try:
        # Set date range - last 2 years
        to_date = datetime.datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y-%m-%d')
        
        logger.info(f"Importing data from {from_date} to {to_date}")
        
        # Pull time series data
        stock_df = pull_time_series_stock_data(from_date, to_date)
        
        # Load to BigQuery
        success = load_to_bigquery(stock_df)
        
        if success:
            logger.info("Successfully imported data to BigQuery")
        else:
            logger.error("Failed to import data to BigQuery")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return False

if __name__ == "__main__":
    # Replace with your Alpha Vantage API key
    ALPHA_VANTAGE_KEY = input("Enter your Alpha Vantage API key: ")
    
    # Run the import
    success = main()
    
    if success:
        print("\nData import completed successfully!")
    else:
        print("\nData import failed. Check the logs for details.")
