"""
Script to create BigQuery schemas for stock analysis system.
"""
import json
import os
from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        with open('config/tracked_stocks.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tracked stocks: {str(e)}")
        return {
            "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META", "ADBE", "ORCL", "ASML"],
            "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", "PLTR", "UBER", "LCID", "INTC", "XPEV"]
        }

def create_bigquery_schemas(project_id):
    """Create BigQuery schemas for stock analysis system."""
    client = bigquery.Client(project=project_id)
    dataset_id = "trading_insights"
    
    # Create dataset if it doesn't exist
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        client.get_dataset(dataset_ref)
        logger.info(f"Dataset {dataset_ref} already exists")
    except Exception:
        # Dataset doesn't exist, create it
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        dataset = client.create_dataset(dataset)
        logger.info(f"Created dataset {dataset_ref}")
    
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    all_stocks = tracked_stocks["buy"] + tracked_stocks["short"]
    
    # Create price tables for each stock
    for ticker in all_stocks:
        table_id = f"{dataset_ref}.stock_{ticker}_prices"
        
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("open", "FLOAT64"),
            bigquery.SchemaField("high", "FLOAT64"),
            bigquery.SchemaField("low", "FLOAT64"),
            bigquery.SchemaField("close", "FLOAT64"),
            bigquery.SchemaField("volume", "INTEGER"),
            bigquery.SchemaField("adjusted_close", "FLOAT64"),
            bigquery.SchemaField("symbol", "STRING")
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date"
        )
        
        try:
            client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except Exception:
            # Table doesn't exist, create it
            table = client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    # Create analysis tables for each stock
    for ticker in all_stocks:
        table_id = f"{dataset_ref}.stock_{ticker}_analysis"
        
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("volume_score", "FLOAT64"),
            bigquery.SchemaField("sentiment_score", "FLOAT64"),
            bigquery.SchemaField("prediction", "FLOAT64"),
            bigquery.SchemaField("signal", "STRING"),
            bigquery.SchemaField("confidence", "FLOAT64"),
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="date"
        )
        
        try:
            client.get_table(table_id)
            logger.info(f"Table {table_id} already exists")
        except Exception:
            # Table doesn't exist, create it
            table = client.create_table(table)
            logger.info(f"Created table {table_id}")
    
    # Create master summary table
    table_id = f"{dataset_ref}.master_summary"
    
    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("direction", "STRING"),
        bigquery.SchemaField("signal", "STRING"),
        bigquery.SchemaField("confidence", "FLOAT64"),
        bigquery.SchemaField("volume_score", "FLOAT64"),
        bigquery.SchemaField("sentiment_score", "FLOAT64"),
        bigquery.SchemaField("timestamp", "TIMESTAMP")
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="date"
    )
    
    try:
        client.get_table(table_id)
        logger.info(f"Table {table_id} already exists")
    except Exception:
        # Table doesn't exist, create it
        table = client.create_table(table)
        logger.info(f"Created table {table_id}")
    
    logger.info("BigQuery schema creation completed")

if __name__ == "__main__":
    # Get project ID from environment variable
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
        exit(1)
    
    create_bigquery_schemas(project_id)
