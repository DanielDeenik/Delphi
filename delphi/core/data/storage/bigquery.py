"""
BigQuery storage service.

This module provides a storage service for Google BigQuery.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from delphi.core.data.storage.base import StorageService

# Configure logger
logger = logging.getLogger(__name__)

class BigQueryStorage(StorageService):
    """Storage service for Google BigQuery."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None, **kwargs):
        """Initialize the BigQuery storage service.
        
        Args:
            project_id: Google Cloud project ID (default: from environment variable)
            dataset_id: BigQuery dataset ID (default: from environment variable or 'market_data')
            **kwargs: Additional arguments
        """
        # Get project ID from environment variable if not provided
        if project_id is None:
            project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
        
        # Get dataset ID from environment variable if not provided
        if dataset_id is None:
            dataset_id = os.environ.get('BIGQUERY_DATASET', 'market_data')
        
        self.project_id = project_id
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)
        
        super().__init__(**kwargs)
    
    def _initialize_storage(self) -> bool:
        """Initialize the BigQuery dataset.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        try:
            # Create dataset if it doesn't exist
            dataset_id = f"{self.project_id}.{self.dataset_id}"
            try:
                self.client.get_dataset(dataset_id)
                logger.info(f"Dataset {dataset_id} already exists")
            except NotFound:
                # Create dataset
                dataset = bigquery.Dataset(dataset_id)
                dataset.location = "US"
                dataset = self.client.create_dataset(dataset)
                logger.info(f"Created dataset {dataset_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing BigQuery dataset: {str(e)}")
            return False
    
    def create_stock_price_table(self, symbol: str) -> bool:
        """Create a table for stock price data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table ID
            table_id = f"{self.project_id}.{self.dataset_id}.stock_{symbol.lower()}_prices"
            
            # Check if table exists
            try:
                self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists")
                return True
            except NotFound:
                # Create table with optimized schema
                schema = [
                    # Primary dimension for partitioning
                    bigquery.SchemaField("date", "DATE", mode="REQUIRED",
                                        description="Trading date"),
                    # Clustering fields (frequently filtered/grouped)
                    bigquery.SchemaField("symbol", "STRING",
                                        description="Stock symbol"),
                    # Core metrics (frequently accessed together)
                    bigquery.SchemaField("close", "FLOAT64",
                                        description="Closing price"),
                    bigquery.SchemaField("adjusted_close", "FLOAT64",
                                        description="Adjusted closing price"),
                    bigquery.SchemaField("volume", "INTEGER",
                                        description="Trading volume"),
                    # Secondary metrics (less frequently accessed)
                    bigquery.SchemaField("open", "FLOAT64",
                                        description="Opening price"),
                    bigquery.SchemaField("high", "FLOAT64",
                                        description="Highest price"),
                    bigquery.SchemaField("low", "FLOAT64",
                                        description="Lowest price"),
                    bigquery.SchemaField("dividend", "FLOAT64",
                                        description="Dividend amount"),
                    bigquery.SchemaField("split_coefficient", "FLOAT64",
                                        description="Split coefficient")
                ]
                
                table = bigquery.Table(table_id, schema=schema)
                
                # Add table description
                table.description = f"Daily stock price data for {symbol}"
                
                # Set partitioning by date
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
                
                # Add clustering for improved query performance
                table.clustering_fields = ["symbol"]
                
                # Create table
                table = self.client.create_table(table)
                logger.info(f"Created table {table_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creating stock price table for {symbol}: {str(e)}")
            return False
    
    def create_volume_analysis_table(self, symbol: str) -> bool:
        """Create a table for volume analysis results.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table ID
            table_id = f"{self.project_id}.{self.dataset_id}.volume_{symbol.lower()}_analysis"
            
            # Check if table exists
            try:
                self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists")
                return True
            except NotFound:
                # Create table with optimized schema
                schema = [
                    # Primary dimension for partitioning
                    bigquery.SchemaField("date", "DATE", mode="REQUIRED",
                                        description="Trading date"),
                    # Clustering fields (frequently filtered/grouped)
                    bigquery.SchemaField("symbol", "STRING",
                                        description="Stock symbol"),
                    bigquery.SchemaField("signal", "STRING",
                                        description="Trading signal"),
                    # Key metrics for analysis (frequently accessed)
                    bigquery.SchemaField("is_volume_spike", "BOOLEAN",
                                        description="Whether a volume spike was detected"),
                    bigquery.SchemaField("volume_z_score", "FLOAT64",
                                        description="Volume Z-score"),
                    bigquery.SchemaField("spike_strength", "FLOAT64",
                                        description="Strength of volume spike"),
                    bigquery.SchemaField("confidence", "FLOAT64",
                                        description="Signal confidence"),
                    # Price and volume data
                    bigquery.SchemaField("close", "FLOAT64",
                                        description="Closing price"),
                    bigquery.SchemaField("volume", "INTEGER",
                                        description="Trading volume"),
                    bigquery.SchemaField("price_change_pct", "FLOAT64",
                                        description="Price change percentage"),
                    # Volume metrics (less frequently accessed)
                    bigquery.SchemaField("volume_ma5", "FLOAT64",
                                        description="5-day volume moving average"),
                    bigquery.SchemaField("volume_ma20", "FLOAT64",
                                        description="20-day volume moving average"),
                    bigquery.SchemaField("volume_ma50", "FLOAT64",
                                        description="50-day volume moving average"),
                    bigquery.SchemaField("relative_volume_5d", "FLOAT64",
                                        description="Volume relative to 5-day average"),
                    bigquery.SchemaField("relative_volume_20d", "FLOAT64",
                                        description="Volume relative to 20-day average"),
                    # Additional metadata
                    bigquery.SchemaField("notes", "STRING",
                                        description="Analysis notes"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP",
                                        description="Analysis timestamp")
                ]
                
                table = bigquery.Table(table_id, schema=schema)
                
                # Add table description
                table.description = f"Volume analysis results for {symbol}"
                
                # Set partitioning by date
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
                
                # Add clustering for improved query performance
                table.clustering_fields = ["symbol", "signal", "is_volume_spike"]
                
                # Create table
                table = self.client.create_table(table)
                logger.info(f"Created table {table_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creating volume analysis table for {symbol}: {str(e)}")
            return False
    
    def initialize_tables(self, symbols: List[str]) -> bool:
        """Initialize tables for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create tables for each symbol
            for symbol in symbols:
                self.create_stock_price_table(symbol)
                self.create_volume_analysis_table(symbol)
            
            logger.info(f"Initialized tables for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            return False
    
    def store_stock_prices(self, symbol: str, df: pd.DataFrame) -> bool:
        """Store stock price data in BigQuery.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with price data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table if it doesn't exist
            if not self.create_stock_price_table(symbol):
                logger.error(f"Failed to create table for {symbol}")
                return False
            
            # Make a copy of the DataFrame to avoid modifying the original
            upload_df = df.copy()
            
            # Ensure symbol column is present
            if 'symbol' not in upload_df.columns:
                upload_df['symbol'] = symbol
            
            # Ensure date column is in the correct format
            if 'date' in upload_df.columns and not pd.api.types.is_datetime64_dtype(upload_df['date']):
                upload_df['date'] = pd.to_datetime(upload_df['date'])
            
            # Upload to BigQuery
            table_id = f"{self.dataset_id}.stock_{symbol.lower()}_prices"
            
            # Use pandas_gbq to upload
            upload_df.to_gbq(
                destination_table=table_id,
                project_id=self.project_id,
                if_exists='replace'  # Use 'append' if you want to add to existing data
            )
            
            logger.info(f"Successfully stored {len(upload_df)} rows for {symbol} in BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error storing stock prices for {symbol}: {str(e)}")
            return False
    
    def store_volume_analysis(self, symbol: str, df: pd.DataFrame) -> bool:
        """Store volume analysis results in BigQuery.
        
        Args:
            symbol: Stock symbol
            df: DataFrame with volume analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table if it doesn't exist
            if not self.create_volume_analysis_table(symbol):
                logger.error(f"Failed to create volume analysis table for {symbol}")
                return False
            
            # Make a copy of the DataFrame to avoid modifying the original
            upload_df = df.copy()
            
            # Ensure symbol column is present
            if 'symbol' not in upload_df.columns:
                upload_df['symbol'] = symbol
            
            # Ensure date column is in the correct format
            if 'date' in upload_df.columns and not pd.api.types.is_datetime64_dtype(upload_df['date']):
                upload_df['date'] = pd.to_datetime(upload_df['date'])
            
            # Add timestamp if not present
            if 'timestamp' not in upload_df.columns:
                upload_df['timestamp'] = datetime.now()
            
            # Upload to BigQuery
            table_id = f"{self.dataset_id}.volume_{symbol.lower()}_analysis"
            
            # Use pandas_gbq to upload
            upload_df.to_gbq(
                destination_table=table_id,
                project_id=self.project_id,
                if_exists='replace'  # Use 'append' if you want to add to existing data
            )
            
            logger.info(f"Successfully stored {len(upload_df)} rows of volume analysis for {symbol} in BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error storing volume analysis for {symbol}: {str(e)}")
            return False
    
    def get_stock_prices(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get stock price data from BigQuery.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with price data
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.stock_{symbol.lower()}_prices`
            """
            
            if start_date or end_date:
                query += " WHERE "
                conditions = []
                
                if start_date:
                    conditions.append(f"date >= '{start_date.strftime('%Y-%m-%d')}'")
                
                if end_date:
                    conditions.append(f"date <= '{end_date.strftime('%Y-%m-%d')}'")
                
                query += " AND ".join(conditions)
            
            query += """
            ORDER BY date DESC
            """
            
            # Execute query
            df = self.client.query(query).to_dataframe()
            
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Retrieved {len(df)} rows of price data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock prices for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_volume_analysis(self, symbol: str, start_date: Optional[datetime] = None, 
                           end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get volume analysis results from BigQuery.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with volume analysis results
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.volume_{symbol.lower()}_analysis`
            """
            
            if start_date or end_date:
                query += " WHERE "
                conditions = []
                
                if start_date:
                    conditions.append(f"date >= '{start_date.strftime('%Y-%m-%d')}'")
                
                if end_date:
                    conditions.append(f"date <= '{end_date.strftime('%Y-%m-%d')}'")
                
                query += " AND ".join(conditions)
            
            query += """
            ORDER BY date DESC
            """
            
            # Execute query
            df = self.client.query(query).to_dataframe()
            
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Retrieved {len(df)} rows of volume analysis for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting volume analysis for {symbol}: {str(e)}")
            return pd.DataFrame()
