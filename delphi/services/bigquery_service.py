"""
BigQuery Storage Service

This module provides a storage service for BigQuery.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from google.cloud import bigquery
import functools

from .storage_service import StorageService
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BigQueryService(StorageService):
    """
    Storage service for BigQuery.
    """
    
    def __init__(self, project_id: str = None, dataset_id: str = None, table_id: str = None, location: str = None, cache_size: int = 128):
        """
        Initialize the BigQuery storage service.
        
        Args:
            project_id: Google Cloud project ID (if None, will try to load from config)
            dataset_id: BigQuery dataset ID (if None, will try to load from config)
            table_id: BigQuery table ID (if None, will try to load from config)
            location: BigQuery location (if None, will try to load from config)
            cache_size: Size of the LRU cache for get_market_data
        """
        # Load config
        config = get_config()
        
        # Set project ID
        self.project_id = project_id
        if self.project_id is None:
            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if self.project_id is None and config and 'database' in config and 'project_id' in config['database']:
            self.project_id = config['database']['project_id']
        
        # Set dataset ID
        self.dataset_id = dataset_id
        if self.dataset_id is None:
            self.dataset_id = os.environ.get("BIGQUERY_DATASET", "market_data")
        if self.dataset_id is None and config and 'database' in config and 'dataset_id' in config['database']:
            self.dataset_id = config['database']['dataset_id']
        
        # Set table ID
        self.table_id = table_id
        if self.table_id is None:
            self.table_id = os.environ.get("BIGQUERY_TABLE", "time_series")
        if self.table_id is None and config and 'database' in config and 'table_id' in config['database']:
            self.table_id = config['database']['table_id']
        
        # Set location
        self.location = location
        if self.location is None:
            self.location = os.environ.get("BIGQUERY_LOCATION", "US")
        if self.location is None and config and 'database' in config and 'location' in config['database']:
            self.location = config['database']['location']
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)
        
        # Apply caching to get_market_data
        self.get_market_data = functools.lru_cache(maxsize=cache_size)(self._get_market_data_impl)
        
        logger.info(f"Initialized BigQuery storage service for {self.project_id}.{self.dataset_id}.{self.table_id}")
    
    def setup_dataset(self) -> bool:
        """
        Set up BigQuery dataset.
        
        Returns:
            bool: Success status
        """
        try:
            # Create dataset if it doesn't exist
            dataset_ref = self.client.dataset(self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
            except Exception:
                # Create dataset
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = self.location
                dataset = self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up BigQuery dataset: {str(e)}")
            return False
    
    def setup_table(self) -> bool:
        """
        Set up BigQuery table.
        
        Returns:
            bool: Success status
        """
        try:
            # Create dataset if it doesn't exist
            if not self.setup_dataset():
                return False
            
            # Create table if it doesn't exist
            dataset_ref = self.client.dataset(self.dataset_id)
            table_ref = dataset_ref.table(self.table_id)
            try:
                self.client.get_table(table_ref)
                logger.info(f"Table {self.table_id} already exists")
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
                # Add clustering by symbol
                table.clustering_fields = ["symbol"]
                
                table = self.client.create_table(table)
                logger.info(f"Created table {self.table_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up BigQuery table: {str(e)}")
            return False
    
    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Store market data for a symbol.
        
        Args:
            symbol: Symbol to store data for
            data: DataFrame with market data
            
        Returns:
            bool: Success status
        """
        try:
            if data is None or data.empty:
                logger.warning(f"No data to store for {symbol}")
                return False
            
            # Ensure symbol column exists
            if "symbol" not in data.columns:
                data["symbol"] = symbol
            
            # Ensure date column exists
            if "date" not in data.columns and data.index.name != "date":
                data["date"] = data.index
            
            # Configure job
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            
            # Load data
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            job = self.client.load_table_from_dataframe(data, table_ref, job_config=job_config)
            job.result()  # Wait for job to complete
            
            logger.info(f"Stored {len(data)} rows for {symbol} in BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {str(e)}")
            return False
    
    def _get_market_data_impl(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Implementation of get_market_data.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            pd.DataFrame: Market data
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = @symbol
            """
            
            # Add date filters if provided
            if start_date:
                query += " AND date >= @start_date"
            
            if end_date:
                query += " AND date <= @end_date"
            
            query += " ORDER BY date"
            
            # Set query parameters
            query_params = [
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
            ]
            
            if start_date:
                query_params.append(bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()))
            
            if end_date:
                query_params.append(bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()))
            
            # Configure job
            job_config = bigquery.QueryJobConfig(
                query_parameters=query_params
            )
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Set date as index
            df = df.set_index("date")
            
            logger.info(f"Retrieved {len(df)} rows for {symbol} from BigQuery")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            pd.DataFrame: Market data
        """
        # This method is replaced by the cached version in __init__
        pass
    
    def get_latest_market_data(self, symbol: str, days: int = 1) -> pd.DataFrame:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            days: Number of days to get
            
        Returns:
            pd.DataFrame: Latest market data
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE symbol = @symbol
            ORDER BY date DESC
            LIMIT @days
            """
            
            # Set query parameters
            query_params = [
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("days", "INT64", days)
            ]
            
            # Configure job
            job_config = bigquery.QueryJobConfig(
                query_parameters=query_params
            )
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            df = query_job.to_dataframe()
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Set date as index
            df = df.set_index("date")
            
            # Sort by date
            df = df.sort_index()
            
            logger.info(f"Retrieved {len(df)} rows for {symbol} from BigQuery")
            return df
            
        except Exception as e:
            logger.error(f"Error getting latest market data for {symbol}: {str(e)}")
            return pd.DataFrame()
