"""
BigQuery client for the Volume Trading System.
"""
import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
import json

from volume_trading.config import config

# Configure logging
logger = logging.getLogger(__name__)

class BigQueryClient:
    """Client for interacting with Google BigQuery."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None,
                credentials_path: Optional[str] = None):
        """Initialize the BigQuery client.
        
        Args:
            project_id: Google Cloud project ID (optional, will use from config if not provided)
            dataset_id: BigQuery dataset ID (optional, will use from config if not provided)
            credentials_path: Path to service account credentials JSON file
        """
        self.project_id = project_id or config.get("google_cloud_project")
        self.dataset_id = dataset_id or config.get("bigquery_dataset")
        self.credentials_path = credentials_path
        
        # Initialize client
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Use service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self.client = bigquery.Client(
                    project=self.project_id,
                    credentials=credentials
                )
                logger.info(f"Initialized BigQuery client with service account credentials")
            else:
                # Use default credentials
                self.client = bigquery.Client(project=self.project_id)
                logger.info(f"Initialized BigQuery client with default credentials")
            
            # Ensure dataset exists
            self._ensure_dataset_exists()
            
        except Exception as e:
            logger.error(f"Error initializing BigQuery client: {str(e)}")
            self.client = None
    
    def _ensure_dataset_exists(self) -> bool:
        """Ensure the dataset exists, create it if it doesn't.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if dataset exists
            dataset_ref = self.client.dataset(self.dataset_id)
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {self.dataset_id} already exists")
                return True
            except Exception:
                # Dataset doesn't exist, create it
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"  # Set the location
                self.client.create_dataset(dataset)
                logger.info(f"Created dataset {self.dataset_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring dataset exists: {str(e)}")
            return False
    
    def create_table_if_not_exists(self, table_id: str, schema: List[Dict]) -> bool:
        """Create a table if it doesn't exist.
        
        Args:
            table_id: Table ID
            schema: Table schema as a list of field dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create table reference
            table_ref = self.client.dataset(self.dataset_id).table(table_id)
            
            try:
                # Check if table exists
                self.client.get_table(table_ref)
                logger.info(f"Table {table_id} already exists")
                return True
            except Exception:
                # Table doesn't exist, create it
                table = bigquery.Table(table_ref, schema=[
                    bigquery.SchemaField(field["name"], field["type"], mode=field.get("mode", "NULLABLE"))
                    for field in schema
                ])
                self.client.create_table(table)
                logger.info(f"Created table {table_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating table {table_id}: {str(e)}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, table_id: str, 
                        write_disposition: str = "WRITE_TRUNCATE") -> bool:
        """Upload a DataFrame to BigQuery.
        
        Args:
            df: DataFrame to upload
            table_id: Table ID
            write_disposition: Write disposition (WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if client is initialized
            if self.client is None:
                logger.error("BigQuery client not initialized")
                return False
            
            # Check if DataFrame is empty
            if df.empty:
                logger.warning(f"Empty DataFrame, skipping upload to {table_id}")
                return False
            
            # Create job config
            job_config = bigquery.LoadJobConfig(
                write_disposition=write_disposition
            )
            
            # Upload DataFrame
            table_ref = self.client.dataset(self.dataset_id).table(table_id)
            job = self.client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()  # Wait for job to complete
            
            logger.info(f"Uploaded {len(df)} rows to {table_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading DataFrame to {table_id}: {str(e)}")
            return False
    
    def query(self, query: str) -> pd.DataFrame:
        """Run a query and return the results as a DataFrame.
        
        Args:
            query: SQL query
            
        Returns:
            DataFrame with query results
        """
        try:
            # Check if client is initialized
            if self.client is None:
                logger.error("BigQuery client not initialized")
                return pd.DataFrame()
            
            # Run query
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            return pd.DataFrame()
    
    def get_table_schema(self, table_id: str) -> List[Dict]:
        """Get the schema of a table.
        
        Args:
            table_id: Table ID
            
        Returns:
            List of field dictionaries
        """
        try:
            # Check if client is initialized
            if self.client is None:
                logger.error("BigQuery client not initialized")
                return []
            
            # Get table
            table_ref = self.client.dataset(self.dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Convert schema to list of dictionaries
            schema = []
            for field in table.schema:
                schema.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode
                })
            
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for {table_id}: {str(e)}")
            return []
