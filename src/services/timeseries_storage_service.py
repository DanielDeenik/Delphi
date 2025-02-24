"""Time series data storage service for Google Cloud Platform."""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from pandas_gbq import to_gbq

logger = logging.getLogger(__name__)

class TimeSeriesStorageService:
    """Manages time series data storage in Google Cloud."""

    def __init__(self):
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

        self.bq_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        self.dataset_id = "market_data"
        self.bucket_name = f"{self.project_id}-market-data"
        self._ensure_storage_setup()

    def _ensure_storage_setup(self):
        """Ensure required BigQuery datasets and GCS buckets exist."""
        # Create BigQuery dataset
        dataset_ref = self.bq_client.dataset(self.dataset_id)
        try:
            self.bq_client.get_dataset(dataset_ref)
        except Exception:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            self.bq_client.create_dataset(dataset)
            logger.info(f"Created BigQuery dataset: {self.dataset_id}")

        # Create GCS bucket
        try:
            self.storage_client.get_bucket(self.bucket_name)
        except Exception:
            bucket = self.storage_client.create_bucket(self.bucket_name)
            logger.info(f"Created GCS bucket: {self.bucket_name}")

    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store market data in BigQuery and GCS."""
        try:
            # Store in BigQuery
            table_id = f"{self.project_id}.{self.dataset_id}.{symbol.lower()}_daily"
            data.to_gbq(table_id, 
                       project_id=self.project_id,
                       if_exists='append')

            # Store backup in Cloud Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"{symbol.lower()}/{datetime.now().strftime('%Y%m%d')}.parquet"
            blob = bucket.blob(blob_name)

            # Save as parquet
            data.to_parquet(f"/tmp/{blob_name}")
            blob.upload_from_filename(f"/tmp/{blob_name}")
            os.remove(f"/tmp/{blob_name}")

            return True

        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
            return False

    def get_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Retrieve market data from BigQuery."""
        try:
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{symbol.lower()}_daily`
            WHERE date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
            ORDER BY date
            """
            return self.bq_client.query(query).to_dataframe()

        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()