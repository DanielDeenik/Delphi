"""Time series data storage service for efficient cloud storage."""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage
from firebase_admin import db
from pandas_gbq import to_gbq

logger = logging.getLogger(__name__)

class TimeSeriesStorageService:
    """Manages time series data storage across different storage solutions."""

    def __init__(self):
        self.bq_client = bigquery.Client()
        self.storage_client = storage.Client()
        self.dataset_id = "market_data"
        self.bucket_name = "market_data_cold_storage"
        self._ensure_storage_setup()

    def _ensure_storage_setup(self):
        """Ensure required BigQuery datasets and GCS buckets exist."""
        try:
            # Create BigQuery dataset if it doesn't exist
            dataset_ref = self.bq_client.dataset(self.dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
            except Exception:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                self.bq_client.create_dataset(dataset)
                logger.info(f"Created BigQuery dataset: {self.dataset_id}")

            # Create GCS bucket if it doesn't exist
            try:
                self.storage_client.get_bucket(self.bucket_name)
            except Exception:
                bucket = self.storage_client.create_bucket(self.bucket_name)
                logger.info(f"Created GCS bucket: {self.bucket_name}")

        except Exception as e:
            logger.error(f"Error setting up storage: {str(e)}")
            raise

    def store_market_data(self, 
                         symbol: str, 
                         data: pd.DataFrame,
                         partition_size: str = "1D") -> bool:
        """
        Store market data efficiently based on time partitions.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data: DataFrame with market data
            partition_size: Size of time partitions ('1D', '1H', etc.)
        """
        try:
            # Ensure data is properly indexed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Partition data
            partitions = self._partition_data(data, partition_size)

            # Store real-time data in Firebase
            latest_data = data.tail(1).to_dict('records')[0]
            latest_data['timestamp'] = datetime.now().isoformat()
            db.reference(f'market_data/{symbol}/latest').set(latest_data)

            # Store historical data in BigQuery
            table_id = f"{self.dataset_id}.{symbol.lower()}_market_data"
            
            # Use efficient schema with clustering
            job_config = bigquery.LoadJobConfig(
                clustering_fields=["date"],
                time_partitioning=bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
            )

            # Upload to BigQuery
            for partition_date, partition_data in partitions.items():
                partition_data['date'] = partition_date
                to_gbq(partition_data, 
                      table_id, 
                      if_exists='append',
                      project_id=os.getenv('GOOGLE_CLOUD_PROJECT'))

            # Archive old data to GCS
            self._archive_old_data(symbol, data)

            return True

        except Exception as e:
            logger.error(f"Error storing market data: {str(e)}")
            return False

    def _partition_data(self, 
                       data: pd.DataFrame, 
                       partition_size: str) -> Dict[str, pd.DataFrame]:
        """Partition data into smaller chunks for efficient storage."""
        return {
            date: group for date, group in data.groupby(
                pd.Grouper(freq=partition_size)
            )
        }

    def _archive_old_data(self, 
                         symbol: str, 
                         data: pd.DataFrame,
                         days_threshold: int = 90):
        """Archive old data to cold storage."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            old_data = data[data.index < cutoff_date]

            if not old_data.empty:
                # Save to GCS
                bucket = self.storage_client.bucket(self.bucket_name)
                archive_path = f"{symbol}/archive_{cutoff_date.strftime('%Y%m')}.parquet"
                blob = bucket.blob(archive_path)
                
                # Save as parquet for efficient storage
                old_data.to_parquet(f"/tmp/{archive_path}")
                blob.upload_from_filename(f"/tmp/{archive_path}")
                os.remove(f"/tmp/{archive_path}")

                logger.info(f"Archived old data for {symbol} to GCS")

        except Exception as e:
            logger.error(f"Error archiving old data: {str(e)}")

    def get_market_data(self,
                       symbol: str,
                       start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """
        Retrieve market data from the most appropriate storage.
        
        Automatically fetches from:
        - Firebase for real-time data
        - BigQuery for recent historical data
        - GCS for archived data
        """
        try:
            # Get real-time data from Firebase
            if end_date >= datetime.now() - timedelta(minutes=5):
                latest_ref = db.reference(f'market_data/{symbol}/latest')
                latest_data = latest_ref.get()
                if latest_data:
                    latest_df = pd.DataFrame([latest_data])
                    latest_df.set_index('timestamp', inplace=True)

            # Query historical data from BigQuery
            query = f"""
                SELECT *
                FROM `{self.dataset_id}.{symbol.lower()}_market_data`
                WHERE date BETWEEN @start_date AND @end_date
                ORDER BY date
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                    bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
                ]
            )
            
            historical_df = self.bq_client.query(query, job_config=job_config).to_dataframe()

            # Combine results
            if 'latest_df' in locals():
                return pd.concat([historical_df, latest_df])
            return historical_df

        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()

    def cleanup_old_data(self, 
                        symbol: str, 
                        days_to_keep: int = 90) -> bool:
        """Clean up old data from BigQuery after successful archival."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            query = f"""
                DELETE FROM `{self.dataset_id}.{symbol.lower()}_market_data`
                WHERE date < @cutoff_date
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("cutoff_date", "DATE", cutoff_date),
                ]
            )
            
            query_job = self.bq_client.query(query, job_config=job_config)
            query_job.result()
            
            return True

        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return False
