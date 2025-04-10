"""Time series data storage service using InfluxDB."""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)

class InfluxDBStorageService:
    """Manages time series data storage in InfluxDB with path to BigQuery migration."""

    def __init__(self):
        # InfluxDB connection parameters
        self.url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        self.token = os.getenv('INFLUXDB_TOKEN', '')
        self.org = os.getenv('INFLUXDB_ORG', 'delphi')
        self.bucket = os.getenv('INFLUXDB_BUCKET', 'market_data')
        
        # Initialize InfluxDB client
        self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        
        # Ensure bucket exists
        self._ensure_storage_setup()

    def _ensure_storage_setup(self):
        """Ensure required InfluxDB bucket exists."""
        try:
            buckets_api = self.client.buckets_api()
            bucket_list = buckets_api.find_buckets().buckets
            bucket_names = [bucket.name for bucket in bucket_list]
            
            if self.bucket not in bucket_names:
                logger.info(f"Creating InfluxDB bucket: {self.bucket}")
                buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)
                logger.info(f"Created InfluxDB bucket: {self.bucket}")
        except Exception as e:
            logger.error(f"Error setting up InfluxDB: {str(e)}")
            # Continue even if setup fails, as bucket might already exist
            pass

    def store_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Store market data in InfluxDB.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            data: DataFrame with time series data
                Expected columns: Open, High, Low, Close, Volume, etc.
                Index should be datetime
        
        Returns:
            bool: Success status
        """
        try:
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    data.index = pd.to_datetime(data.index)
            
            # Convert DataFrame to InfluxDB points
            points = []
            for timestamp, row in data.iterrows():
                point = Point("market_data") \
                    .tag("symbol", symbol) \
                    .time(timestamp, WritePrecision.NS)
                
                # Add all numeric columns as fields
                for col, val in row.items():
                    if pd.api.types.is_numeric_dtype(type(val)):
                        point = point.field(col, float(val))
                
                points.append(point)
            
            # Write to InfluxDB
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"Successfully stored {len(points)} data points for {symbol} in InfluxDB")
            
            return True
        except Exception as e:
            logger.error(f"Error storing market data in InfluxDB: {str(e)}")
            return False

    def get_market_data(self, symbol: str, start_date: Optional[datetime] = None, 
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve market data from InfluxDB.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            start_date: Start date for data retrieval (default: 30 days ago)
            end_date: End date for data retrieval (default: now)
        
        Returns:
            pd.DataFrame: DataFrame with time series data
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=30)
            
            # Format dates for Flux query
            start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Build Flux query
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_str}, stop: {end_str})
                |> filter(fn: (r) => r._measurement == "market_data")
                |> filter(fn: (r) => r.symbol == "{symbol}")
            '''
            
            # Execute query
            tables = self.query_api.query(query)
            
            # Convert to DataFrame
            if not tables:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return pd.DataFrame()
            
            # Process results into a DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    records.append({
                        'time': record.get_time(),
                        'field': record.get_field(),
                        'value': record.get_value(),
                        'symbol': record.values.get('symbol')
                    })
            
            if not records:
                return pd.DataFrame()
            
            # Convert to wide format DataFrame
            df = pd.DataFrame(records)
            df_pivot = df.pivot_table(
                index='time', 
                columns='field', 
                values='value'
            )
            
            # Reset index to make datetime a column
            df_pivot.reset_index(inplace=True)
            df_pivot.rename(columns={'time': 'date'}, inplace=True)
            
            return df_pivot
            
        except Exception as e:
            logger.error(f"Error retrieving market data from InfluxDB: {str(e)}")
            return pd.DataFrame()
    
    def export_to_bigquery(self, symbol: str, project_id: str, dataset_id: str = "market_data",
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> bool:
        """Export data from InfluxDB to BigQuery for long-term storage.
        
        This provides the migration path to BigQuery when needed.
        
        Args:
            symbol: Trading symbol to export
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
            start_date: Start date for data export
            end_date: End date for data export
            
        Returns:
            bool: Success status
        """
        try:
            # Get data from InfluxDB
            df = self.get_market_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data to export for {symbol}")
                return False
            
            # Export to BigQuery
            table_id = f"{project_id}.{dataset_id}.{symbol.lower()}_daily"
            df.to_gbq(table_id, project_id=project_id, if_exists='append')
            
            logger.info(f"Successfully exported {len(df)} rows for {symbol} to BigQuery")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to BigQuery: {str(e)}")
            return False
