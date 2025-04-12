"""
BigQuery storage module for the Volume Intelligence Trading System.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas_gbq

from trading_ai.config import config_manager

# Configure logging
logger = logging.getLogger(__name__)

class BigQueryStorage:
    """BigQuery storage for market data and analysis results."""

    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None):
        """Initialize the BigQuery storage.

        Args:
            project_id: Google Cloud project ID. If not provided, will use the one from config.
            dataset_id: BigQuery dataset ID. If not provided, will use the one from config.
        """
        self.project_id = project_id or config_manager.system_config.google_cloud_project
        self.dataset_id = dataset_id or config_manager.system_config.bigquery_dataset

        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)

        # Initialize dataset
        self._initialize_dataset()

    def _initialize_dataset(self) -> bool:
        """Initialize the BigQuery dataset."""
        try:
            # Check if dataset exists
            dataset_ref = f"{self.project_id}.{self.dataset_id}"
            try:
                self.client.get_dataset(dataset_ref)
                logger.info(f"Dataset {dataset_ref} already exists")
            except NotFound:
                # Create dataset
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                self.client.create_dataset(dataset)
                logger.info(f"Created dataset {dataset_ref}")

            return True

        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            return False

    def create_stock_price_table(self, ticker: str) -> bool:
        """Create a table for stock price data.

        Args:
            ticker: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Table reference
            table_id = f"{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices"

            # Check if table exists and delete it if it does
            try:
                self.client.delete_table(table_id)
                logger.info(f"Deleted existing table {table_id}")
            except NotFound:
                logger.info(f"Table {table_id} does not exist, creating new table")

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
            table.description = f"Daily stock price data for {ticker}"

            # Set partitioning by month instead of day to avoid partition limits
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.MONTH,
                field="date"
            )

            # Add clustering for improved query performance
            table.clustering_fields = ["symbol"]

            self.client.create_table(table)
            logger.info(f"Created table {table_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating stock price table for {ticker}: {str(e)}")
            return False

    def create_volume_analysis_table(self, ticker: str) -> bool:
        """Create a table for volume analysis results.

        Args:
            ticker: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Table reference
            table_id = f"{self.project_id}.{self.dataset_id}.volume_{ticker.lower()}_analysis"

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
                table.description = f"Volume analysis results for {ticker}"

                # Set partitioning by date
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )

                # Add clustering for improved query performance
                table.clustering_fields = ["symbol", "signal", "is_volume_spike"]

                self.client.create_table(table)
                logger.info(f"Created volume analysis table {table_id}")
                return True

        except Exception as e:
            logger.error(f"Error creating volume analysis table for {ticker}: {str(e)}")
            return False

    def create_master_summary_table(self) -> bool:
        """Create a table for the master summary.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Table reference
            table_id = f"{self.project_id}.{self.dataset_id}.master_summary"

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
                    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED",
                                        description="Stock symbol"),
                    bigquery.SchemaField("direction", "STRING",
                                        description="Trading direction (buy or short)"),
                    bigquery.SchemaField("signal", "STRING",
                                        description="Trading signal"),
                    # Key metrics for analysis (frequently accessed)
                    bigquery.SchemaField("is_volume_spike", "BOOLEAN",
                                        description="Whether a volume spike was detected"),
                    bigquery.SchemaField("confidence", "FLOAT64",
                                        description="Signal confidence"),
                    # Price and volume data
                    bigquery.SchemaField("close", "FLOAT64",
                                        description="Closing price"),
                    bigquery.SchemaField("volume", "INTEGER",
                                        description="Trading volume"),
                    bigquery.SchemaField("relative_volume", "FLOAT64",
                                        description="Volume relative to average"),
                    bigquery.SchemaField("volume_z_score", "FLOAT64",
                                        description="Volume Z-score"),
                    bigquery.SchemaField("spike_strength", "FLOAT64",
                                        description="Strength of volume spike"),
                    bigquery.SchemaField("price_change_pct", "FLOAT64",
                                        description="Price change percentage"),
                    # Trade parameters
                    bigquery.SchemaField("stop_loss", "FLOAT64",
                                        description="Stop loss price"),
                    bigquery.SchemaField("take_profit", "FLOAT64",
                                        description="Take profit price"),
                    bigquery.SchemaField("risk_reward_ratio", "FLOAT64",
                                        description="Risk-reward ratio"),
                    # Additional metadata
                    bigquery.SchemaField("notes", "STRING",
                                        description="Analysis notes"),
                    bigquery.SchemaField("notebook_url", "STRING",
                                        description="URL to analysis notebook"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP",
                                        description="Analysis timestamp")
                ]

                table = bigquery.Table(table_id, schema=schema)

                # Add table description
                table.description = "Master summary of volume analysis across all stocks"

                # Set partitioning by date
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )

                # Add clustering for improved query performance
                table.clustering_fields = ["symbol", "signal", "direction"]

                self.client.create_table(table)
                logger.info(f"Created master summary table {table_id}")
                return True

        except Exception as e:
            logger.error(f"Error creating master summary table: {str(e)}")
            return False

    def create_trade_log_table(self) -> bool:
        """Create a table for trade logs.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Table reference
            table_id = f"{self.project_id}.{self.dataset_id}.trade_logs"

            # Check if table exists
            try:
                self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists")
                return True
            except NotFound:
                # Create table
                schema = [
                    bigquery.SchemaField("trade_id", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
                    bigquery.SchemaField("direction", "STRING"),  # 'buy' or 'short'
                    bigquery.SchemaField("entry_date", "DATE"),
                    bigquery.SchemaField("entry_price", "FLOAT64"),
                    bigquery.SchemaField("position_size", "FLOAT64"),
                    bigquery.SchemaField("stop_loss", "FLOAT64"),
                    bigquery.SchemaField("take_profit", "FLOAT64"),
                    bigquery.SchemaField("exit_date", "DATE"),
                    bigquery.SchemaField("exit_price", "FLOAT64"),
                    bigquery.SchemaField("pnl_amount", "FLOAT64"),
                    bigquery.SchemaField("pnl_percent", "FLOAT64"),
                    bigquery.SchemaField("trade_duration_days", "INTEGER"),
                    bigquery.SchemaField("status", "STRING"),  # 'open', 'closed', 'cancelled'
                    bigquery.SchemaField("trigger_reason", "STRING"),
                    bigquery.SchemaField("exit_reason", "STRING"),
                    bigquery.SchemaField("notes", "STRING"),
                    bigquery.SchemaField("created_at", "TIMESTAMP"),
                    bigquery.SchemaField("updated_at", "TIMESTAMP")
                ]

                table = bigquery.Table(table_id, schema=schema)

                self.client.create_table(table)
                logger.info(f"Created trade log table {table_id}")
                return True

        except Exception as e:
            logger.error(f"Error creating trade log table: {str(e)}")
            return False

    def initialize_tables(self) -> bool:
        """Initialize all required tables.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create tables for each tracked stock
            all_tickers = config_manager.get_all_tickers()

            for ticker in all_tickers:
                self.create_stock_price_table(ticker)
                self.create_volume_analysis_table(ticker)

            # Create master summary table
            self.create_master_summary_table()

            # Create trade log table
            self.create_trade_log_table()

            logger.info("All tables initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing tables: {str(e)}")
            return False

    def store_stock_prices(self, ticker: str, df: pd.DataFrame, if_exists: str = 'replace') -> bool:
        """Store stock price data in BigQuery.

        Args:
            ticker: Stock symbol
            df: DataFrame with price data
            if_exists: What to do if the table already exists ('replace', 'append', 'fail')

        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker}, skipping storage")
                return False

            # Ensure the table exists
            self.create_stock_price_table(ticker)

            # Prepare DataFrame for upload
            upload_df = df.copy()

            # Reset index if date is the index
            if isinstance(upload_df.index, pd.DatetimeIndex):
                upload_df = upload_df.reset_index()
                upload_df = upload_df.rename(columns={'index': 'date'})

            # Ensure date column is in the correct format
            if 'date' in upload_df.columns and not pd.api.types.is_datetime64_dtype(upload_df['date']):
                upload_df['date'] = pd.to_datetime(upload_df['date'])

            # Ensure all required columns are present
            required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in upload_df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for {ticker}: {missing_columns}")
                return False

            # Ensure numeric columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in upload_df.columns:
                    upload_df[col] = pd.to_numeric(upload_df[col], errors='coerce')

            # Check for NaN values and handle them
            nan_counts = upload_df[numeric_columns].isna().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"NaN values found in {ticker} data: {nan_counts}")
                # Fill NaN values with appropriate defaults or drop rows
                upload_df = upload_df.dropna(subset=numeric_columns)
                if upload_df.empty:
                    logger.error(f"All rows contain NaN values for {ticker}, skipping storage")
                    return False

            # Upload to BigQuery
            table_id = f"{self.dataset_id}.stock_{ticker.lower()}_prices"

            # Use a try-except block to handle potential errors during upload
            try:
                # Use pandas_gbq to upload data
                import pandas_gbq

                # Create a schema to ensure proper data types
                schema = [
                    {'name': 'date', 'type': 'DATE'},
                    {'name': 'symbol', 'type': 'STRING'},
                    {'name': 'open', 'type': 'FLOAT'},
                    {'name': 'high', 'type': 'FLOAT'},
                    {'name': 'low', 'type': 'FLOAT'},
                    {'name': 'close', 'type': 'FLOAT'},
                    {'name': 'adjusted_close', 'type': 'FLOAT'},
                    {'name': 'volume', 'type': 'INTEGER'},
                    {'name': 'dividend', 'type': 'FLOAT'},
                    {'name': 'split_coefficient', 'type': 'FLOAT'}
                ]

                # Upload with schema
                pandas_gbq.to_gbq(
                    upload_df,
                    destination_table=table_id,
                    project_id=self.project_id,
                    if_exists=if_exists,
                    table_schema=schema
                )
            except Exception as upload_error:
                logger.error(f"Error during BigQuery upload for {ticker}: {str(upload_error)}")

                # If the error is related to schema mismatch, try to fix it
                if 'schema mismatch' in str(upload_error).lower():
                    logger.info(f"Attempting to fix schema mismatch for {ticker}")
                    try:
                        # Get the existing table schema
                        table_ref = f"{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices"
                        table = self.client.get_table(table_ref)
                        existing_schema = {field.name: field.field_type for field in table.schema}

                        # Adjust DataFrame types to match schema
                        for col, dtype in existing_schema.items():
                            if col in upload_df.columns:
                                if dtype == 'FLOAT' or dtype == 'FLOAT64':
                                    upload_df[col] = pd.to_numeric(upload_df[col], errors='coerce')
                                elif dtype == 'INTEGER' or dtype == 'INT64':
                                    upload_df[col] = pd.to_numeric(upload_df[col], errors='coerce').astype('Int64')
                                elif dtype == 'BOOLEAN':
                                    upload_df[col] = upload_df[col].astype('bool')
                                elif dtype == 'STRING':
                                    upload_df[col] = upload_df[col].astype('str')
                                elif dtype == 'DATE' or dtype == 'DATETIME' or dtype == 'TIMESTAMP':
                                    upload_df[col] = pd.to_datetime(upload_df[col], errors='coerce')

                        # Try a different approach - recreate the table and then upload
                        self.create_stock_price_table(ticker)

                        # Try upload again with schema
                        import pandas_gbq

                        # Create a schema to ensure proper data types
                        schema = [
                            {'name': 'date', 'type': 'DATE'},
                            {'name': 'symbol', 'type': 'STRING'},
                            {'name': 'open', 'type': 'FLOAT'},
                            {'name': 'high', 'type': 'FLOAT'},
                            {'name': 'low', 'type': 'FLOAT'},
                            {'name': 'close', 'type': 'FLOAT'},
                            {'name': 'adjusted_close', 'type': 'FLOAT'},
                            {'name': 'volume', 'type': 'INTEGER'},
                            {'name': 'dividend', 'type': 'FLOAT'},
                            {'name': 'split_coefficient', 'type': 'FLOAT'}
                        ]

                        # Upload with schema
                        pandas_gbq.to_gbq(
                            upload_df,
                            destination_table=table_id,
                            project_id=self.project_id,
                            if_exists='append',  # Use append since we just recreated the table
                            table_schema=schema
                        )
                    except Exception as schema_fix_error:
                        logger.error(f"Failed to fix schema mismatch for {ticker}: {str(schema_fix_error)}")
                        return False
                else:
                    return False

            logger.info(f"Successfully stored {len(upload_df)} rows for {ticker} in BigQuery")
            return True

        except Exception as e:
            logger.error(f"Error storing stock prices for {ticker}: {str(e)}")
            return False

    def store_volume_analysis(self, ticker: str, df: pd.DataFrame) -> bool:
        """Store volume analysis results in BigQuery.

        Args:
            ticker: Stock symbol
            df: DataFrame with volume analysis results

        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker} volume analysis, skipping storage")
                return False

            # Ensure the table exists
            self.create_volume_analysis_table(ticker)

            # Prepare DataFrame for upload
            upload_df = df.copy()

            # Reset index if date is the index
            if isinstance(upload_df.index, pd.DatetimeIndex):
                upload_df = upload_df.reset_index()
                upload_df = upload_df.rename(columns={'index': 'date'})

            # Ensure date column is in the correct format
            if 'date' in upload_df.columns and not pd.api.types.is_datetime64_dtype(upload_df['date']):
                upload_df['date'] = pd.to_datetime(upload_df['date'])

            # Keep date as datetime for BigQuery (it expects DATE type for partitioning)

            # Add timestamp if not present
            if 'timestamp' not in upload_df.columns:
                upload_df['timestamp'] = datetime.now()

            # Upload to BigQuery
            table_id = f"{self.dataset_id}.volume_{ticker.lower()}_analysis"

            upload_df.to_gbq(
                destination_table=table_id,
                project_id=self.project_id,
                if_exists='replace'  # Use 'append' if you want to add to existing data
            )

            logger.info(f"Successfully stored {len(upload_df)} rows of volume analysis for {ticker} in BigQuery")
            return True

        except Exception as e:
            logger.error(f"Error storing volume analysis for {ticker}: {str(e)}")
            return False

    def store_master_summary(self, df: pd.DataFrame) -> bool:
        """Store master summary in BigQuery.

        Args:
            df: DataFrame with master summary data

        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame for master summary, skipping storage")
                return False

            # Ensure the table exists
            self.create_master_summary_table()

            # Prepare DataFrame for upload
            upload_df = df.copy()

            # Reset index if date is the index
            if isinstance(upload_df.index, pd.DatetimeIndex):
                upload_df = upload_df.reset_index()
                upload_df = upload_df.rename(columns={'index': 'date'})

            # Ensure date column is in the correct format
            if 'date' in upload_df.columns and not pd.api.types.is_datetime64_dtype(upload_df['date']):
                upload_df['date'] = pd.to_datetime(upload_df['date'])

            # Keep date as datetime for BigQuery (it expects DATE type for partitioning)

            # Add timestamp if not present
            if 'timestamp' not in upload_df.columns:
                upload_df['timestamp'] = datetime.now()

            # Upload to BigQuery
            table_id = f"{self.dataset_id}.master_summary"

            upload_df.to_gbq(
                destination_table=table_id,
                project_id=self.project_id,
                if_exists='replace'  # Use 'append' if you want to add to existing data
            )

            logger.info(f"Successfully stored {len(upload_df)} rows of master summary in BigQuery")
            return True

        except Exception as e:
            logger.error(f"Error storing master summary: {str(e)}")
            return False

    def log_trade(self, trade_data: Dict) -> bool:
        """Log a trade in BigQuery.

        Args:
            trade_data: Dictionary with trade data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure the table exists
            self.create_trade_log_table()

            # Convert to DataFrame
            df = pd.DataFrame([trade_data])

            # Ensure date columns are in the correct format
            for date_col in ['entry_date', 'exit_date']:
                if date_col in df.columns and not pd.api.types.is_datetime64_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col])
                    # Keep as datetime for BigQuery

            # Add timestamps if not present
            now = datetime.now()
            if 'created_at' not in df.columns:
                df['created_at'] = now
            if 'updated_at' not in df.columns:
                df['updated_at'] = now

            # Upload to BigQuery
            table_id = f"{self.dataset_id}.trade_logs"

            df.to_gbq(
                destination_table=table_id,
                project_id=self.project_id,
                if_exists='append'
            )

            logger.info(f"Successfully logged trade for {trade_data.get('symbol')} in BigQuery")
            return True

        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            return False

    def get_stock_prices(self, ticker: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, days: Optional[int] = None) -> pd.DataFrame:
        """Get stock price data from BigQuery.

        Args:
            ticker: Stock symbol
            start_date: Start date for data retrieval (inclusive)
            end_date: End date for data retrieval (inclusive)
            days: Number of days of data to retrieve (None for all data)
                  Only used if start_date is not provided

        Returns:
            DataFrame with price data
        """
        try:
            # Check if table exists
            table_ref = f"{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices"
            try:
                self.client.get_table(table_ref)
            except NotFound:
                logger.warning(f"Table {table_ref} does not exist")
                return pd.DataFrame()

            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices`
            """

            # Add date filters
            where_clauses = []

            if start_date:
                start_date_str = start_date.strftime('%Y-%m-%d')
                where_clauses.append(f"date >= '{start_date_str}'")
            elif days:
                where_clauses.append(f"date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)")

            if end_date:
                end_date_str = end_date.strftime('%Y-%m-%d')
                where_clauses.append(f"date <= '{end_date_str}'")

            if where_clauses:
                query += "\nWHERE " + " AND ".join(where_clauses)

            query += """
            ORDER BY date DESC
            """

            # Execute query with error handling
            try:
                df = self.client.query(query).to_dataframe()
            except Exception as query_error:
                logger.error(f"Query error for {ticker}: {str(query_error)}")
                # Try a simpler query if the first one fails
                try:
                    simple_query = f"""
                    SELECT *
                    FROM `{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices`
                    ORDER BY date DESC
                    LIMIT 100
                    """
                    df = self.client.query(simple_query).to_dataframe()
                    logger.info(f"Retrieved data with simplified query for {ticker}")
                except Exception as simple_query_error:
                    logger.error(f"Simple query also failed for {ticker}: {str(simple_query_error)}")
                    return pd.DataFrame()

            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Ensure numeric columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for NaN values
            nan_counts = df[numeric_columns].isna().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"NaN values found in retrieved data for {ticker}: {nan_counts}")

            logger.info(f"Retrieved {len(df)} rows of price data for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving stock prices for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_volume_analysis(self, ticker: str, days: int = None) -> pd.DataFrame:
        """Get volume analysis results from BigQuery.

        Args:
            ticker: Stock symbol
            days: Number of days of data to retrieve (None for all data)

        Returns:
            DataFrame with volume analysis results
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.volume_{ticker.lower()}_analysis`
            """

            if days:
                query += f"""
                WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
                """

            query += """
            ORDER BY date DESC
            """

            # Execute query
            df = self.client.query(query).to_dataframe()

            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Retrieved {len(df)} rows of volume analysis for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving volume analysis for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_master_summary(self, days: int = None) -> pd.DataFrame:
        """Get master summary from BigQuery.

        Args:
            days: Number of days of data to retrieve (None for all data)

        Returns:
            DataFrame with master summary
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.master_summary`
            """

            if days:
                query += f"""
                WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
                """

            query += """
            ORDER BY date DESC, confidence DESC
            """

            # Execute query
            df = self.client.query(query).to_dataframe()

            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            logger.info(f"Retrieved {len(df)} rows of master summary")
            return df

        except Exception as e:
            logger.error(f"Error retrieving master summary: {str(e)}")
            return pd.DataFrame()

    def get_trade_logs(self, symbol: Optional[str] = None, status: Optional[str] = None) -> pd.DataFrame:
        """Get trade logs from BigQuery.

        Args:
            symbol: Filter by stock symbol (None for all symbols)
            status: Filter by trade status (None for all statuses)

        Returns:
            DataFrame with trade logs
        """
        try:
            # Build query
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.trade_logs`
            """

            where_clauses = []

            if symbol:
                where_clauses.append(f"symbol = '{symbol}'")

            if status:
                where_clauses.append(f"status = '{status}'")

            if where_clauses:
                query += "WHERE " + " AND ".join(where_clauses)

            query += """
            ORDER BY created_at DESC
            """

            # Execute query
            df = self.client.query(query).to_dataframe()

            # Convert date columns to datetime
            for date_col in ['entry_date', 'exit_date']:
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

            logger.info(f"Retrieved {len(df)} trade logs")
            return df

        except Exception as e:
            logger.error(f"Error retrieving trade logs: {str(e)}")
            return pd.DataFrame()

    def archive_ticker_data(self, ticker: str) -> bool:
        """Archive data for a ticker that is no longer tracked.

        Args:
            ticker: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        try:
            # Archive price data
            price_table_id = f"{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices"
            archive_price_table_id = f"{self.project_id}.{self.dataset_id}.archived_stock_{ticker.lower()}_prices"

            # Check if price table exists
            try:
                self.client.get_table(price_table_id)

                # Copy data to archive table
                query = f"""
                CREATE OR REPLACE TABLE `{archive_price_table_id}`
                AS SELECT * FROM `{price_table_id}`
                """

                self.client.query(query).result()
                logger.info(f"Archived price data for {ticker}")

            except NotFound:
                logger.info(f"Price table for {ticker} does not exist, nothing to archive")

            # Archive volume analysis data
            volume_table_id = f"{self.project_id}.{self.dataset_id}.volume_{ticker.lower()}_analysis"
            archive_volume_table_id = f"{self.project_id}.{self.dataset_id}.archived_volume_{ticker.lower()}_analysis"

            # Check if volume analysis table exists
            try:
                self.client.get_table(volume_table_id)

                # Copy data to archive table
                query = f"""
                CREATE OR REPLACE TABLE `{archive_volume_table_id}`
                AS SELECT * FROM `{volume_table_id}`
                """

                self.client.query(query).result()
                logger.info(f"Archived volume analysis data for {ticker}")

            except NotFound:
                logger.info(f"Volume analysis table for {ticker} does not exist, nothing to archive")

            return True

        except Exception as e:
            logger.error(f"Error archiving data for {ticker}: {str(e)}")
            return False
