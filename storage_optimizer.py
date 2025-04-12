"""
BigQuery storage optimizer for time series data.
"""
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("optimizer_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StorageOptimizer:
    """Tool for optimizing BigQuery storage for time series data."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "trading_insights"):
        """Initialize the storage optimizer.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        from trading_ai.config import config_manager
        
        self.project_id = project_id or config_manager.system_config.google_cloud_project
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Dictionary with table information
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        try:
            table = self.client.get_table(table_id)
            
            return {
                'table_id': table_id,
                'num_rows': table.num_rows,
                'size_bytes': table.num_bytes,
                'size_mb': table.num_bytes / 1024 / 1024,
                'created': table.created.isoformat(),
                'modified': table.modified.isoformat(),
                'schema': [field.name for field in table.schema],
                'partitioning': table.time_partitioning.type_ if table.time_partitioning else None,
                'partition_field': table.time_partitioning.field if table.time_partitioning else None,
                'clustering_fields': table.clustering_fields
            }
        except NotFound:
            logger.warning(f"Table {table_id} not found")
            return {}
    
    def get_all_tables(self) -> List[str]:
        """Get all tables in the dataset.
        
        Returns:
            List of table names
        """
        dataset_ref = self.client.dataset(self.dataset_id)
        tables = list(self.client.list_tables(dataset_ref))
        return [table.table_id for table in tables]
    
    def get_storage_usage(self) -> Dict:
        """Get storage usage for all tables.
        
        Returns:
            Dictionary with storage usage information
        """
        tables = self.get_all_tables()
        
        # Get info for each table
        table_info = {}
        total_size_bytes = 0
        total_rows = 0
        
        for table_name in tables:
            info = self.get_table_info(table_name)
            if info:
                table_info[table_name] = info
                total_size_bytes += info['size_bytes']
                total_rows += info['num_rows']
        
        # Calculate summary
        return {
            'tables': table_info,
            'total_tables': len(table_info),
            'total_size_bytes': total_size_bytes,
            'total_size_mb': total_size_bytes / 1024 / 1024,
            'total_rows': total_rows
        }
    
    def optimize_table_schema(self, table_name: str) -> bool:
        """Optimize the schema of a table.
        
        Args:
            table_name: Table name
            
        Returns:
            True if successful, False otherwise
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        try:
            # Get current table info
            table = self.client.get_table(table_id)
            
            # Check if table is already optimized
            is_partitioned = table.time_partitioning is not None
            is_clustered = table.clustering_fields is not None
            
            if is_partitioned and is_clustered:
                logger.info(f"Table {table_name} is already optimized")
                return True
            
            # Determine if this is a stock price table
            is_stock_price = table_name.startswith("stock_") and table_name.endswith("_prices")
            is_volume_analysis = table_name.startswith("volume_") and table_name.endswith("_analysis")
            is_master_summary = table_name == "master_summary"
            
            # Create optimized schema based on table type
            if is_stock_price:
                # Get ticker from table name
                ticker = table_name.replace("stock_", "").replace("_prices", "")
                
                # Create optimized schema
                schema = [
                    # Primary dimension for partitioning
                    bigquery.SchemaField("date", "DATE", mode="REQUIRED",
                                        description="Trading date"),
                    # Clustering fields
                    bigquery.SchemaField("symbol", "STRING",
                                        description="Stock symbol"),
                    # Core metrics
                    bigquery.SchemaField("close", "FLOAT64",
                                        description="Closing price"),
                    bigquery.SchemaField("adjusted_close", "FLOAT64",
                                        description="Adjusted closing price"),
                    bigquery.SchemaField("volume", "INTEGER",
                                        description="Trading volume"),
                    # Secondary metrics
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
                
                # Create new table
                new_table = bigquery.Table(table_id, schema=schema)
                new_table.description = f"Daily stock price data for {ticker}"
                new_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
                new_table.clustering_fields = ["symbol"]
                
            elif is_volume_analysis:
                # Get ticker from table name
                ticker = table_name.replace("volume_", "").replace("_analysis", "")
                
                # Create optimized schema
                schema = [
                    # Primary dimension for partitioning
                    bigquery.SchemaField("date", "DATE", mode="REQUIRED",
                                        description="Trading date"),
                    # Clustering fields
                    bigquery.SchemaField("symbol", "STRING",
                                        description="Stock symbol"),
                    bigquery.SchemaField("signal", "STRING",
                                        description="Trading signal"),
                    # Key metrics
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
                    # Volume metrics
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
                
                # Create new table
                new_table = bigquery.Table(table_id, schema=schema)
                new_table.description = f"Volume analysis results for {ticker}"
                new_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
                new_table.clustering_fields = ["symbol", "signal", "is_volume_spike"]
                
            elif is_master_summary:
                # Create optimized schema
                schema = [
                    # Primary dimension for partitioning
                    bigquery.SchemaField("date", "DATE", mode="REQUIRED",
                                        description="Trading date"),
                    # Clustering fields
                    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED",
                                        description="Stock symbol"),
                    bigquery.SchemaField("direction", "STRING",
                                        description="Trading direction (buy or short)"),
                    bigquery.SchemaField("signal", "STRING",
                                        description="Trading signal"),
                    # Key metrics
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
                
                # Create new table
                new_table = bigquery.Table(table_id, schema=schema)
                new_table.description = "Master summary of volume analysis across all stocks"
                new_table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="date"
                )
                new_table.clustering_fields = ["symbol", "signal", "direction"]
            
            else:
                logger.warning(f"Table {table_name} is not a recognized type, skipping optimization")
                return False
            
            # Copy data to new table
            temp_table_id = f"{self.project_id}.{self.dataset_id}.{table_name}_optimized"
            
            # Create the new table
            self.client.create_table(new_table)
            
            # Copy data
            copy_query = f"""
            INSERT INTO `{table_id}`
            SELECT * FROM `{table_id}`
            """
            
            copy_job = self.client.query(copy_query)
            copy_job.result()
            
            logger.info(f"Successfully optimized table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing table {table_name}: {str(e)}")
            return False
    
    def optimize_all_tables(self) -> Dict[str, bool]:
        """Optimize all tables in the dataset.
        
        Returns:
            Dictionary of table name to success status
        """
        tables = self.get_all_tables()
        results = {}
        
        for table_name in tables:
            try:
                success = self.optimize_table_schema(table_name)
                results[table_name] = success
            except Exception as e:
                logger.error(f"Error optimizing table {table_name}: {str(e)}")
                results[table_name] = False
        
        # Generate summary
        success_count = sum(1 for status in results.values() if status)
        logger.info(f"Optimization summary: {success_count}/{len(tables)} tables optimized successfully")
        
        return results
    
    def compress_historical_data(self, ticker: str, older_than_days: int = 365) -> bool:
        """Compress historical data for a ticker.
        
        Args:
            ticker: Stock symbol
            older_than_days: Compress data older than this many days
            
        Returns:
            True if successful, False otherwise
        """
        table_name = f"stock_{ticker.lower()}_prices"
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=older_than_days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            # Create compressed table
            compressed_table = f"{table_name}_compressed"
            compressed_table_id = f"{self.project_id}.{self.dataset_id}.{compressed_table}"
            
            # Create weekly aggregation for old data
            compress_query = f"""
            CREATE OR REPLACE TABLE `{compressed_table_id}` AS
            
            -- Recent data (daily)
            SELECT *
            FROM `{table_id}`
            WHERE date >= '{cutoff_str}'
            
            UNION ALL
            
            -- Historical data (weekly)
            SELECT
                DATE_TRUNC(date, WEEK) as date,
                symbol,
                AVG(open) as open,
                MAX(high) as high,
                MIN(low) as low,
                AVG(close) as close,
                AVG(adjusted_close) as adjusted_close,
                SUM(volume) as volume,
                SUM(dividend) as dividend,
                MAX(split_coefficient) as split_coefficient
            FROM `{table_id}`
            WHERE date < '{cutoff_str}'
            GROUP BY DATE_TRUNC(date, WEEK), symbol
            """
            
            compress_job = self.client.query(compress_query)
            compress_job.result()
            
            # Get table info before and after
            before_info = self.get_table_info(table_name)
            after_info = self.get_table_info(compressed_table)
            
            # Calculate savings
            size_before = before_info.get('size_bytes', 0)
            size_after = after_info.get('size_bytes', 0)
            savings = size_before - size_after
            savings_percent = (savings / size_before * 100) if size_before > 0 else 0
            
            logger.info(f"Compression for {ticker}:")
            logger.info(f"  - Size before: {size_before / 1024 / 1024:.2f} MB")
            logger.info(f"  - Size after: {size_after / 1024 / 1024:.2f} MB")
            logger.info(f"  - Savings: {savings / 1024 / 1024:.2f} MB ({savings_percent:.2f}%)")
            
            # Swap tables if compression was successful
            if size_after > 0 and savings_percent > 5:
                swap_query = f"""
                DROP TABLE IF EXISTS `{table_id}`;
                ALTER TABLE `{compressed_table_id}` RENAME TO `{table_id}`;
                """
                
                swap_job = self.client.query(swap_query)
                swap_job.result()
                
                logger.info(f"Successfully compressed historical data for {ticker}")
                return True
            else:
                logger.info(f"Compression did not yield significant savings for {ticker}, keeping original data")
                
                # Clean up compressed table
                self.client.delete_table(compressed_table_id)
                return False
            
        except Exception as e:
            logger.error(f"Error compressing historical data for {ticker}: {str(e)}")
            return False
    
    def compress_all_historical_data(self, older_than_days: int = 365) -> Dict[str, bool]:
        """Compress historical data for all tickers.
        
        Args:
            older_than_days: Compress data older than this many days
            
        Returns:
            Dictionary of ticker to success status
        """
        # Get all stock price tables
        tables = self.get_all_tables()
        stock_tables = [t for t in tables if t.startswith("stock_") and t.endswith("_prices")]
        
        results = {}
        total_savings = 0
        
        for table_name in stock_tables:
            # Extract ticker from table name
            ticker = table_name.replace("stock_", "").replace("_prices", "")
            
            try:
                # Get table info before compression
                before_info = self.get_table_info(table_name)
                size_before = before_info.get('size_bytes', 0)
                
                # Compress historical data
                success = self.compress_historical_data(ticker, older_than_days)
                results[ticker] = success
                
                # Calculate savings if successful
                if success:
                    after_info = self.get_table_info(table_name)
                    size_after = after_info.get('size_bytes', 0)
                    savings = size_before - size_after
                    total_savings += savings
            except Exception as e:
                logger.error(f"Error compressing historical data for {ticker}: {str(e)}")
                results[ticker] = False
        
        # Generate summary
        success_count = sum(1 for status in results.values() if status)
        logger.info(f"Compression summary: {success_count}/{len(stock_tables)} tables compressed successfully")
        logger.info(f"Total savings: {total_savings / 1024 / 1024:.2f} MB")
        
        return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize BigQuery storage for time series data")
    parser.add_argument("--info", action="store_true", help="Get storage usage information")
    parser.add_argument("--optimize", action="store_true", help="Optimize all tables")
    parser.add_argument("--compress", action="store_true", help="Compress historical data")
    parser.add_argument("--days", type=int, default=365, help="Compress data older than this many days")
    parser.add_argument("--ticker", type=str, help="Optimize or compress a specific ticker")
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = StorageOptimizer()
    
    if args.info:
        # Get storage usage information
        usage = optimizer.get_storage_usage()
        print(f"Storage usage:")
        print(f"  - Total tables: {usage['total_tables']}")
        print(f"  - Total size: {usage['total_size_mb']:.2f} MB")
        print(f"  - Total rows: {usage['total_rows']}")
        
        # Show top 5 tables by size
        tables_by_size = sorted(usage['tables'].items(), key=lambda x: x[1]['size_bytes'], reverse=True)
        print("\nTop tables by size:")
        for table_name, info in tables_by_size[:5]:
            print(f"  - {table_name}: {info['size_mb']:.2f} MB ({info['num_rows']} rows)")
    
    elif args.optimize and args.ticker:
        # Optimize a specific table
        if args.ticker.startswith("stock_") or args.ticker.startswith("volume_") or args.ticker == "master_summary":
            optimizer.optimize_table_schema(args.ticker)
        else:
            # Assume it's a ticker symbol
            optimizer.optimize_table_schema(f"stock_{args.ticker.lower()}_prices")
            optimizer.optimize_table_schema(f"volume_{args.ticker.lower()}_analysis")
    
    elif args.optimize:
        # Optimize all tables
        optimizer.optimize_all_tables()
    
    elif args.compress and args.ticker:
        # Compress historical data for a specific ticker
        optimizer.compress_historical_data(args.ticker, args.days)
    
    elif args.compress:
        # Compress historical data for all tickers
        optimizer.compress_all_historical_data(args.days)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
