"""
Data repair tool for fixing inconsistencies in BigQuery data.
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
        logging.FileHandler("repair_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataRepairTool:
    """Tool for repairing data inconsistencies in BigQuery."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "trading_insights"):
        """Initialize the data repair tool.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        from trading_ai.config import config_manager
        
        self.project_id = project_id or config_manager.system_config.google_cloud_project
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)
        
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            table_name: Table name
            
        Returns:
            True if the table exists, False otherwise
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        try:
            self.client.get_table(table_id)
            return True
        except NotFound:
            return False
    
    def get_date_range(self, ticker: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Tuple of (start_date, end_date)
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return None, None
        
        query = f"""
        SELECT MIN(date) as start_date, MAX(date) as end_date
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        """
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                return row.start_date, row.end_date
            
            return None, None
        except Exception as e:
            logger.error(f"Error getting date range for {ticker}: {str(e)}")
            return None, None
    
    def find_missing_dates(self, ticker: str, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> List[datetime]:
        """Find missing dates for a ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            List of missing dates
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return []
        
        # Get date range if not provided
        if start_date is None or end_date is None:
            table_start, table_end = self.get_date_range(ticker)
            start_date = start_date or table_start
            end_date = end_date or table_end
        
        if start_date is None or end_date is None:
            logger.warning(f"Could not determine date range for {ticker}")
            return []
        
        # Get all dates in the table
        query = f"""
        SELECT date
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        WHERE date BETWEEN @start_date AND @end_date
        ORDER BY date
        """
        
        query_params = [
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date)
        ]
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        
        try:
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert to list of dates
            existing_dates = [row.date for row in results]
            
            # Generate all dates in range
            all_dates = []
            current_date = start_date
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                    all_dates.append(current_date)
                current_date += timedelta(days=1)
            
            # Find missing dates
            missing_dates = [date for date in all_dates if date not in existing_dates]
            
            return missing_dates
        except Exception as e:
            logger.error(f"Error finding missing dates for {ticker}: {str(e)}")
            return []
    
    def find_duplicate_dates(self, ticker: str) -> Dict[datetime, int]:
        """Find duplicate dates for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary of date to count
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return {}
        
        query = f"""
        SELECT date, COUNT(*) as count
        FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        GROUP BY date
        HAVING COUNT(*) > 1
        ORDER BY date
        """
        
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            
            # Convert to dictionary
            duplicates = {row.date: row.count for row in results}
            
            return duplicates
        except Exception as e:
            logger.error(f"Error finding duplicate dates for {ticker}: {str(e)}")
            return {}
    
    def find_null_values(self, ticker: str) -> Dict[str, int]:
        """Find null values for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary of column to null count
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return {}
        
        # Get table schema
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        table = self.client.get_table(table_id)
        columns = [field.name for field in table.schema]
        
        # Check each column for nulls
        null_counts = {}
        for column in columns:
            query = f"""
            SELECT COUNT(*) as count
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            WHERE {column} IS NULL
            """
            
            try:
                query_job = self.client.query(query)
                results = query_job.result()
                
                for row in results:
                    if row.count > 0:
                        null_counts[column] = row.count
                
            except Exception as e:
                logger.error(f"Error checking nulls in {column} for {ticker}: {str(e)}")
        
        return null_counts
    
    def remove_duplicates(self, ticker: str) -> bool:
        """Remove duplicate dates for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return False
        
        # Create a temporary table with deduplicated data
        temp_table = f"{table_name}_dedup"
        
        query = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{temp_table}` AS
        SELECT * EXCEPT(row_num)
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY date ORDER BY date) as row_num
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
        )
        WHERE row_num = 1
        """
        
        try:
            # Create deduplicated table
            query_job = self.client.query(query)
            query_job.result()
            
            # Swap tables
            swap_query = f"""
            DROP TABLE IF EXISTS `{self.project_id}.{self.dataset_id}.{table_name}`;
            ALTER TABLE `{self.project_id}.{self.dataset_id}.{temp_table}` 
            RENAME TO `{self.project_id}.{self.dataset_id}.{table_name}`;
            """
            
            swap_job = self.client.query(swap_query)
            swap_job.result()
            
            logger.info(f"Successfully removed duplicates for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error removing duplicates for {ticker}: {str(e)}")
            return False
    
    def fill_missing_values(self, ticker: str) -> bool:
        """Fill missing values for a ticker using interpolation.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        table_name = f"stock_{ticker.lower()}_prices"
        if not self.check_table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return False
        
        try:
            # Get data
            query = f"""
            SELECT *
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            ORDER BY date
            """
            
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return False
            
            # Check for nulls
            null_columns = [col for col in df.columns if df[col].isnull().any()]
            if not null_columns:
                logger.info(f"No missing values to fill for {ticker}")
                return True
            
            # Fill missing values
            for col in null_columns:
                if col != 'date' and col != 'symbol':
                    # Use interpolation for numeric columns
                    df[col] = df[col].interpolate(method='linear')
            
            # Upload back to BigQuery
            temp_table = f"{table_name}_filled"
            df.to_gbq(
                destination_table=f"{self.dataset_id}.{temp_table}",
                project_id=self.project_id,
                if_exists='replace'
            )
            
            # Swap tables
            swap_query = f"""
            DROP TABLE IF EXISTS `{self.project_id}.{self.dataset_id}.{table_name}`;
            ALTER TABLE `{self.project_id}.{self.dataset_id}.{temp_table}` 
            RENAME TO `{self.project_id}.{self.dataset_id}.{table_name}`;
            """
            
            swap_job = self.client.query(swap_query)
            swap_job.result()
            
            logger.info(f"Successfully filled missing values for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Error filling missing values for {ticker}: {str(e)}")
            return False
    
    def check_data_quality(self, ticker: str) -> Dict:
        """Check data quality for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with data quality metrics
        """
        # Get date range
        start_date, end_date = self.get_date_range(ticker)
        
        # Find missing dates
        missing_dates = self.find_missing_dates(ticker, start_date, end_date)
        
        # Find duplicate dates
        duplicate_dates = self.find_duplicate_dates(ticker)
        
        # Find null values
        null_values = self.find_null_values(ticker)
        
        # Calculate trading days
        trading_days = 0
        if start_date and end_date:
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                    trading_days += 1
                current_date += timedelta(days=1)
        
        # Calculate expected vs actual days
        table_name = f"stock_{ticker.lower()}_prices"
        row_count = 0
        if self.check_table_exists(table_name):
            query = f"""
            SELECT COUNT(*) as count
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            """
            
            query_job = self.client.query(query)
            results = query_job.result()
            
            for row in results:
                row_count = row.count
        
        # Calculate completeness
        completeness = 0
        if trading_days > 0:
            completeness = (trading_days - len(missing_dates)) / trading_days * 100
        
        return {
            'ticker': ticker,
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'trading_days': trading_days,
            'row_count': row_count,
            'missing_dates_count': len(missing_dates),
            'missing_dates': [d.isoformat() for d in missing_dates[:10]],  # Show first 10
            'duplicate_dates_count': len(duplicate_dates),
            'duplicate_dates': {d.isoformat(): count for d, count in list(duplicate_dates.items())[:10]},
            'null_values': null_values,
            'completeness': completeness
        }
    
    def repair_ticker_data(self, ticker: str) -> bool:
        """Repair data for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Repairing data for {ticker}")
        
        # Check data quality
        quality = self.check_data_quality(ticker)
        
        # Remove duplicates if needed
        if quality['duplicate_dates_count'] > 0:
            logger.info(f"Removing {quality['duplicate_dates_count']} duplicates for {ticker}")
            self.remove_duplicates(ticker)
        
        # Fill missing values if needed
        if quality['null_values']:
            logger.info(f"Filling missing values for {ticker}")
            self.fill_missing_values(ticker)
        
        # Check data quality again
        new_quality = self.check_data_quality(ticker)
        
        # Log improvements
        logger.info(f"Data quality for {ticker}:")
        logger.info(f"  - Duplicates: {quality['duplicate_dates_count']} -> {new_quality['duplicate_dates_count']}")
        logger.info(f"  - Null values: {sum(quality['null_values'].values()) if quality['null_values'] else 0} -> {sum(new_quality['null_values'].values()) if new_quality['null_values'] else 0}")
        logger.info(f"  - Completeness: {quality['completeness']:.2f}% -> {new_quality['completeness']:.2f}%")
        
        return True
    
    def repair_all_tickers(self, tickers: List[str]) -> Dict[str, bool]:
        """Repair data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary of ticker to success status
        """
        results = {}
        
        for ticker in tickers:
            try:
                success = self.repair_ticker_data(ticker)
                results[ticker] = success
            except Exception as e:
                logger.error(f"Error repairing data for {ticker}: {str(e)}")
                results[ticker] = False
        
        # Generate summary
        success_count = sum(1 for status in results.values() if status)
        logger.info(f"Repair summary: {success_count}/{len(tickers)} tickers repaired successfully")
        
        return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Repair data inconsistencies in BigQuery")
    parser.add_argument("--ticker", type=str, help="Repair a specific ticker")
    parser.add_argument("--check", action="store_true", help="Check data quality without repairing")
    parser.add_argument("--all", action="store_true", help="Repair all tickers")
    args = parser.parse_args()
    
    # Initialize repair tool
    repair_tool = DataRepairTool()
    
    if args.ticker and args.check:
        # Check data quality for a specific ticker
        quality = repair_tool.check_data_quality(args.ticker)
        print(f"Data quality for {args.ticker}:")
        print(f"  - Date range: {quality['start_date']} to {quality['end_date']}")
        print(f"  - Trading days: {quality['trading_days']}")
        print(f"  - Row count: {quality['row_count']}")
        print(f"  - Missing dates: {quality['missing_dates_count']}")
        print(f"  - Duplicate dates: {quality['duplicate_dates_count']}")
        print(f"  - Null values: {sum(quality['null_values'].values()) if quality['null_values'] else 0}")
        print(f"  - Completeness: {quality['completeness']:.2f}%")
    elif args.ticker:
        # Repair a specific ticker
        repair_tool.repair_ticker_data(args.ticker)
    elif args.all:
        # Repair all tickers
        from trading_ai.config import config_manager
        all_tickers = config_manager.get_all_tickers()
        repair_tool.repair_all_tickers(all_tickers)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
