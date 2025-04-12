"""
Data reconciliation tool for ensuring data completeness and consistency in BigQuery.

This tool provides functionality to:
1. Check for missing dates in time series data
2. Validate data quality and consistency
3. Compare data with external sources
4. Generate reconciliation reports
5. Trigger reimports for missing or inconsistent data
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas_market_calendars as mcal

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"reconciliation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataReconciliationTool:
    """Tool for reconciling data in BigQuery."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: str = "trading_insights"):
        """Initialize the data reconciliation tool.
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        # Get project ID from environment if not provided
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("Google Cloud project ID not specified")
        
        self.dataset_id = dataset_id
        
        # Initialize BigQuery client
        self.client = bigquery.Client(project=self.project_id)
        
        # Create directory for reconciliation reports
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize NYSE calendar for trading days
        self.nyse = mcal.get_calendar('NYSE')
        
        logger.info(f"Initialized data reconciliation tool for project {self.project_id}, dataset {self.dataset_id}")
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get list of trading days between start_date and end_date.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        # Get NYSE calendar for date range
        trading_days = self.nyse.valid_days(start_date=start_date, end_date=end_date)
        return trading_days.tolist()
    
    def get_stock_data(self, ticker: str, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get stock data from BigQuery.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            
        Returns:
            DataFrame with stock data
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Format dates for query
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Build query
            query = f"""
            SELECT * 
            FROM `{self.project_id}.{self.dataset_id}.stock_{ticker.lower()}_prices`
            WHERE date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY date
            """
            
            # Execute query
            df = self.client.query(query).to_dataframe()
            
            if df.empty:
                logger.warning(f"No data found for {ticker} in the specified date range")
            else:
                logger.info(f"Retrieved {len(df)} rows for {ticker}")
            
            return df
            
        except NotFound:
            logger.warning(f"Table for {ticker} not found")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def check_missing_dates(self, ticker: str, start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Check for missing dates in time series data.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            
        Returns:
            Dictionary with missing dates information
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Get stock data
            df = self.get_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No data found",
                    "missing_dates": [],
                    "missing_count": 0
                }
            
            # Get trading days
            trading_days = self.get_trading_days(start_date, end_date)
            
            # Convert dates to strings for comparison
            trading_days_str = [d.strftime('%Y-%m-%d') for d in trading_days]
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Convert to strings for comparison
                data_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            else:
                logger.warning(f"No date column found in data for {ticker}")
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No date column found",
                    "missing_dates": [],
                    "missing_count": 0
                }
            
            # Find missing dates
            missing_dates = [d for d in trading_days_str if d not in data_dates]
            
            # Create result
            result = {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "status": "success",
                "total_trading_days": len(trading_days),
                "available_days": len(data_dates),
                "missing_dates": missing_dates,
                "missing_count": len(missing_dates)
            }
            
            if missing_dates:
                logger.warning(f"Found {len(missing_dates)} missing dates for {ticker}")
                result["status"] = "incomplete"
            else:
                logger.info(f"No missing dates found for {ticker}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking missing dates for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
                "status": "error",
                "message": str(e),
                "missing_dates": [],
                "missing_count": 0
            }
    
    def validate_data_quality(self, ticker: str, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Validate data quality for a ticker.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            
        Returns:
            Dictionary with data quality information
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Get stock data
            df = self.get_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No data found",
                    "quality_score": 0
                }
            
            # Initialize quality checks
            quality_checks = {
                "has_required_columns": False,
                "no_null_values": False,
                "price_range_valid": False,
                "volume_valid": False,
                "date_sequence_valid": False,
                "no_duplicate_dates": False
            }
            
            # Check for required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            quality_checks["has_required_columns"] = all(col in df.columns for col in required_columns)
            
            if not quality_checks["has_required_columns"]:
                missing_columns = [col for col in required_columns if col not in df.columns]
                logger.warning(f"Missing required columns for {ticker}: {missing_columns}")
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": f"Missing required columns: {missing_columns}",
                    "quality_score": 0,
                    "quality_checks": quality_checks
                }
            
            # Check for null values
            null_counts = df[required_columns].isnull().sum().to_dict()
            quality_checks["no_null_values"] = all(count == 0 for count in null_counts.values())
            
            # Check price range validity
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Check if prices are positive
                price_positive = (
                    (df['open'] > 0).all() and
                    (df['high'] > 0).all() and
                    (df['low'] > 0).all() and
                    (df['close'] > 0).all()
                )
                
                # Check if high >= low
                high_gte_low = (df['high'] >= df['low']).all()
                
                # Check if high >= open and high >= close
                high_gte_open_close = (
                    (df['high'] >= df['open']).all() and
                    (df['high'] >= df['close']).all()
                )
                
                # Check if low <= open and low <= close
                low_lte_open_close = (
                    (df['low'] <= df['open']).all() and
                    (df['low'] <= df['close']).all()
                )
                
                quality_checks["price_range_valid"] = (
                    price_positive and
                    high_gte_low and
                    high_gte_open_close and
                    low_lte_open_close
                )
            
            # Check volume validity
            if 'volume' in df.columns:
                quality_checks["volume_valid"] = (df['volume'] >= 0).all()
            
            # Check date sequence validity
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Check if dates are sorted
                quality_checks["date_sequence_valid"] = df['date'].is_monotonic_increasing
                
                # Check for duplicate dates
                quality_checks["no_duplicate_dates"] = not df['date'].duplicated().any()
            
            # Calculate quality score (percentage of passed checks)
            quality_score = sum(1 for check in quality_checks.values() if check) / len(quality_checks) * 100
            
            # Create result
            result = {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "status": "success" if quality_score == 100 else "warning",
                "quality_score": quality_score,
                "quality_checks": quality_checks,
                "null_counts": null_counts if not quality_checks["no_null_values"] else {}
            }
            
            if quality_score < 100:
                logger.warning(f"Data quality issues found for {ticker}, score: {quality_score:.2f}%")
                failed_checks = [check for check, passed in quality_checks.items() if not passed]
                logger.warning(f"Failed checks: {failed_checks}")
            else:
                logger.info(f"Data quality validation passed for {ticker}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating data quality for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
                "status": "error",
                "message": str(e),
                "quality_score": 0
            }
    
    def compare_with_external_source(self, ticker: str, external_df: pd.DataFrame,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Compare BigQuery data with external source.
        
        Args:
            ticker: Stock symbol
            external_df: DataFrame with external data
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Get BigQuery data
            bq_df = self.get_stock_data(ticker, start_date, end_date)
            
            if bq_df.empty:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No data found in BigQuery",
                    "match_percentage": 0
                }
            
            if external_df.empty:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No data found in external source",
                    "match_percentage": 0
                }
            
            # Ensure date columns are datetime
            if 'date' in bq_df.columns and not pd.api.types.is_datetime64_dtype(bq_df['date']):
                bq_df['date'] = pd.to_datetime(bq_df['date'])
            
            if 'date' in external_df.columns and not pd.api.types.is_datetime64_dtype(external_df['date']):
                external_df['date'] = pd.to_datetime(external_df['date'])
            
            # Set date as index for both DataFrames
            if 'date' in bq_df.columns:
                bq_df = bq_df.set_index('date')
            
            if 'date' in external_df.columns:
                external_df = external_df.set_index('date')
            
            # Find common dates
            common_dates = bq_df.index.intersection(external_df.index)
            
            if len(common_dates) == 0:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No common dates found",
                    "match_percentage": 0
                }
            
            # Filter to common dates
            bq_df = bq_df.loc[common_dates]
            external_df = external_df.loc[common_dates]
            
            # Find common columns
            common_columns = [col for col in bq_df.columns if col in external_df.columns]
            
            if not common_columns:
                return {
                    "ticker": ticker,
                    "start_date": start_date.strftime('%Y-%m-%d'),
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "status": "error",
                    "message": "No common columns found",
                    "match_percentage": 0
                }
            
            # Compare data
            comparison_results = {}
            for col in common_columns:
                # Calculate percentage difference
                diff = (bq_df[col] - external_df[col]).abs()
                pct_diff = diff / external_df[col].abs().replace(0, np.nan)
                
                # Count matches (within 0.1% tolerance)
                tolerance = 0.001  # 0.1%
                matches = (pct_diff <= tolerance).sum()
                match_pct = matches / len(common_dates) * 100
                
                comparison_results[col] = {
                    "matches": int(matches),
                    "total": len(common_dates),
                    "match_percentage": float(match_pct),
                    "mean_difference": float(diff.mean()),
                    "max_difference": float(diff.max())
                }
            
            # Calculate overall match percentage
            overall_match_pct = sum(result["match_percentage"] for result in comparison_results.values()) / len(common_columns)
            
            # Create result
            result = {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "status": "success" if overall_match_pct >= 99 else "warning",
                "common_dates_count": len(common_dates),
                "common_columns": common_columns,
                "match_percentage": overall_match_pct,
                "column_results": comparison_results
            }
            
            if overall_match_pct < 99:
                logger.warning(f"Data discrepancies found for {ticker}, match: {overall_match_pct:.2f}%")
                for col, res in comparison_results.items():
                    if res["match_percentage"] < 99:
                        logger.warning(f"Column {col} match: {res['match_percentage']:.2f}%")
            else:
                logger.info(f"Data comparison passed for {ticker}, match: {overall_match_pct:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error comparing data for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
                "status": "error",
                "message": str(e),
                "match_percentage": 0
            }
    
    def generate_reconciliation_report(self, tickers: List[str], 
                                     start_date: Optional[datetime] = None,
                                     end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate reconciliation report for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (default: 90 days ago)
            end_date: End date (default: today)
            
        Returns:
            Dictionary with reconciliation report
        """
        try:
            # Set default date range if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=90)
            
            # Initialize report
            report = {
                "report_id": f"recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now().isoformat(),
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "tickers_count": len(tickers),
                "tickers": tickers,
                "summary": {
                    "complete": 0,
                    "incomplete": 0,
                    "error": 0,
                    "missing_dates_total": 0,
                    "quality_issues_total": 0
                },
                "ticker_results": {}
            }
            
            # Process each ticker
            for ticker in tickers:
                logger.info(f"Processing {ticker} for reconciliation report")
                
                # Check missing dates
                missing_dates_result = self.check_missing_dates(ticker, start_date, end_date)
                
                # Validate data quality
                quality_result = self.validate_data_quality(ticker, start_date, end_date)
                
                # Combine results
                ticker_result = {
                    "missing_dates": missing_dates_result,
                    "data_quality": quality_result
                }
                
                # Update summary
                if missing_dates_result["status"] == "error" or quality_result["status"] == "error":
                    report["summary"]["error"] += 1
                elif missing_dates_result["status"] == "incomplete" or quality_result["status"] == "warning":
                    report["summary"]["incomplete"] += 1
                else:
                    report["summary"]["complete"] += 1
                
                report["summary"]["missing_dates_total"] += missing_dates_result.get("missing_count", 0)
                
                if quality_result["status"] == "warning":
                    report["summary"]["quality_issues_total"] += 1
                
                # Add to ticker results
                report["ticker_results"][ticker] = ticker_result
            
            # Save report to file
            report_file = self.reports_dir / f"{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Reconciliation report generated: {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating reconciliation report: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_tickers_needing_reimport(self, report: Dict[str, Any], 
                                   missing_dates_threshold: int = 1,
                                   quality_score_threshold: float = 95.0) -> List[str]:
        """Get list of tickers that need reimport based on reconciliation report.
        
        Args:
            report: Reconciliation report
            missing_dates_threshold: Threshold for missing dates count
            quality_score_threshold: Threshold for quality score
            
        Returns:
            List of tickers needing reimport
        """
        tickers_to_reimport = []
        
        for ticker, result in report["ticker_results"].items():
            missing_dates = result["missing_dates"]
            data_quality = result["data_quality"]
            
            # Check if ticker needs reimport
            needs_reimport = False
            
            # Check missing dates
            if missing_dates["status"] == "incomplete" and missing_dates.get("missing_count", 0) >= missing_dates_threshold:
                needs_reimport = True
                logger.info(f"{ticker} needs reimport due to {missing_dates['missing_count']} missing dates")
            
            # Check data quality
            if data_quality["status"] == "warning" and data_quality.get("quality_score", 0) < quality_score_threshold:
                needs_reimport = True
                logger.info(f"{ticker} needs reimport due to quality score {data_quality['quality_score']:.2f}%")
            
            # Add to list if needed
            if needs_reimport:
                tickers_to_reimport.append(ticker)
        
        return tickers_to_reimport

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data Reconciliation Tool")
    
    parser.add_argument("--tickers", nargs="+", help="Tickers to reconcile")
    parser.add_argument("--project-id", help="Google Cloud project ID")
    parser.add_argument("--dataset-id", default="trading_insights", help="BigQuery dataset ID")
    parser.add_argument("--days", type=int, default=90, help="Number of days to check")
    parser.add_argument("--check-missing", action="store_true", help="Check for missing dates")
    parser.add_argument("--validate-quality", action="store_true", help="Validate data quality")
    parser.add_argument("--generate-report", action="store_true", help="Generate reconciliation report")
    parser.add_argument("--missing-threshold", type=int, default=1, help="Threshold for missing dates count")
    parser.add_argument("--quality-threshold", type=float, default=95.0, help="Threshold for quality score")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize reconciliation tool
        reconciliation_tool = DataReconciliationTool(
            project_id=args.project_id,
            dataset_id=args.dataset_id
        )
        
        # Get tickers from args or environment
        tickers = args.tickers
        if not tickers:
            # Try to get from environment or config
            try:
                from trading_ai.config import config_manager
                tickers = config_manager.get_all_tickers()
            except ImportError:
                logger.error("No tickers specified and could not import config_manager")
                return 1
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Check missing dates
        if args.check_missing:
            logger.info(f"Checking missing dates for {len(tickers)} tickers")
            for ticker in tickers:
                result = reconciliation_tool.check_missing_dates(ticker, start_date, end_date)
                if result["status"] == "incomplete":
                    logger.warning(f"{ticker}: {result['missing_count']} missing dates")
                elif result["status"] == "error":
                    logger.error(f"{ticker}: {result.get('message', 'Unknown error')}")
                else:
                    logger.info(f"{ticker}: No missing dates")
        
        # Validate data quality
        if args.validate_quality:
            logger.info(f"Validating data quality for {len(tickers)} tickers")
            for ticker in tickers:
                result = reconciliation_tool.validate_data_quality(ticker, start_date, end_date)
                if result["status"] == "warning":
                    logger.warning(f"{ticker}: Quality score {result['quality_score']:.2f}%")
                elif result["status"] == "error":
                    logger.error(f"{ticker}: {result.get('message', 'Unknown error')}")
                else:
                    logger.info(f"{ticker}: Quality score {result['quality_score']:.2f}%")
        
        # Generate reconciliation report
        if args.generate_report:
            logger.info(f"Generating reconciliation report for {len(tickers)} tickers")
            report = reconciliation_tool.generate_reconciliation_report(tickers, start_date, end_date)
            
            # Get tickers needing reimport
            tickers_to_reimport = reconciliation_tool.get_tickers_needing_reimport(
                report,
                missing_dates_threshold=args.missing_threshold,
                quality_score_threshold=args.quality_threshold
            )
            
            if tickers_to_reimport:
                logger.warning(f"{len(tickers_to_reimport)} tickers need reimport: {', '.join(tickers_to_reimport)}")
                
                # Save list of tickers to reimport
                reimport_file = Path("reimport_tickers.txt")
                with open(reimport_file, 'w') as f:
                    f.write('\n'.join(tickers_to_reimport))
                
                logger.info(f"Saved list of tickers to reimport to {reimport_file}")
            else:
                logger.info("No tickers need reimport")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
