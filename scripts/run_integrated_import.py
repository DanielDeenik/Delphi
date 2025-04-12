#!/usr/bin/env python3
"""
Integrated Import Script for Delphi

This script integrates with the existing codebase to provide a unified
interface for importing data to BigQuery. It uses the existing import manager,
data repair tool, and storage optimizer.
"""
import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"integrated_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Integrated Import Script for Delphi")
    
    # Import options
    parser.add_argument("--ticker", type=str, help="Import a specific ticker")
    parser.add_argument("--force-full", action="store_true", help="Force a full data refresh")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of stocks to process in each batch")
    parser.add_argument("--retries", type=int, default=3, help="Maximum number of retries for failed imports")
    parser.add_argument("--retry-delay", type=int, default=5, help="Delay between retries in seconds")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed imports")
    
    # Data repair options
    parser.add_argument("--repair", action="store_true", help="Repair data inconsistencies")
    parser.add_argument("--check-quality", action="store_true", help="Check data quality without repairing")
    
    # Storage optimization options
    parser.add_argument("--optimize", action="store_true", help="Optimize storage")
    parser.add_argument("--optimize-schema", action="store_true", help="Optimize table schemas")
    parser.add_argument("--compress-history", action="store_true", help="Compress historical data")
    parser.add_argument("--compress-days", type=int, default=365, help="Compress data older than this many days")
    
    # Reporting options
    parser.add_argument("--report", action="store_true", help="Generate import report")
    
    return parser.parse_args()

def run_import(args):
    """Run the import process."""
    try:
        # Import required modules
        from import_manager import ImportManager
        
        # Initialize import manager
        import_manager = ImportManager(max_retries=args.retries, retry_delay=args.retry_delay)
        
        # Get tickers to import
        if args.ticker:
            tickers = [args.ticker]
        else:
            from trading_ai.config import config_manager
            tickers = config_manager.get_all_tickers()
        
        # Import data
        logger.info(f"Starting import for {len(tickers)} tickers")
        results = import_manager.import_all_tickers(tickers, args.batch_size, args.force_full)
        
        # Check for failed imports
        failed_tickers = [ticker for ticker, success in results.items() if not success]
        if failed_tickers:
            logger.warning(f"{len(failed_tickers)} tickers failed to import: {', '.join(failed_tickers)}")
            
            if args.retry_failed:
                logger.info("Retrying failed imports...")
                retry_results = import_manager.retry_failed_imports(args.batch_size)
                
                # Update results
                results.update(retry_results)
                
                # Check for still failed imports
                still_failed = [ticker for ticker, success in retry_results.items() if not success]
                if still_failed:
                    logger.warning(f"{len(still_failed)} tickers still failed after retry: {', '.join(still_failed)}")
        
        # Generate report
        if args.report:
            report = import_manager.generate_import_report()
            logger.info(f"Import report: {report['status_counts']['success']}/{report['total_tickers']} tickers imported successfully")
        
        return len(failed_tickers) == 0
        
    except Exception as e:
        logger.error(f"Error running import: {str(e)}")
        return False

def run_data_repair(args):
    """Run data repair process."""
    try:
        from data_repair import DataRepairTool
        
        # Initialize repair tool
        repair_tool = DataRepairTool()
        
        # Get tickers to repair
        if args.ticker:
            tickers = [args.ticker]
        else:
            from trading_ai.config import config_manager
            tickers = config_manager.get_all_tickers()
        
        if args.check_quality:
            # Check data quality without repairing
            logger.info(f"Checking data quality for {len(tickers)} tickers")
            for ticker in tickers:
                quality = repair_tool.check_data_quality(ticker)
                logger.info(f"Data quality for {ticker}:")
                logger.info(f"  - Date range: {quality['start_date']} to {quality['end_date']}")
                logger.info(f"  - Trading days: {quality['trading_days']}")
                logger.info(f"  - Row count: {quality['row_count']}")
                logger.info(f"  - Missing dates: {quality['missing_dates_count']}")
                logger.info(f"  - Duplicate dates: {quality['duplicate_dates_count']}")
                logger.info(f"  - Null values: {sum(quality['null_values'].values()) if quality['null_values'] else 0}")
                logger.info(f"  - Completeness: {quality['completeness']:.2f}%")
            return True
        else:
            # Repair data
            logger.info(f"Starting data repair for {len(tickers)} tickers")
            results = repair_tool.repair_all_tickers(tickers)
            
            # Check for failed repairs
            failed_tickers = [ticker for ticker, success in results.items() if not success]
            if failed_tickers:
                logger.warning(f"{len(failed_tickers)} tickers failed to repair: {', '.join(failed_tickers)}")
            
            return len(failed_tickers) == 0
        
    except Exception as e:
        logger.error(f"Error running data repair: {str(e)}")
        return False

def run_storage_optimization(args):
    """Run storage optimization process."""
    try:
        from storage_optimizer import StorageOptimizer
        
        # Initialize optimizer
        optimizer = StorageOptimizer()
        
        # Get storage usage before optimization
        before_usage = optimizer.get_storage_usage()
        logger.info(f"Storage usage before optimization: {before_usage['total_size_mb']:.2f} MB")
        
        # Optimize tables
        if args.optimize_schema:
            logger.info("Optimizing table schemas...")
            optimizer.optimize_all_tables()
        
        # Compress historical data
        if args.compress_history:
            logger.info(f"Compressing historical data older than {args.compress_days} days...")
            optimizer.compress_all_historical_data(args.compress_days)
        
        # Get storage usage after optimization
        after_usage = optimizer.get_storage_usage()
        logger.info(f"Storage usage after optimization: {after_usage['total_size_mb']:.2f} MB")
        
        # Calculate savings
        savings = before_usage['total_size_mb'] - after_usage['total_size_mb']
        savings_percent = (savings / before_usage['total_size_mb'] * 100) if before_usage['total_size_mb'] > 0 else 0
        logger.info(f"Storage savings: {savings:.2f} MB ({savings_percent:.2f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running storage optimization: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Set default actions if none specified
    if not any([args.ticker, args.force_full, args.retry_failed, args.repair, args.optimize, 
                args.optimize_schema, args.compress_history, args.check_quality, args.report]):
        args.retry_failed = True
        args.repair = True
        args.optimize_schema = True
    
    # Run import
    if args.ticker or args.force_full or args.retry_failed or args.report or not any([args.repair, args.optimize, 
                                                                                    args.optimize_schema, args.compress_history, args.check_quality]):
        logger.info("Running data import...")
        if not run_import(args):
            logger.error("Data import failed")
            return 1
    
    # Run data repair
    if args.repair or args.check_quality:
        logger.info("Running data repair...")
        if not run_data_repair(args):
            logger.warning("Data repair had some issues")
    
    # Run storage optimization
    if args.optimize or args.optimize_schema or args.compress_history:
        logger.info("Running storage optimization...")
        if not run_storage_optimization(args):
            logger.warning("Storage optimization had some issues")
    
    logger.info("All processes completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
