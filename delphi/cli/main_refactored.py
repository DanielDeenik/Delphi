"""
Main CLI Entry Point (Refactored)

This module provides the main entry point for the CLI using the refactored codebase.
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from delphi.core.data.sources import create_data_source
from delphi.core.data.storage import create_storage_service
from delphi.core.data.repository import MarketDataRepository
from delphi.core.models.volume.analyzer import VolumeAnalyzer
from delphi.core.models.volume.strategies import SimpleVolumeStrategy
from delphi.utils.config import config_manager, load_env

# Configure logger
logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging.
    
    Args:
        log_level: Logging level
        log_file: Log file path (optional)
    """
    # Set up logging
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def import_data(args):
    """Import data from a data source to storage.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Create data source
        data_source = create_data_source(
            args.source,
            api_key=args.api_key
        )
        
        # Create storage service
        storage_service = create_storage_service(
            args.storage,
            project_id=args.project_id,
            dataset_id=args.dataset_id
        )
        
        # Create repository
        repository = MarketDataRepository(data_source, storage_service)
        
        # Get symbols
        symbols = args.symbols or config_manager.get_all_tickers()
        
        # Initialize tables
        if args.init_tables:
            storage_service.initialize_tables(symbols)
        
        # Import data
        logger.info(f"Importing data for {len(symbols)} symbols")
        
        results = repository.get_batch_stock_data(
            symbols,
            force_refresh=args.force_refresh,
            max_workers=args.max_workers
        )
        
        # Print summary
        success_count = sum(1 for df in results.values() if not df.empty)
        logger.info(f"Successfully imported data for {success_count}/{len(symbols)} symbols")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error importing data: {str(e)}")
        return 1

def analyze_data(args):
    """Analyze data.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Create storage service
        storage_service = create_storage_service(
            args.storage,
            project_id=args.project_id,
            dataset_id=args.dataset_id
        )
        
        # Create repository
        repository = MarketDataRepository(None, storage_service)
        
        # Create analyzer
        strategy = SimpleVolumeStrategy(
            z_score_threshold=args.z_score_threshold,
            lookback_period=args.lookback_period
        )
        
        analyzer = VolumeAnalyzer(strategy=strategy)
        
        # Get symbols
        symbols = args.symbols or config_manager.get_all_tickers()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Analyze data
        logger.info(f"Analyzing data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            # Get data
            df = repository.get_stock_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                continue
            
            # Analyze data
            analysis = analyzer.analyze(df)
            
            # Store results
            results[symbol] = analysis
            
            # Store volume analysis
            if args.store_analysis and 'analysis_df' in analysis:
                repository.store_volume_analysis(symbol, analysis['analysis_df'])
        
        # Print summary
        for symbol, analysis in results.items():
            if 'error' in analysis:
                logger.warning(f"Error analyzing {symbol}: {analysis['error']}")
                continue
            
            summary = analysis.get('summary', {})
            
            print(f"\nSummary for {symbol}:")
            print(f"  Total volume spikes: {summary.get('total_spikes', 0)}")
            print(f"  Bullish spikes: {summary.get('bullish_spikes', 0)}")
            print(f"  Bearish spikes: {summary.get('bearish_spikes', 0)}")
            print(f"  Average spike strength: {summary.get('avg_spike_strength', 0):.2f}")
            print(f"  Latest signal: {summary.get('latest_signal', 'NEUTRAL')}")
            print(f"  Latest signal strength: {summary.get('latest_signal_strength', 0):.2f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return 1

def main():
    """Main entry point for the CLI."""
    # Load environment variables
    load_env()
    
    # Create parser
    parser = argparse.ArgumentParser(description="Delphi - Financial Analysis Platform (Refactored)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data")
    import_parser.add_argument(
        "--source",
        choices=["alpha_vantage", "yfinance"],
        default="alpha_vantage",
        help="Data source (default: alpha_vantage)"
    )
    import_parser.add_argument(
        "--storage",
        choices=["bigquery", "sqlite"],
        default="bigquery",
        help="Storage service (default: bigquery)"
    )
    import_parser.add_argument(
        "--api-key",
        help="API key for the data source (default: from config)"
    )
    import_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to import (default: from config)"
    )
    import_parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    import_parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    import_parser.add_argument(
        "--init-tables",
        action="store_true",
        help="Initialize tables"
    )
    import_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh from data source"
    )
    import_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4)"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument(
        "--storage",
        choices=["bigquery", "sqlite"],
        default="bigquery",
        help="Storage service (default: bigquery)"
    )
    analyze_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to analyze (default: from config)"
    )
    analyze_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)"
    )
    analyze_parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    analyze_parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    analyze_parser.add_argument(
        "--z-score-threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for volume spikes (default: 2.0)"
    )
    analyze_parser.add_argument(
        "--lookback-period",
        type=int,
        default=20,
        help="Lookback period for volume statistics (default: 20)"
    )
    analyze_parser.add_argument(
        "--store-analysis",
        action="store_true",
        help="Store analysis results"
    )
    
    # Common arguments
    for subparser in [import_parser, analyze_parser]:
        subparser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level (default: INFO)"
        )
        subparser.add_argument(
            "--log-file",
            help="Log file (default: None)"
        )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level if hasattr(args, 'log_level') else "INFO",
                 args.log_file if hasattr(args, 'log_file') else None)
    
    # Run command
    if args.command == "import":
        return import_data(args)
    elif args.command == "analyze":
        return analyze_data(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
