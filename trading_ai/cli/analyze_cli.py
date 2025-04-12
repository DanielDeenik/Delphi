#!/usr/bin/env python3
"""
Command-line interface for analyzing time series data.
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from trading_ai module
from trading_ai.config import config_manager
from trading_ai.core.bigquery_io import BigQueryStorage

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze time series data")
    
    # Ticker options
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument("--ticker", type=str, help="Analyze a specific ticker")
    ticker_group.add_argument("--tickers", type=str, nargs="+", help="Analyze multiple tickers")
    ticker_group.add_argument("--tickers-file", type=str, help="File with tickers to analyze (one per line)")
    
    # Analysis options
    parser.add_argument("--days", type=int, default=90, help="Number of days of data to analyze")
    parser.add_argument("--volume-analysis", action="store_true", help="Perform volume analysis")
    parser.add_argument("--correlation-analysis", action="store_true", help="Perform correlation analysis")
    parser.add_argument("--volatility-analysis", action="store_true", help="Perform volatility analysis")
    parser.add_argument("--all-analyses", action="store_true", help="Perform all analyses")
    
    # Output options
    parser.add_argument("--output", type=str, choices=["json", "csv", "table"], default="table", help="Output format")
    parser.add_argument("--output-file", type=str, help="Output file path")
    
    return parser.parse_args()

def get_tickers(args) -> List[str]:
    """Get list of tickers to analyze."""
    if args.ticker:
        return [args.ticker]
    elif args.tickers:
        return args.tickers
    elif args.tickers_file:
        with open(args.tickers_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    else:
        return config_manager.get_all_tickers()

def perform_volume_analysis(ticker: str, days: int):
    """Perform volume analysis for a ticker."""
    try:
        # Import the VolumeAnalyzer class
        from trading_ai.analysis.volume_analyzer import VolumeAnalyzer
        
        # Initialize analyzer
        analyzer = VolumeAnalyzer()
        
        # Perform analysis
        results = analyzer.analyze_ticker(ticker, days=days)
        
        return results
    except Exception as e:
        logger.error(f"Error performing volume analysis for {ticker}: {str(e)}")
        return None

def perform_correlation_analysis(tickers: List[str], days: int):
    """Perform correlation analysis for multiple tickers."""
    try:
        # Import the CorrelationAnalyzer class
        from trading_ai.analysis.correlation_analyzer import CorrelationAnalyzer
        
        # Initialize analyzer
        analyzer = CorrelationAnalyzer()
        
        # Perform analysis
        results = analyzer.analyze_tickers(tickers, days=days)
        
        return results
    except Exception as e:
        logger.error(f"Error performing correlation analysis: {str(e)}")
        return None

def perform_volatility_analysis(ticker: str, days: int):
    """Perform volatility analysis for a ticker."""
    try:
        # Import the VolatilityAnalyzer class
        from trading_ai.analysis.volatility_analyzer import VolatilityAnalyzer
        
        # Initialize analyzer
        analyzer = VolatilityAnalyzer()
        
        # Perform analysis
        results = analyzer.analyze_ticker(ticker, days=days)
        
        return results
    except Exception as e:
        logger.error(f"Error performing volatility analysis for {ticker}: {str(e)}")
        return None

def output_results(results, format: str, file_path: Optional[str] = None):
    """Output analysis results in the specified format."""
    if format == "json":
        output = json.dumps(results, indent=2)
    elif format == "csv":
        # Convert results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        output = df.to_csv(index=False)
    else:  # table
        import pandas as pd
        from tabulate import tabulate
        df = pd.DataFrame(results)
        output = tabulate(df, headers="keys", tablefmt="psql")
    
    if file_path:
        with open(file_path, "w") as f:
            f.write(output)
    else:
        print(output)

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Get tickers to analyze
        tickers = get_tickers(args)
        
        if not tickers:
            logger.error("No tickers specified")
            return 1
        
        logger.info(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}")
        
        # Determine which analyses to perform
        perform_volume = args.volume_analysis or args.all_analyses
        perform_correlation = args.correlation_analysis or args.all_analyses
        perform_volatility = args.volatility_analysis or args.all_analyses
        
        # If no specific analysis is requested, perform volume analysis by default
        if not (perform_volume or perform_correlation or perform_volatility):
            perform_volume = True
        
        # Perform analyses
        results = {}
        
        if perform_volume:
            logger.info("Performing volume analysis...")
            volume_results = {}
            for ticker in tickers:
                volume_results[ticker] = perform_volume_analysis(ticker, args.days)
            results["volume_analysis"] = volume_results
        
        if perform_correlation and len(tickers) > 1:
            logger.info("Performing correlation analysis...")
            results["correlation_analysis"] = perform_correlation_analysis(tickers, args.days)
        
        if perform_volatility:
            logger.info("Performing volatility analysis...")
            volatility_results = {}
            for ticker in tickers:
                volatility_results[ticker] = perform_volatility_analysis(ticker, args.days)
            results["volatility_analysis"] = volatility_results
        
        # Output results
        output_results(results, args.output, args.output_file)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
