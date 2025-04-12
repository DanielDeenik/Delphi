"""
Runner script for the Volume Trading System.
"""
import os
import logging
import argparse
from typing import List, Optional
import pandas as pd
from datetime import datetime

from volume_trading.config import config
from volume_trading.core.data_fetcher import AlphaVantageClient
from volume_trading.core.volume_analyzer import VolumeAnalyzer
from volume_trading.core.data_storage import DataStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"volume_trading_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

def fetch_data(tickers: Optional[List[str]] = None, force_refresh: bool = False) -> bool:
    """Fetch data for the specified tickers.
    
    Args:
        tickers: List of tickers to fetch (None for all tracked tickers)
        force_refresh: Whether to force a full refresh
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get tickers
        if tickers is None:
            tickers = config.get_all_tickers()
        
        logger.info(f"Fetching data for {len(tickers)} tickers...")
        
        # Initialize clients
        client = AlphaVantageClient()
        storage = DataStorage()
        
        # Fetch and save data for each ticker
        for ticker in tickers:
            # Check if we already have data
            existing_df = storage.load_price_data(ticker)
            
            # Determine output size
            outputsize = "full" if force_refresh or existing_df.empty else "compact"
            
            # Fetch data
            df = client.fetch_daily_data(ticker, outputsize=outputsize)
            
            if not df.empty:
                # Save data
                storage.save_price_data(ticker, df)
            else:
                logger.warning(f"No data fetched for {ticker}")
        
        logger.info("Data fetching completed")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return False

def analyze_data(tickers: Optional[List[str]] = None) -> bool:
    """Analyze data for the specified tickers.
    
    Args:
        tickers: List of tickers to analyze (None for all tracked tickers)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get tickers
        if tickers is None:
            tickers = config.get_all_tickers()
        
        logger.info(f"Analyzing data for {len(tickers)} tickers...")
        
        # Initialize analyzer and storage
        analyzer = VolumeAnalyzer()
        storage = DataStorage()
        
        # Analyze data for each ticker
        summaries = []
        
        for ticker in tickers:
            # Load price data
            df = storage.load_price_data(ticker)
            
            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                continue
            
            # Analyze data
            analysis_df = analyzer.analyze(df)
            
            # Save analysis results
            storage.save_analysis_results(ticker, analysis_df)
            
            # Get summary
            summary = analyzer.get_summary(analysis_df)
            
            # Save summary
            storage.save_summary(ticker, summary)
            
            # Add to summaries list
            summaries.append(summary)
        
        # Save master summary
        storage.save_master_summary(summaries)
        
        logger.info("Data analysis completed")
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        return False

def print_summary(ticker: Optional[str] = None) -> bool:
    """Print summary for the specified ticker or all tickers.
    
    Args:
        ticker: Ticker to print summary for (None for all tickers)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize storage
        storage = DataStorage()
        
        if ticker:
            # Load summary for specific ticker
            summary = storage.load_summary(ticker)
            
            if not summary:
                logger.warning(f"No summary found for {ticker}")
                return False
            
            # Print summary
            print(f"\nSummary for {ticker}:")
            print(f"Direction: {summary.get('direction', '')}")
            print(f"Latest Close: ${summary.get('latest_close', 0):.2f}")
            print(f"Latest Volume: {summary.get('latest_volume', 0):,}")
            print(f"Volume Z-Score: {summary.get('latest_volume_z_score', 0):.2f}")
            print(f"Is Volume Spike: {summary.get('is_volume_spike', False)}")
            print(f"Latest Signal: {summary.get('latest_signal', 'NEUTRAL')}")
            print(f"Signal Strength: {summary.get('signal_strength', 0):.2f}")
            print(f"Notes: {summary.get('notes', '')}")
            print(f"Volume Spikes Count: {summary.get('volume_spikes_count', 0)}")
            print(f"Buy Signals Count: {summary.get('buy_signals_count', 0)}")
            print(f"Short Signals Count: {summary.get('short_signals_count', 0)}")
            print(f"Reversal Signals Count: {summary.get('reversal_signals_count', 0)}")
            print(f"Timestamp: {summary.get('timestamp', '')}")
            
        else:
            # Load master summary
            master_summary = storage.load_master_summary()
            
            if not master_summary:
                logger.warning("No master summary found")
                return False
            
            # Get summaries
            summaries = master_summary.get("summaries", [])
            
            if not summaries:
                logger.warning("No summaries found in master summary")
                return False
            
            # Print master summary
            print(f"\nMaster Summary ({len(summaries)} tickers):")
            print(f"Timestamp: {master_summary.get('timestamp', '')}")
            
            # Sort summaries by signal strength
            sorted_summaries = sorted(summaries, key=lambda x: x.get("signal_strength", 0), reverse=True)
            
            # Print buy signals
            buy_signals = [s for s in sorted_summaries if s.get("direction") == "buy" and s.get("latest_signal") != "NEUTRAL"]
            if buy_signals:
                print("\nBuy Signals:")
                for summary in buy_signals:
                    print(f"  {summary.get('ticker')}: {summary.get('latest_signal')} (Strength: {summary.get('signal_strength', 0):.2f})")
                    print(f"    {summary.get('notes', '')}")
            
            # Print short signals
            short_signals = [s for s in sorted_summaries if s.get("direction") == "short" and s.get("latest_signal") != "NEUTRAL"]
            if short_signals:
                print("\nShort Signals:")
                for summary in short_signals:
                    print(f"  {summary.get('ticker')}: {summary.get('latest_signal')} (Strength: {summary.get('signal_strength', 0):.2f})")
                    print(f"    {summary.get('notes', '')}")
            
            # Print volume spikes
            volume_spikes = [s for s in sorted_summaries if s.get("is_volume_spike", False)]
            if volume_spikes:
                print("\nRecent Volume Spikes:")
                for summary in volume_spikes:
                    print(f"  {summary.get('ticker')}: Z-Score: {summary.get('latest_volume_z_score', 0):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error printing summary: {str(e)}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Volume Trading System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch data")
    fetch_parser.add_argument("--tickers", nargs="+", help="Tickers to fetch (space-separated)")
    fetch_parser.add_argument("--force", action="store_true", help="Force full refresh")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument("--tickers", nargs="+", help="Tickers to analyze (space-separated)")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Print summary")
    summary_parser.add_argument("--ticker", help="Ticker to print summary for")
    
    # Run command (fetch + analyze)
    run_parser = subparsers.add_parser("run", help="Fetch and analyze data")
    run_parser.add_argument("--tickers", nargs="+", help="Tickers to process (space-separated)")
    run_parser.add_argument("--force", action="store_true", help="Force full refresh")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "fetch":
        fetch_data(args.tickers, args.force)
    elif args.command == "analyze":
        analyze_data(args.tickers)
    elif args.command == "summary":
        print_summary(args.ticker)
    elif args.command == "run":
        fetch_data(args.tickers, args.force)
        analyze_data(args.tickers)
        print_summary()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
