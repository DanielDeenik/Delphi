"""
Analyze Command

This module provides a command-line interface for analyzing data.
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..services import BigQueryService
from ..models import VolumeAnalyzer, CorrelationAnalyzer
from ..visualization import plot_volume, plot_correlation_matrix, plot_normalized_prices
from ..utils.config import get_config, load_env
from ..utils.logger import get_logger
from ..utils.parallel import parallel_process

logger = logging.getLogger(__name__)

def analyze_data():
    """
    Analyze data from BigQuery.
    """
    # Load environment variables
    load_env()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze data from BigQuery")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to analyze (default: from config)"
    )
    parser.add_argument(
        "--project-id",
        help="Google Cloud project ID (default: from config)"
    )
    parser.add_argument(
        "--dataset-id",
        help="BigQuery dataset ID (default: from config)"
    )
    parser.add_argument(
        "--table-id",
        help="BigQuery table ID (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to save plots to (default: plots)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of workers for parallel processing (default: 4)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Log file (default: None)"
    )
    
    args = parser.parse_args()
    
    # Set up logger
    logger = get_logger(__name__, getattr(logging, args.log_level), args.log_file)
    
    # Get config
    config = get_config()
    
    # Get symbols
    symbols = args.symbols
    if not symbols:
        symbols = config['symbols']['default']
    
    try:
        # Create BigQuery service
        service = BigQueryService(
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            table_id=args.table_id
        )
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Fetch data
        data = {}
        for symbol in symbols:
            try:
                df = service.get_market_data(symbol, start_date, end_date)
                if not df.empty:
                    data[symbol] = df
                    logger.info(f"Retrieved {len(df)} rows for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        if not data:
            logger.error("No data to analyze")
            return 1
        
        # Analyze volume patterns
        volume_analyzer = VolumeAnalyzer()
        
        # Define function for parallel processing
        def analyze_volume(symbol):
            try:
                df = data[symbol]
                result = volume_analyzer.analyze(df)
                
                # Plot volume patterns
                plot_paths = plot_volume(df, output_dir=args.output_dir, symbol=symbol)
                
                return symbol, result
            except Exception as e:
                logger.error(f"Error analyzing volume patterns for {symbol}: {str(e)}")
                return symbol, {"error": str(e), "success": False}
        
        # Analyze volume patterns in parallel
        volume_results = {}
        results = parallel_process(list(data.keys()), analyze_volume, max_workers=args.max_workers)
        for symbol, result in results:
            volume_results[symbol] = result
        
        # Analyze correlations
        correlation_analyzer = CorrelationAnalyzer()
        correlation_results = correlation_analyzer.analyze_parallel(data, max_workers=args.max_workers)
        
        # Plot correlations
        if correlation_results.get('success', False):
            # Create correlation DataFrame
            return_corr = pd.DataFrame(correlation_results['correlations']['return_correlation'])
            
            # Plot correlation matrix
            plot_correlation_matrix(return_corr, output_dir=args.output_dir)
            
            # Create price DataFrame
            price_df = pd.DataFrame()
            for symbol, df in data.items():
                price_df[symbol] = df['close']
            
            # Plot normalized prices
            plot_normalized_prices(price_df, output_dir=args.output_dir)
        
        # Print volume analysis summary
        logger.info("\nVolume Analysis Summary:")
        for symbol, result in volume_results.items():
            if 'error' in result:
                logger.error(f"{symbol}: Error - {result['error']}")
                continue
                
            logger.info(f"\n{symbol} Analysis:")
            logger.info(f"  Data points: {result['data_points']}")
            logger.info(f"  Volume spikes: {result['patterns']['volume_spikes']}")
            logger.info(f"  Volume drops: {result['patterns']['volume_drops']}")
            logger.info(f"  Price up, volume down: {result['patterns']['price_up_volume_down']}")
            logger.info(f"  Price down, volume up: {result['patterns']['price_down_volume_up']}")
            logger.info(f"  OBV bearish divergence: {result['patterns']['obv_bearish_divergence']}")
            logger.info(f"  OBV bullish divergence: {result['patterns']['obv_bullish_divergence']}")
            logger.info(f"  Average volume: {result['metrics']['avg_volume']:.2f}")
            logger.info(f"  Last close: {result['metrics']['last_close']:.2f}")
            logger.info(f"  Last volume: {result['metrics']['last_volume']}")
            logger.info(f"  Last volume ratio: {result['metrics']['last_volume_ratio']:.2f}")
            
            # Print signals
            if result['summary']['signals']:
                logger.info(f"  Signals:")
                for signal in result['summary']['signals']:
                    logger.info(f"    {signal['type']}: {signal['description']} ({signal['strength']})")
        
        # Print correlation analysis summary
        logger.info("\nCorrelation Analysis Summary:")
        if 'error' in correlation_results:
            logger.error(f"Correlation analysis failed: {correlation_results['error']}")
        else:
            for pair, corr in correlation_results['key_correlations'].items():
                if corr is not None:
                    logger.info(f"  {pair}: {corr:.4f}")
            
            # Print signals
            if correlation_results['summary']['signals']:
                logger.info(f"  Signals:")
                for signal in correlation_results['summary']['signals']:
                    logger.info(f"    {signal['type']}: {signal['description']} ({signal['strength']})")
        
        logger.info("\nAnalysis completed successfully!")
        logger.info(f"Plots saved to {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error in analyze_data: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(analyze_data())
