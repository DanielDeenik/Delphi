#!/usr/bin/env python
"""
Evaluate Stocks Script

This script evaluates stocks using the Mosaic Theory approach.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.mosaic_theory_evaluator import MosaicTheoryEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate stocks using Mosaic Theory')
    
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols to evaluate')
    parser.add_argument('--days', type=int, default=90, help='Number of days of data to analyze (default: 90)')
    parser.add_argument('--output', help='Output file for evaluation results (JSON format)')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    if not symbols:
        logger.error("No symbols specified")
        return 1
    
    logger.info(f"Evaluating {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Initialize evaluator
    evaluator = MosaicTheoryEvaluator()
    
    # Evaluate stocks
    results = evaluator.evaluate_multiple(symbols, args.days)
    
    # Print summary
    for symbol, evaluation in results.items():
        if 'error' in evaluation:
            logger.warning(f"{symbol}: Error - {evaluation['error']}")
        else:
            logger.info(f"{symbol}: {evaluation['rating']} (Score: {evaluation['composite_score']:.2f})")
    
    # Save results if output file specified
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
