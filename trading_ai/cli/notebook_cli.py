#!/usr/bin/env python3
"""
Command-line interface for generating and managing notebooks.
"""
import argparse
import logging
import sys
import os
import webbrowser
from pathlib import Path
from typing import List, Optional

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
from trading_ai.notebooks.generator import NotebookGenerator
from trading_ai.config.notebook_config import notebook_config

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and manage Google Colab notebooks")
    
    # Notebook options
    parser.add_argument("--generate", action="store_true", help="Generate notebooks")
    parser.add_argument("--tickers", type=str, nargs="+", help="Tickers to generate notebooks for (default: all tracked tickers)")
    parser.add_argument("--upload", action="store_true", help="Upload notebooks to Google Colab")
    parser.add_argument("--open", action="store_true", help="Open notebooks in browser")
    parser.add_argument("--ticker", type=str, help="Open notebook for a specific ticker")
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Initialize notebook generator
        generator = NotebookGenerator()
        
        if args.generate:
            # Generate notebooks
            logger.info("Generating notebooks...")
            notebook_paths = generator.generate_notebooks(args.tickers)
            
            if notebook_paths:
                logger.info(f"Generated {len(notebook_paths)} notebooks")
                
                # Upload notebooks if requested
                if args.upload:
                    logger.info("Uploading notebooks to Google Colab...")
                    notebook_urls = generator.upload_notebooks_to_colab(notebook_paths)
                    
                    if notebook_urls:
                        logger.info(f"Uploaded {len(notebook_urls)} notebooks")
                    else:
                        logger.warning("Failed to upload notebooks")
            else:
                logger.warning("Failed to generate notebooks")
        
        if args.open:
            # Open notebook in browser
            if args.ticker:
                # Open specific ticker notebook
                url = notebook_config.get_notebook_url(args.ticker)
                if url:
                    logger.info(f"Opening notebook for {args.ticker}...")
                    webbrowser.open(url)
                else:
                    logger.warning(f"No notebook URL found for {args.ticker}")
            else:
                # Open master notebook
                url = notebook_config.get_notebook_url('master')
                if url:
                    logger.info("Opening master notebook...")
                    webbrowser.open(url)
                else:
                    logger.warning("No master notebook URL found")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in notebook CLI: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
