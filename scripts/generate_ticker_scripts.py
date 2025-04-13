#!/usr/bin/env python3
"""
Generate import scripts for individual tickers.

This script generates import scripts for each ticker in the configuration.
"""
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from trading_ai module
from trading_ai.config import config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def generate_ticker_script(ticker):
    """Generate an import script for a ticker."""
    # Get template
    template_path = Path(__file__).parent / "ticker_imports" / "import_template.py"
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        return False
    
    # Read template
    with open(template_path, "r") as f:
        template = f.read()
    
    # Replace placeholders
    script = template.replace("{TICKER}", ticker)
    
    # Create output file
    output_path = Path(__file__).parent / "ticker_imports" / f"import_{ticker.lower()}.py"
    with open(output_path, "w") as f:
        f.write(script)
    
    # Make executable on Unix
    try:
        os.chmod(output_path, 0o755)
    except:
        pass
    
    logger.info(f"Generated script for {ticker}: {output_path}")
    return True

def generate_batch_file(ticker):
    """Generate a batch file for a ticker."""
    # Create batch file for Windows
    batch_path = Path(__file__).parent / "ticker_imports" / f"import_{ticker.lower()}.bat"
    batch_content = f"""@echo off
REM Import data for {ticker}
python %~dp0\\import_{ticker.lower()}.py %*
pause
"""
    with open(batch_path, "w") as f:
        f.write(batch_content)
    
    # Create shell script for Unix
    shell_path = Path(__file__).parent / "ticker_imports" / f"import_{ticker.lower()}.sh"
    shell_content = f"""#!/bin/bash
# Import data for {ticker}
python "$(dirname "$0")/import_{ticker.lower()}.py" "$@"
"""
    with open(shell_path, "w") as f:
        f.write(shell_content)
    
    # Make shell script executable on Unix
    try:
        os.chmod(shell_path, 0o755)
    except:
        pass
    
    logger.info(f"Generated batch files for {ticker}")
    return True

def main():
    """Main entry point for the script."""
    try:
        # Get all tickers
        tickers = config_manager.get_all_tickers()
        
        if not tickers:
            logger.error("No tickers found in configuration")
            return 1
        
        logger.info(f"Generating scripts for {len(tickers)} tickers")
        
        # Add master script
        tickers = ["MASTER"] + tickers
        
        # Generate scripts for each ticker
        success_count = 0
        for ticker in tickers:
            if generate_ticker_script(ticker) and generate_batch_file(ticker):
                success_count += 1
        
        logger.info(f"Generated {success_count}/{len(tickers)} ticker scripts")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
