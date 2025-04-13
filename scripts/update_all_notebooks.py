#!/usr/bin/env python3
"""
Update all notebooks with fresh data.

This script:
1. Imports data for all tickers
2. Generates notebooks for all tickers
3. Uploads notebooks to Google Colab
"""
import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a command and return the result."""
    try:
        logger.info(f"Running command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def main():
    """Main entry point for the script."""
    try:
        # Get script directory
        script_dir = Path(__file__).parent
        
        # Step 1: Import data for all tickers
        logger.info("Step 1: Importing data for all tickers")
        import_script = script_dir / "ticker_imports" / "import_master.py"
        success, output = run_command(f"python {import_script} --days 90")
        
        if not success:
            logger.error("Failed to import data for all tickers")
            return 1
        
        # Step 2: Generate notebooks for all tickers
        logger.info("Step 2: Generating notebooks for all tickers")
        notebook_script = script_dir / "generate_notebooks.py"
        success, output = run_command(f"python {notebook_script} --generate")
        
        if not success:
            logger.error("Failed to generate notebooks for all tickers")
            return 1
        
        # Step 3: Upload notebooks to Google Colab
        logger.info("Step 3: Uploading notebooks to Google Colab")
        success, output = run_command(f"python {notebook_script} --upload")
        
        if not success:
            logger.error("Failed to upload notebooks to Google Colab")
            return 1
        
        logger.info("All notebooks updated successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
