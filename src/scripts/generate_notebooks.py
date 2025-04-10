"""
Script to generate individual stock analysis notebooks from template.
"""
import json
import os
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        with open('config/tracked_stocks.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tracked stocks: {str(e)}")
        return None

def generate_notebooks(project_id):
    """Generate individual stock analysis notebooks from template."""
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    if not tracked_stocks:
        logger.error("Failed to load tracked stocks")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path('notebooks/individual')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load template
    template_path = Path('notebooks/stock_analysis_template.ipynb')
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        return False
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        logger.error(f"Error reading template file: {str(e)}")
        return False
    
    # Generate notebooks for each stock
    for direction, tickers in tracked_stocks.items():
        for ticker in tickers:
            try:
                # Replace placeholders in template
                notebook_content = template_content
                notebook_content = notebook_content.replace('{TICKER}', ticker)
                notebook_content = notebook_content.replace('{PROJECT_ID}', project_id)
                notebook_content = notebook_content.replace('{DIRECTION}', direction)
                
                # Replace other placeholders with dummy values (will be updated when notebook runs)
                notebook_content = notebook_content.replace('{SPIKE_COUNT}', '0')
                notebook_content = notebook_content.replace('{LATEST_Z_SCORE}', '0.0')
                notebook_content = notebook_content.replace('{CURRENT_SIGNAL}', 'NEUTRAL')
                
                # Save notebook
                output_path = output_dir / f"{ticker}_analysis.ipynb"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(notebook_content)
                
                logger.info(f"Generated notebook for {ticker}: {output_path}")
            
            except Exception as e:
                logger.error(f"Error generating notebook for {ticker}: {str(e)}")
    
    logger.info(f"Generated notebooks for {len(tracked_stocks['buy']) + len(tracked_stocks['short'])} stocks")
    return True

def main():
    """Main function."""
    # Get Google Cloud project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "delphi-449908")
    
    # Generate notebooks
    success = generate_notebooks(project_id)
    
    if success:
        logger.info("Notebook generation completed successfully")
    else:
        logger.error("Notebook generation failed")

if __name__ == "__main__":
    main()
