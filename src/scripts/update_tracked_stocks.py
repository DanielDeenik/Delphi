"""
Script to update the tracked stocks configuration and generate new notebooks.
"""
import json
import os
import logging
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_tracked_stocks():
    """Load the list of tracked stocks from configuration."""
    try:
        config_path = os.getenv("CONFIG_PATH", "config/tracked_stocks.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading tracked stocks: {str(e)}")
        return None

def update_tracked_stocks(buy_stocks=None, short_stocks=None):
    """Update the tracked stocks configuration and generate new notebooks."""
    # Get current tracked stocks
    current_stocks = load_tracked_stocks()
    if not current_stocks:
        logger.error("Failed to load current tracked stocks")
        return False
    
    # Update with new stocks if provided
    if buy_stocks is not None:
        current_stocks['buy'] = buy_stocks
    
    if short_stocks is not None:
        current_stocks['short'] = short_stocks
    
    # Create a backup of the current configuration
    config_path = os.getenv("CONFIG_PATH", "config/tracked_stocks.json")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{config_path}.{timestamp}.bak"
    
    try:
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup of tracked stocks configuration: {backup_path}")
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
    
    # Update the configuration file
    try:
        with open(config_path, 'w') as f:
            json.dump(current_stocks, f, indent=2)
        logger.info(f"Updated tracked stocks configuration: {config_path}")
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return False
    
    # Generate new notebooks
    from src.scripts.generate_notebooks import generate_notebooks
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "delphi-449908")
    generate_notebooks(project_id)
    
    return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update tracked stocks configuration")
    parser.add_argument("--buy", nargs="+", help="List of buy stocks")
    parser.add_argument("--short", nargs="+", help="List of short stocks")
    
    args = parser.parse_args()
    
    if not args.buy and not args.short:
        logger.error("No stocks provided. Use --buy and/or --short to specify stocks.")
        return False
    
    success = update_tracked_stocks(
        buy_stocks=args.buy,
        short_stocks=args.short
    )
    
    if success:
        logger.info("Tracked stocks updated successfully")
    else:
        logger.error("Failed to update tracked stocks")

if __name__ == "__main__":
    main()
