"""
Script to generate the master summary notebook from template.
"""
import json
import os
import logging
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

def generate_master_notebook(project_id):
    """Generate the master summary notebook from template."""
    # Load tracked stocks
    tracked_stocks = load_tracked_stocks()
    if not tracked_stocks:
        logger.error("Failed to load tracked stocks")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path('notebooks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load template
    template_path = Path('notebooks/master_summary_template.ipynb')
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        return False
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except Exception as e:
        logger.error(f"Error reading template file: {str(e)}")
        return False
    
    try:
        # Replace placeholders in template
        notebook_content = template_content
        notebook_content = notebook_content.replace('{PROJECT_ID}', project_id)
        
        # Convert tracked stocks to JSON string
        tracked_stocks_json = json.dumps(tracked_stocks, indent=2)
        notebook_content = notebook_content.replace('{TRACKED_STOCKS_JSON}', tracked_stocks_json)
        
        # Replace other placeholders with dummy values (will be updated when notebook runs)
        notebook_content = notebook_content.replace('{BUY_SIGNAL_COUNT}', '0')
        notebook_content = notebook_content.replace('{SHORT_SIGNAL_COUNT}', '0')
        notebook_content = notebook_content.replace('{TOP_BUY_TICKER}', 'None')
        notebook_content = notebook_content.replace('{TOP_SHORT_TICKER}', 'None')
        
        # Save notebook
        output_path = output_dir / "master_summary.ipynb"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(notebook_content)
        
        logger.info(f"Generated master summary notebook: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating master summary notebook: {str(e)}")
        return False

def main():
    """Main function."""
    # Get Google Cloud project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "delphi-449908")
    
    # Generate master notebook
    success = generate_master_notebook(project_id)
    
    if success:
        logger.info("Master notebook generation completed successfully")
    else:
        logger.error("Master notebook generation failed")

if __name__ == "__main__":
    main()
