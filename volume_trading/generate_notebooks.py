"""
Script to generate notebooks for the Volume Trading System.
"""
import os
import logging
from pathlib import Path
from datetime import datetime

from volume_trading.config import config
from volume_trading.notebooks.generator import NotebookGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"notebook_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def generate_notebooks():
    """Generate notebooks for all tracked stocks."""
    logger.info("Starting notebook generation...")
    
    # Create notebook generator
    generator = NotebookGenerator()
    
    # Generate notebooks
    notebook_paths = generator.generate_all_notebooks()
    
    # Print summary
    print("\nNotebook Generation Summary:")
    print(f"Generated {len(notebook_paths)} notebooks:")
    
    # Group by type
    stock_notebooks = [p for p in notebook_paths if "master" not in p.name]
    master_notebooks = [p for p in notebook_paths if "master" in p.name]
    
    print(f"- Stock notebooks: {len(stock_notebooks)}")
    for path in stock_notebooks:
        print(f"  - {path}")
    
    print(f"- Master notebooks: {len(master_notebooks)}")
    for path in master_notebooks:
        print(f"  - {path}")
    
    print("\nTo open the master dashboard, run:")
    print("jupyter notebook notebooks/master_dashboard.ipynb")

if __name__ == "__main__":
    generate_notebooks()
