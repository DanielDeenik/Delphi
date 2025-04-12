"""
Installation script for the Volume Trading System.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install():
    """Install the Volume Trading System."""
    logger.info("Installing Volume Trading System...")
    
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    
    try:
        # Install requirements
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install package in development mode
        logger.info("Installing package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        
        # Create config directory
        config_dir = current_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data directory
        data_dir = current_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create notebooks directory
        notebooks_dir = current_dir / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Installation completed successfully")
        
        # Print instructions
        print("\nVolume Trading System installed successfully!")
        print("\nTo set up your API key, create a .env file with the following content:")
        print("ALPHA_VANTAGE_API_KEY=your_api_key")
        print("\nTo import data and generate notebooks, run:")
        print("python -m volume_trading.run_all")
        print("\nTo open the master dashboard, run:")
        print("jupyter notebook notebooks/master_dashboard.ipynb")
        
    except Exception as e:
        logger.error(f"Error installing Volume Trading System: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Installation failed. Please check the logs for details.")

if __name__ == "__main__":
    install()
