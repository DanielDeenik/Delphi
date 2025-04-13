"""
Configuration manager for the trading_ai package.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for the trading_ai package."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.config_dir = Path("config")
        self.config_file = self.config_dir / "config.json"
        self.config = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(exist_ok=True)
            
            # Load configuration if file exists
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return True
            else:
                # Create default configuration
                self.config = {
                    "alpha_vantage": {
                        "api_key": os.environ.get("ALPHA_VANTAGE_API_KEY", "IAS7UEKOT0HZW0MY"),
                        "base_url": "https://www.alphavantage.co/query"
                    },
                    "google_cloud": {
                        "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", "delphi-449908"),
                        "dataset": "stock_data"
                    },
                    "tickers": [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
                        "TSLA", "NVDA", "JPM", "V", "JNJ",
                        "WMT", "PG", "MA", "UNH", "HD",
                        "BAC", "PFE", "CSCO", "DIS", "VZ"
                    ]
                }
                self.save_config()
                logger.info(f"Created default configuration at {self.config_file}")
                return True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(exist_ok=True)
            
            # Save configuration
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get_alpha_vantage_api_key(self) -> str:
        """Get Alpha Vantage API key.
        
        Returns:
            str: Alpha Vantage API key
        """
        return self.config.get("alpha_vantage", {}).get("api_key", "")
    
    def get_alpha_vantage_base_url(self) -> str:
        """Get Alpha Vantage base URL.
        
        Returns:
            str: Alpha Vantage base URL
        """
        return self.config.get("alpha_vantage", {}).get("base_url", "")
    
    def get_google_cloud_project_id(self) -> str:
        """Get Google Cloud project ID.
        
        Returns:
            str: Google Cloud project ID
        """
        return self.config.get("google_cloud", {}).get("project_id", "")
    
    def get_google_cloud_dataset(self) -> str:
        """Get Google Cloud dataset.
        
        Returns:
            str: Google Cloud dataset
        """
        return self.config.get("google_cloud", {}).get("dataset", "")
    
    def get_all_tickers(self) -> List[str]:
        """Get all tickers.
        
        Returns:
            List[str]: List of all tickers
        """
        return self.config.get("tickers", [])
    
    def add_ticker(self, ticker: str) -> bool:
        """Add a ticker to the configuration.
        
        Args:
            ticker: Ticker to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current tickers
            tickers = self.get_all_tickers()
            
            # Add ticker if it doesn't exist
            if ticker not in tickers:
                tickers.append(ticker)
                self.config["tickers"] = tickers
                self.save_config()
                logger.info(f"Added ticker {ticker} to configuration")
                return True
            else:
                logger.info(f"Ticker {ticker} already exists in configuration")
                return True
        except Exception as e:
            logger.error(f"Error adding ticker {ticker}: {str(e)}")
            return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from the configuration.
        
        Args:
            ticker: Ticker to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current tickers
            tickers = self.get_all_tickers()
            
            # Remove ticker if it exists
            if ticker in tickers:
                tickers.remove(ticker)
                self.config["tickers"] = tickers
                self.save_config()
                logger.info(f"Removed ticker {ticker} from configuration")
                return True
            else:
                logger.info(f"Ticker {ticker} not found in configuration")
                return True
        except Exception as e:
            logger.error(f"Error removing ticker {ticker}: {str(e)}")
            return False

# Create singleton instance
config_manager = ConfigManager()
