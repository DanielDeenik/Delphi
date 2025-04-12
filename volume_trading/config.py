"""
Basic configuration system for the Volume Trading System.
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the trading system."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        # Set up paths
        self.root_dir = Path(__file__).parent.parent
        self.config_dir = self.root_dir / "config"
        self.data_dir = self.root_dir / "data"
        
        # Create directories if they don't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            # API keys
            "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            
            # Google Cloud settings
            "google_cloud_project": os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            "bigquery_dataset": os.getenv("BIGQUERY_DATASET", "trading_insights"),
            
            # Analysis settings
            "volume_ma_periods": [5, 20, 50],
            "volume_spike_threshold": 2.0,  # Z-score threshold for volume spikes
            
            # Tracked stocks
            "tracked_stocks": {
                "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
                "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN"]
            }
        }
        
        # Load configuration from file if it exists
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        config_file = self.config_dir / "config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Update configuration with loaded values
                self.config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_file}")
                return True
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return False
        else:
            # Save default configuration
            self.save_config()
            return True
    
    def save_config(self) -> bool:
        """Save configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        config_file = self.config_dir / "config.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved configuration to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def get_tracked_stocks(self) -> Dict[str, List[str]]:
        """Get the tracked stocks.
        
        Returns:
            Dictionary mapping directions to lists of tickers
        """
        return self.config.get("tracked_stocks", {"buy": [], "short": []})
    
    def get_all_tickers(self) -> List[str]:
        """Get all tracked tickers.
        
        Returns:
            List of all tracked tickers
        """
        tracked_stocks = self.get_tracked_stocks()
        all_tickers = []
        
        for direction in tracked_stocks:
            all_tickers.extend(tracked_stocks[direction])
        
        return all_tickers
    
    def update_tracked_stocks(self, buy_stocks: Optional[List[str]] = None, 
                             short_stocks: Optional[List[str]] = None) -> bool:
        """Update the tracked stocks.
        
        Args:
            buy_stocks: List of buy stocks
            short_stocks: List of short stocks
            
        Returns:
            True if successful, False otherwise
        """
        tracked_stocks = self.get_tracked_stocks()
        
        if buy_stocks is not None:
            tracked_stocks["buy"] = buy_stocks
        
        if short_stocks is not None:
            tracked_stocks["short"] = short_stocks
        
        self.set("tracked_stocks", tracked_stocks)
        return self.save_config()

# Global configuration instance
config = Config()
