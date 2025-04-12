"""
Configuration settings for the Volume Intelligence Trading System.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class StockConfig:
    """Configuration for a stock."""
    ticker: str
    direction: str  # 'buy' or 'short'
    sector: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class SystemConfig:
    """System-wide configuration."""
    google_cloud_project: str
    bigquery_dataset: str
    alpha_vantage_api_key: str
    discord_webhook_url: Optional[str] = None
    discord_bot_token: Optional[str] = None
    
    # Trading parameters
    trading_days_lookback: int = 252  # ~1 year of trading days
    volume_ma_periods: List[int] = None
    
    def __post_init__(self):
        if self.volume_ma_periods is None:
            self.volume_ma_periods = [5, 20, 50]

class ConfigManager:
    """Manager for system configuration."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.system_config = None
        self.tracked_stocks = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from files."""
        try:
            # Load system config
            system_config_path = CONFIG_DIR / "system_config.json"
            if system_config_path.exists():
                with open(system_config_path, 'r') as f:
                    config_data = json.load(f)
                self.system_config = SystemConfig(**config_data)
            else:
                # Create default config
                self.system_config = SystemConfig(
                    google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                    bigquery_dataset=os.getenv("BIGQUERY_DATASET", "trading_insights"),
                    alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", "")
                )
                self.save_system_config()
            
            # Load tracked stocks
            stocks_config_path = CONFIG_DIR / "tracked_stocks.json"
            if stocks_config_path.exists():
                with open(stocks_config_path, 'r') as f:
                    self.tracked_stocks = json.load(f)
            else:
                # Create default tracked stocks
                self.tracked_stocks = {
                    "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", 
                           "TSLA", "META", "ADBE", "ORCL", "ASML"],
                    "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", 
                             "PLTR", "UBER", "LCID", "INTC", "XPEV"]
                }
                self.save_tracked_stocks()
            
            logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return False
    
    def save_system_config(self) -> bool:
        """Save system configuration to file."""
        try:
            system_config_path = CONFIG_DIR / "system_config.json"
            with open(system_config_path, 'w') as f:
                # Convert dataclass to dict
                config_dict = {
                    "google_cloud_project": self.system_config.google_cloud_project,
                    "bigquery_dataset": self.system_config.bigquery_dataset,
                    "alpha_vantage_api_key": self.system_config.alpha_vantage_api_key,
                    "discord_webhook_url": self.system_config.discord_webhook_url,
                    "discord_bot_token": self.system_config.discord_bot_token,
                    "trading_days_lookback": self.system_config.trading_days_lookback,
                    "volume_ma_periods": self.system_config.volume_ma_periods
                }
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"System configuration saved to {system_config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving system configuration: {str(e)}")
            return False
    
    def save_tracked_stocks(self) -> bool:
        """Save tracked stocks to file."""
        try:
            stocks_config_path = CONFIG_DIR / "tracked_stocks.json"
            with open(stocks_config_path, 'w') as f:
                json.dump(self.tracked_stocks, f, indent=2)
            
            logger.info(f"Tracked stocks saved to {stocks_config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving tracked stocks: {str(e)}")
            return False
    
    def update_tracked_stocks(self, buy_stocks: Optional[List[str]] = None, 
                             short_stocks: Optional[List[str]] = None) -> bool:
        """Update the tracked stocks configuration."""
        try:
            # Create a backup of current configuration
            stocks_config_path = CONFIG_DIR / "tracked_stocks.json"
            if stocks_config_path.exists():
                import shutil
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = CONFIG_DIR / f"tracked_stocks.{timestamp}.bak"
                shutil.copy2(stocks_config_path, backup_path)
                logger.info(f"Created backup of tracked stocks: {backup_path}")
            
            # Update stocks
            if buy_stocks is not None:
                self.tracked_stocks["buy"] = buy_stocks
            
            if short_stocks is not None:
                self.tracked_stocks["short"] = short_stocks
            
            # Save updated configuration
            return self.save_tracked_stocks()
            
        except Exception as e:
            logger.error(f"Error updating tracked stocks: {str(e)}")
            return False
    
    def get_all_tickers(self) -> List[str]:
        """Get a list of all tracked tickers."""
        all_tickers = []
        for direction in self.tracked_stocks:
            all_tickers.extend(self.tracked_stocks[direction])
        return all_tickers
    
    def get_ticker_direction(self, ticker: str) -> Optional[str]:
        """Get the direction (buy/short) for a ticker."""
        for direction, tickers in self.tracked_stocks.items():
            if ticker in tickers:
                return direction
        return None

# Global config instance
config_manager = ConfigManager()
