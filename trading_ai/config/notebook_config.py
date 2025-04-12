"""
Configuration module for notebook URLs.

This module provides a centralized place to manage Google Colab notebook URLs.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class NotebookConfig:
    """Configuration manager for notebook URLs."""
    
    def __init__(self):
        """Initialize the notebook configuration manager."""
        self.config_dir = Path("config")
        self.config_file = self.config_dir / "notebook_urls.json"
        self.notebook_urls = {}
        self.load_config()
    
    def load_config(self) -> bool:
        """Load notebook URLs from configuration file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(exist_ok=True)
            
            # Load configuration if file exists
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.notebook_urls = json.load(f)
                logger.info(f"Loaded {len(self.notebook_urls)} notebook URLs from {self.config_file}")
                return True
            else:
                # Create default configuration
                self.notebook_urls = {
                    'master': 'https://colab.research.google.com/drive/1Z7wSY5xQcK9qCzVo_-Vsf5r5rDH6Q9Xf?usp=sharing',  # Master summary notebook
                    'AAPL': 'https://colab.research.google.com/drive/1Zn_Hm-rlYrJK9xPdCmzU-VYH2Oo0YiLT?usp=sharing',    # Apple volume analysis
                    'MSFT': 'https://colab.research.google.com/drive/1ZoXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # Microsoft volume analysis
                    'GOOGL': 'https://colab.research.google.com/drive/1ZpXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',   # Google volume analysis
                    'AMZN': 'https://colab.research.google.com/drive/1ZqXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # Amazon volume analysis
                    'META': 'https://colab.research.google.com/drive/1ZrXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # Meta volume analysis
                    'TSLA': 'https://colab.research.google.com/drive/1ZsXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # Tesla volume analysis
                    'NVDA': 'https://colab.research.google.com/drive/1ZtXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # NVIDIA volume analysis
                    'JPM': 'https://colab.research.google.com/drive/1ZuXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # JPMorgan volume analysis
                    'V': 'https://colab.research.google.com/drive/1ZvXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',      # Visa volume analysis
                    'JNJ': 'https://colab.research.google.com/drive/1ZwXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # Johnson & Johnson volume analysis
                    'WMT': 'https://colab.research.google.com/drive/1ZxXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # Walmart volume analysis
                    'PG': 'https://colab.research.google.com/drive/1ZyXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',      # Procter & Gamble volume analysis
                    'MA': 'https://colab.research.google.com/drive/1ZzXnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',      # Mastercard volume analysis
                    'UNH': 'https://colab.research.google.com/drive/1Z0YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # UnitedHealth volume analysis
                    'HD': 'https://colab.research.google.com/drive/1Z1YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',      # Home Depot volume analysis
                    'BAC': 'https://colab.research.google.com/drive/1Z2YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # Bank of America volume analysis
                    'PFE': 'https://colab.research.google.com/drive/1Z3YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # Pfizer volume analysis
                    'CSCO': 'https://colab.research.google.com/drive/1Z4YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',    # Cisco volume analysis
                    'DIS': 'https://colab.research.google.com/drive/1Z5YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',     # Disney volume analysis
                    'VZ': 'https://colab.research.google.com/drive/1Z6YnvZ7JvZyJ5JQyGkJKX5ZyZZ5ZyZZZ?usp=sharing',      # Verizon volume analysis
                }
                self.save_config()
                logger.info(f"Created default notebook URL configuration with {len(self.notebook_urls)} entries")
                return True
        except Exception as e:
            logger.error(f"Error loading notebook configuration: {str(e)}")
            return False
    
    def save_config(self) -> bool:
        """Save notebook URLs to configuration file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(exist_ok=True)
            
            # Save configuration
            with open(self.config_file, 'w') as f:
                json.dump(self.notebook_urls, f, indent=2)
            logger.info(f"Saved {len(self.notebook_urls)} notebook URLs to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving notebook configuration: {str(e)}")
            return False
    
    def get_notebook_url(self, ticker: str) -> str:
        """Get the notebook URL for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            str: Notebook URL
        """
        # Default notebook URL (used if ticker not in dictionary)
        default_url = 'https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb'
        
        # Return URL for ticker or default URL if not found
        return self.notebook_urls.get(ticker, default_url)
    
    def set_notebook_url(self, ticker: str, url: str) -> bool:
        """Set the notebook URL for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            url: Notebook URL
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update URL
            self.notebook_urls[ticker] = url
            
            # Save configuration
            return self.save_config()
        except Exception as e:
            logger.error(f"Error setting notebook URL for {ticker}: {str(e)}")
            return False
    
    def get_all_notebook_urls(self) -> Dict[str, str]:
        """Get all notebook URLs.
        
        Returns:
            Dict[str, str]: Dictionary of ticker to URL mappings
        """
        return self.notebook_urls.copy()

# Create singleton instance
notebook_config = NotebookConfig()
