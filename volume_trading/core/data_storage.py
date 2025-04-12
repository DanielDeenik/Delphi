"""
Data storage module for saving and loading data.
"""
import os
import logging
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from volume_trading.config import config

# Configure logging
logger = logging.getLogger(__name__)

class DataStorage:
    """Storage for market data and analysis results."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the data storage.
        
        Args:
            data_dir: Directory for storing data
        """
        self.data_dir = data_dir or config.data_dir
        
        # Create subdirectories
        self.price_dir = self.data_dir / "prices"
        self.analysis_dir = self.data_dir / "analysis"
        self.summary_dir = self.data_dir / "summary"
        
        # Create directories if they don't exist
        for directory in [self.price_dir, self.analysis_dir, self.summary_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_price_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save price data to CSV.
        
        Args:
            ticker: Stock symbol
            df: DataFrame with price data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker}, skipping save")
                return False
            
            # Create file path
            file_path = self.price_dir / f"{ticker.lower()}_prices.csv"
            
            # Save to CSV
            df.to_csv(file_path)
            
            logger.info(f"Saved price data for {ticker} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving price data for {ticker}: {str(e)}")
            return False
    
    def load_price_data(self, ticker: str) -> pd.DataFrame:
        """Load price data from CSV.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with price data
        """
        try:
            # Create file path
            file_path = self.price_dir / f"{ticker.lower()}_prices.csv"
            
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Price data file for {ticker} not found")
                return pd.DataFrame()
            
            # Load from CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            logger.info(f"Loaded price data for {ticker} from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def save_analysis_results(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save analysis results to CSV.
        
        Args:
            ticker: Stock symbol
            df: DataFrame with analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if df.empty:
                logger.warning(f"Empty DataFrame for {ticker}, skipping save")
                return False
            
            # Create file path
            file_path = self.analysis_dir / f"{ticker.lower()}_analysis.csv"
            
            # Save to CSV
            df.to_csv(file_path)
            
            logger.info(f"Saved analysis results for {ticker} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis results for {ticker}: {str(e)}")
            return False
    
    def load_analysis_results(self, ticker: str) -> pd.DataFrame:
        """Load analysis results from CSV.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            DataFrame with analysis results
        """
        try:
            # Create file path
            file_path = self.analysis_dir / f"{ticker.lower()}_analysis.csv"
            
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Analysis results file for {ticker} not found")
                return pd.DataFrame()
            
            # Load from CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            logger.info(f"Loaded analysis results for {ticker} from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading analysis results for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def save_summary(self, ticker: str, summary: Dict) -> bool:
        """Save summary to JSON.
        
        Args:
            ticker: Stock symbol
            summary: Dictionary with summary information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create file path
            file_path = self.summary_dir / f"{ticker.lower()}_summary.json"
            
            # Add timestamp
            summary["timestamp"] = datetime.now().isoformat()
            
            # Save to JSON
            with open(file_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved summary for {ticker} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving summary for {ticker}: {str(e)}")
            return False
    
    def load_summary(self, ticker: str) -> Dict:
        """Load summary from JSON.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with summary information
        """
        try:
            # Create file path
            file_path = self.summary_dir / f"{ticker.lower()}_summary.json"
            
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"Summary file for {ticker} not found")
                return {}
            
            # Load from JSON
            with open(file_path, "r") as f:
                summary = json.load(f)
            
            logger.info(f"Loaded summary for {ticker} from {file_path}")
            return summary
            
        except Exception as e:
            logger.error(f"Error loading summary for {ticker}: {str(e)}")
            return {}
    
    def save_master_summary(self, summaries: List[Dict]) -> bool:
        """Save master summary to JSON.
        
        Args:
            summaries: List of summary dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create file path
            file_path = self.summary_dir / "master_summary.json"
            
            # Add timestamp
            master_summary = {
                "summaries": summaries,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to JSON
            with open(file_path, "w") as f:
                json.dump(master_summary, f, indent=2)
            
            logger.info(f"Saved master summary to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving master summary: {str(e)}")
            return False
    
    def load_master_summary(self) -> Dict:
        """Load master summary from JSON.
        
        Returns:
            Dictionary with master summary information
        """
        try:
            # Create file path
            file_path = self.summary_dir / "master_summary.json"
            
            # Check if file exists
            if not file_path.exists():
                logger.warning("Master summary file not found")
                return {}
            
            # Load from JSON
            with open(file_path, "r") as f:
                master_summary = json.load(f)
            
            logger.info(f"Loaded master summary from {file_path}")
            return master_summary
            
        except Exception as e:
            logger.error(f"Error loading master summary: {str(e)}")
            return {}
