"""
Notebook generator module.

This module provides functionality to generate Google Colab notebooks for stock analysis.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil

from trading_ai.config import config_manager
from trading_ai.config.notebook_config import notebook_config

# Configure logging
logger = logging.getLogger(__name__)

class NotebookGenerator:
    """Generator for Google Colab notebooks."""
    
    def __init__(self):
        """Initialize the notebook generator."""
        self.template_dir = Path("notebooks/templates")
        self.output_dir = Path("notebooks/output")
        self.template_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_notebooks(self, tickers: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate notebooks for the specified tickers.
        
        Args:
            tickers: List of tickers to generate notebooks for (default: all tracked tickers)
            
        Returns:
            Dict[str, str]: Dictionary mapping tickers to notebook file paths
        """
        try:
            # Get tickers to process
            if tickers is None:
                tickers = config_manager.get_all_tickers()
            
            # Ensure master is included
            if 'master' not in tickers:
                tickers = ['master'] + tickers
            
            # Generate notebooks
            notebook_paths = {}
            
            # Generate master notebook
            master_path = self._generate_master_notebook()
            if master_path:
                notebook_paths['master'] = str(master_path)
            
            # Generate individual notebooks
            for ticker in tickers:
                if ticker == 'master':
                    continue
                
                notebook_path = self._generate_ticker_notebook(ticker)
                if notebook_path:
                    notebook_paths[ticker] = str(notebook_path)
            
            logger.info(f"Generated {len(notebook_paths)} notebooks")
            return notebook_paths
            
        except Exception as e:
            logger.error(f"Error generating notebooks: {str(e)}")
            return {}
    
    def _generate_master_notebook(self) -> Optional[Path]:
        """Generate the master summary notebook.
        
        Returns:
            Optional[Path]: Path to the generated notebook, or None if generation failed
        """
        try:
            # Check if template exists
            template_path = self.template_dir / "master_template.ipynb"
            if not template_path.exists():
                # Copy template from notebooks directory
                source_template = Path("notebooks/master_summary_template.ipynb")
                if source_template.exists():
                    shutil.copy(source_template, template_path)
                else:
                    logger.error(f"Master notebook template not found: {source_template}")
                    return None
            
            # Load template
            with open(template_path, 'r') as f:
                notebook_data = json.load(f)
            
            # Generate notebook
            output_path = self.output_dir / "master_summary.ipynb"
            with open(output_path, 'w') as f:
                json.dump(notebook_data, f, indent=2)
            
            logger.info(f"Generated master notebook: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating master notebook: {str(e)}")
            return None
    
    def _generate_ticker_notebook(self, ticker: str) -> Optional[Path]:
        """Generate a notebook for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Optional[Path]: Path to the generated notebook, or None if generation failed
        """
        try:
            # Check if template exists
            template_path = self.template_dir / "ticker_template.ipynb"
            if not template_path.exists():
                # Copy template from notebooks directory
                source_template = Path("notebooks/volume_analysis_template.ipynb")
                if source_template.exists():
                    shutil.copy(source_template, template_path)
                else:
                    logger.error(f"Ticker notebook template not found: {source_template}")
                    return None
            
            # Load template
            with open(template_path, 'r') as f:
                notebook_data = json.load(f)
            
            # Replace placeholders
            notebook_json = json.dumps(notebook_data)
            notebook_json = notebook_json.replace("{TICKER}", ticker)
            notebook_data = json.loads(notebook_json)
            
            # Generate notebook
            output_path = self.output_dir / f"{ticker}_analysis.ipynb"
            with open(output_path, 'w') as f:
                json.dump(notebook_data, f, indent=2)
            
            logger.info(f"Generated notebook for {ticker}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating notebook for {ticker}: {str(e)}")
            return None
    
    def upload_notebooks_to_colab(self, notebook_paths: Dict[str, str]) -> Dict[str, str]:
        """Upload notebooks to Google Colab and update URLs.
        
        Args:
            notebook_paths: Dictionary mapping tickers to notebook file paths
            
        Returns:
            Dict[str, str]: Dictionary mapping tickers to Colab URLs
        """
        # In a real implementation, this would use the Google Drive API to upload notebooks
        # For now, we'll just simulate the upload by returning the existing URLs
        
        logger.info("Simulating notebook upload to Google Colab")
        
        # Get existing URLs
        notebook_urls = notebook_config.get_all_notebook_urls()
        
        # For each notebook, update the URL if it doesn't exist
        for ticker, path in notebook_paths.items():
            if ticker not in notebook_urls:
                # Generate a fake URL for demonstration purposes
                fake_url = f"https://colab.research.google.com/drive/{ticker.lower()}-analysis-{hash(ticker) % 1000000:06d}"
                notebook_config.set_notebook_url(ticker, fake_url)
                logger.info(f"Set URL for {ticker}: {fake_url}")
        
        return notebook_config.get_all_notebook_urls()
