"""
Notebook generator module for Delphi.

This module provides a class for generating Google Colab notebooks.
"""
from typing import Dict, List, Optional, Any, Union
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import functools

from delphi.core.base.service import Service
from delphi.core.notebooks.templates import NotebookTemplate

# Configure logger
logger = logging.getLogger(__name__)

class NotebookGenerator(Service):
    """Service for generating Google Colab notebooks."""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None,
                template_dir: Optional[str] = None, output_dir: Optional[str] = None,
                cache_size: int = 128, **kwargs):
        """Initialize the notebook generator.
        
        Args:
            project_id: Google Cloud project ID (default: from environment)
            dataset_id: BigQuery dataset ID (default: from environment)
            template_dir: Directory containing notebook templates (default: 'templates/notebooks')
            output_dir: Directory for output notebooks (default: 'notebooks')
            cache_size: Size of the LRU cache for service methods
            **kwargs: Additional arguments
        """
        super().__init__(cache_size=cache_size, **kwargs)
        
        # Get project ID from environment if not provided
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "delphi-449908")
        
        # Get dataset ID from environment if not provided
        self.dataset_id = dataset_id or os.environ.get("BIGQUERY_DATASET", "trading_insights")
        
        # Set template and output directories
        self.template_dir = Path(template_dir or 'templates/notebooks')
        self.output_dir = Path(output_dir or 'notebooks')
        
        # Create directories if they don't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath('individual').mkdir(parents=True, exist_ok=True)
        
        # Initialize templates
        self.templates = {}
        
        logger.info(f"Initialized notebook generator for project {self.project_id}")
    
    def _apply_caching(self):
        """Apply LRU caching to service methods."""
        # Apply caching to load_template
        self._load_template_impl = self.load_template
        self.load_template = functools.lru_cache(maxsize=self.cache_size)(self._load_template_impl)
    
    def clear_cache(self):
        """Clear the LRU cache for service methods."""
        self.load_template.cache_clear()
        logger.debug("Cleared cache for notebook generator")
    
    def initialize(self, **kwargs) -> bool:
        """Initialize the notebook generator.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            True if initialization is successful, False otherwise
        """
        # Load templates
        return self.load_templates()
    
    def load_templates(self) -> bool:
        """Load notebook templates.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if template directory exists
            if not self.template_dir.exists():
                logger.warning(f"Template directory {self.template_dir} does not exist")
                return False
            
            # Load templates
            for template_path in self.template_dir.glob('*.ipynb'):
                template_name = template_path.stem
                
                # Load template
                template = self.load_template(template_name)
                
                if template is not None:
                    self.templates[template_name] = template
            
            logger.info(f"Loaded {len(self.templates)} templates")
            return True
            
        except Exception as e:
            logger.error(f"Error loading templates: {str(e)}")
            return False
    
    def load_template(self, template_name: str) -> Optional[NotebookTemplate]:
        """Load a notebook template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            Notebook template or None if not found
        """
        try:
            # Build template path
            template_path = self.template_dir / f"{template_name}.ipynb"
            
            # Check if template exists
            if not template_path.exists():
                logger.warning(f"Template not found: {template_path}")
                return None
            
            # Load template
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = json.load(f)
            
            # Create template
            template = NotebookTemplate(template_name, template_content)
            
            logger.info(f"Loaded template: {template_name}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {str(e)}")
            return None
    
    def load_tracked_stocks(self, config_path: Optional[str] = None) -> Dict[str, List[str]]:
        """Load tracked stocks from configuration.
        
        Args:
            config_path: Path to tracked stocks configuration file
            
        Returns:
            Dictionary with tracked stocks
        """
        try:
            # Use default path if not provided
            if config_path is None:
                config_path = 'config/tracked_stocks.json'
            
            # Load configuration
            with open(config_path, 'r') as f:
                tracked_stocks = json.load(f)
            
            logger.info(f"Loaded {len(tracked_stocks['buy']) + len(tracked_stocks['short'])} tracked stocks")
            return tracked_stocks
            
        except Exception as e:
            logger.error(f"Error loading tracked stocks: {str(e)}")
            
            # Return default tracked stocks
            default_stocks = {
                "buy": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", 
                       "TSLA", "META", "ADBE", "ORCL", "ASML"],
                "short": ["BIDU", "NIO", "PINS", "SNAP", "COIN", 
                         "PLTR", "UBER", "LCID", "INTC", "XPEV"]
            }
            
            logger.info(f"Using default tracked stocks: {len(default_stocks['buy']) + len(default_stocks['short'])} stocks")
            return default_stocks
    
    def generate_stock_notebook(self, ticker: str, direction: str) -> Optional[Path]:
        """Generate a notebook for a stock.
        
        Args:
            ticker: Stock symbol
            direction: Trading direction ('buy' or 'short')
            
        Returns:
            Path to the generated notebook or None if failed
        """
        try:
            # Load template
            template = self.load_template('stock_analysis_template')
            if template is None:
                logger.error("Stock analysis template not found")
                return None
            
            # Create variables
            variables = {
                'TICKER': ticker,
                'PROJECT_ID': self.project_id,
                'DATASET': self.dataset_id,
                'DIRECTION': direction,
                'SPIKE_COUNT': '0',
                'LATEST_Z_SCORE': '0.0',
                'CURRENT_SIGNAL': 'NEUTRAL'
            }
            
            # Generate notebook
            notebook_content = template.render(variables)
            
            # Save notebook
            output_path = self.output_dir / 'individual' / f"{ticker}_analysis.ipynb"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2)
            
            logger.info(f"Generated notebook for {ticker}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating notebook for {ticker}: {str(e)}")
            return None
    
    def generate_master_notebook(self) -> Optional[Path]:
        """Generate the master summary notebook.
        
        Returns:
            Path to the generated notebook or None if failed
        """
        try:
            # Load template
            template = self.load_template('master_summary_template')
            if template is None:
                logger.error("Master summary template not found")
                return None
            
            # Load tracked stocks
            tracked_stocks = self.load_tracked_stocks()
            
            # Create variables
            variables = {
                'PROJECT_ID': self.project_id,
                'DATASET': self.dataset_id,
                'TRACKED_STOCKS_JSON': json.dumps(tracked_stocks, indent=2),
                'BUY_SIGNAL_COUNT': '0',
                'SHORT_SIGNAL_COUNT': '0',
                'TOP_BUY_TICKER': 'None',
                'TOP_SHORT_TICKER': 'None'
            }
            
            # Generate notebook
            notebook_content = template.render(variables)
            
            # Save notebook
            output_path = self.output_dir / "master_summary.ipynb"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2)
            
            logger.info(f"Generated master summary notebook: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating master summary notebook: {str(e)}")
            return None
    
    def generate_performance_notebook(self) -> Optional[Path]:
        """Generate the performance dashboard notebook.
        
        Returns:
            Path to the generated notebook or None if failed
        """
        try:
            # Load template
            template = self.load_template('performance_dashboard_template')
            if template is None:
                logger.error("Performance dashboard template not found")
                return None
            
            # Create variables
            variables = {
                'PROJECT_ID': self.project_id,
                'DATASET': self.dataset_id,
                'TOTAL_TRADES': '0',
                'WIN_RATE': '0.0',
                'PROFIT_FACTOR': '0.0',
                'SHARPE_RATIO': '0.0'
            }
            
            # Generate notebook
            notebook_content = template.render(variables)
            
            # Save notebook
            output_path = self.output_dir / "performance_dashboard.ipynb"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2)
            
            logger.info(f"Generated performance dashboard notebook: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard notebook: {str(e)}")
            return None
    
    def generate_model_training_notebook(self) -> Optional[Path]:
        """Generate the model training notebook.
        
        Returns:
            Path to the generated notebook or None if failed
        """
        try:
            # Load template
            template = self.load_template('model_training_template')
            if template is None:
                logger.error("Model training template not found")
                return None
            
            # Create variables
            variables = {
                'PROJECT_ID': self.project_id,
                'DATASET': self.dataset_id
            }
            
            # Generate notebook
            notebook_content = template.render(variables)
            
            # Save notebook
            output_path = self.output_dir / "model_training.ipynb"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_content, f, indent=2)
            
            logger.info(f"Generated model training notebook: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating model training notebook: {str(e)}")
            return None
    
    def generate_all_notebooks(self) -> List[Path]:
        """Generate all notebooks.
        
        Returns:
            List of paths to generated notebooks
        """
        # Load tracked stocks
        tracked_stocks = self.load_tracked_stocks()
        
        # Generate individual notebooks
        notebook_paths = []
        
        for direction, tickers in tracked_stocks.items():
            for ticker in tickers:
                notebook_path = self.generate_stock_notebook(ticker, direction)
                if notebook_path:
                    notebook_paths.append(notebook_path)
        
        # Generate master notebook
        master_path = self.generate_master_notebook()
        if master_path:
            notebook_paths.append(master_path)
        
        # Generate performance dashboard
        performance_path = self.generate_performance_notebook()
        if performance_path:
            notebook_paths.append(performance_path)
        
        # Generate model training notebook
        training_path = self.generate_model_training_notebook()
        if training_path:
            notebook_paths.append(training_path)
        
        logger.info(f"Generated {len(notebook_paths)} notebooks")
        return notebook_paths
    
    def download_templates(self, force: bool = False) -> bool:
        """Download notebook templates from GitHub.
        
        Args:
            force: Whether to force download even if templates exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            # Template URLs
            template_urls = {
                'stock_analysis_template': 'https://raw.githubusercontent.com/DanielDeenik/Delphi/feature/volume-analysis/notebooks/stock_analysis_template.ipynb',
                'master_summary_template': 'https://raw.githubusercontent.com/DanielDeenik/Delphi/feature/volume-analysis/notebooks/master_summary_template.ipynb',
                'performance_dashboard_template': 'https://raw.githubusercontent.com/DanielDeenik/Delphi/feature/volume-analysis/notebooks/performance_dashboard_template.ipynb',
                'model_training_template': 'https://raw.githubusercontent.com/DanielDeenik/Delphi/feature/volume-analysis/notebooks/model_training_template.ipynb'
            }
            
            # Download each template
            for template_name, url in template_urls.items():
                template_path = self.template_dir / f"{template_name}.ipynb"
                
                # Skip if template exists and not forcing download
                if template_path.exists() and not force:
                    logger.info(f"Template {template_name} already exists, skipping download")
                    continue
                
                # Download template
                logger.info(f"Downloading template {template_name} from {url}")
                response = requests.get(url)
                
                if response.status_code == 200:
                    with open(template_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.info(f"Downloaded template {template_name}")
                else:
                    logger.error(f"Failed to download template {template_name}: {response.status_code}")
                    return False
            
            # Reload templates
            self.load_templates()
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading templates: {str(e)}")
            return False
