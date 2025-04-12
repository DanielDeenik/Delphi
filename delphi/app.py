"""
Main application entry point for Delphi.

This module provides the main application entry point for the Delphi platform.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

from delphi.core.data.sources.alpha_vantage import AlphaVantageClient
from delphi.core.data.storage.bigquery import BigQueryStorage
from delphi.core.data.repository import MarketDataRepository
from delphi.core.models.volume.analyzer import VolumeAnalyzer
from delphi.core.models.volume.strategies import SimpleVolumeStrategy
from delphi.core.performance.trade_repository import TradeRepository
from delphi.core.performance.paper_trader import PaperTrader
from delphi.core.rl.service import RLService
from delphi.core.notebooks.generator import NotebookGenerator
from delphi.core.notebooks.launcher import NotebookLauncher
from delphi.utils.config import config_manager, load_env

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"delphi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

class DelphiApp:
    """Main application class for Delphi."""
    
    def __init__(self):
        """Initialize the Delphi application."""
        # Load environment variables
        load_env()
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Initialized Delphi application")
    
    def _initialize_components(self):
        """Initialize application components."""
        try:
            # Get configuration
            self.project_id = config_manager.system_config.google_cloud_project
            self.dataset_id = config_manager.system_config.bigquery_dataset
            self.api_key = config_manager.system_config.alpha_vantage_api_key
            
            # Initialize data components
            self.data_source = AlphaVantageClient(api_key=self.api_key)
            self.storage_service = BigQueryStorage(project_id=self.project_id, dataset_id=self.dataset_id)
            self.data_repository = MarketDataRepository(self.data_source, self.storage_service)
            
            # Initialize analysis components
            self.volume_analyzer = VolumeAnalyzer(strategy=SimpleVolumeStrategy())
            
            # Initialize performance components
            self.trade_repository = TradeRepository(self.storage_service)
            self.paper_trader = PaperTrader(self.trade_repository)
            
            # Initialize RL components
            self.rl_service = RLService(self.storage_service, self.paper_trader)
            
            # Initialize notebook components
            self.notebook_generator = NotebookGenerator(project_id=self.project_id, dataset_id=self.dataset_id)
            self.notebook_launcher = NotebookLauncher()
            
            logger.info("Successfully initialized all components")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def import_data(self, symbols: Optional[List[str]] = None, force_refresh: bool = False, 
                   analyze_volume: bool = True, **kwargs) -> bool:
        """Import data for symbols.
        
        Args:
            symbols: List of symbols to import (default: from config)
            force_refresh: Whether to force refresh from data source
            analyze_volume: Whether to analyze volume patterns
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get symbols from config if not provided
            if symbols is None:
                symbols = config_manager.get_all_tickers()
            
            logger.info(f"Importing data for {len(symbols)} symbols")
            
            # Import data
            results = self.data_repository.get_batch_stock_data(
                symbols,
                force_refresh=force_refresh,
                **kwargs
            )
            
            # Analyze volume if requested
            if analyze_volume:
                logger.info("Analyzing volume patterns")
                
                for symbol, df in results.items():
                    if df.empty:
                        logger.warning(f"No data found for {symbol}")
                        continue
                    
                    # Analyze volume
                    analysis = self.volume_analyzer.analyze(df)
                    
                    # Store results
                    if 'analysis_df' in analysis:
                        self.storage_service.store_volume_analysis(symbol, analysis['analysis_df'])
            
            # Print summary
            success_count = sum(1 for df in results.values() if not df.empty)
            logger.info(f"Successfully imported data for {success_count}/{len(symbols)} symbols")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            return False
    
    def generate_notebooks(self, download_templates: bool = True, 
                          upload: bool = False, launch: bool = False, **kwargs) -> bool:
        """Generate and launch notebooks.
        
        Args:
            download_templates: Whether to download templates
            upload: Whether to upload notebooks to Google Drive
            launch: Whether to launch notebooks in browser
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Download templates if requested
            if download_templates:
                logger.info("Downloading notebook templates")
                self.notebook_generator.download_templates()
            
            # Generate notebooks
            logger.info("Generating notebooks")
            notebook_paths = self.notebook_generator.generate_all_notebooks()
            
            if not notebook_paths:
                logger.error("Failed to generate notebooks")
                return False
            
            logger.info(f"Generated {len(notebook_paths)} notebooks")
            
            # Upload to Google Drive if requested
            notebook_urls = {}
            if upload:
                logger.info("Uploading notebooks to Google Drive")
                notebook_urls = self.notebook_launcher.upload_to_drive(notebook_paths)
                
                if not notebook_urls:
                    logger.error("Failed to upload notebooks to Google Drive")
                    return False
                
                logger.info(f"Uploaded {len(notebook_urls)} notebooks to Google Drive")
            
            # Launch notebooks if requested
            if launch and notebook_urls:
                logger.info("Launching notebooks in browser")
                self.notebook_launcher.launch_notebooks(notebook_urls)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating notebooks: {str(e)}")
            return False
    
    def train_rl_model(self, ticker: str, days: int = 365, agent_type: str = 'dqn', **kwargs) -> bool:
        """Train a reinforcement learning model.
        
        Args:
            ticker: Stock symbol
            days: Number of days of data to use
            agent_type: Type of agent to train ('dqn' or 'ppo')
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            df = self.data_repository.get_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                logger.error(f"No data found for {ticker}")
                return False
            
            # Train model
            logger.info(f"Training {agent_type} model for {ticker}")
            results = self.rl_service.train_agent(ticker, df, agent_type=agent_type, **kwargs)
            
            if 'error' in results:
                logger.error(f"Error training model: {results['error']}")
                return False
            
            logger.info(f"Successfully trained model for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error training RL model: {str(e)}")
            return False
    
    def execute_paper_trades(self, ticker: str, days: int = 30, **kwargs) -> bool:
        """Execute paper trades based on signals.
        
        Args:
            ticker: Stock symbol
            days: Number of days of data to use
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get data
            end_date = datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            df = self.data_repository.get_stock_data(ticker, start_date, end_date)
            
            if df.empty:
                logger.error(f"No data found for {ticker}")
                return False
            
            # Get signals
            signals = self.rl_service.generate_trading_signals(ticker, df, **kwargs)
            
            if signals.empty:
                logger.warning(f"No signals generated for {ticker}")
                return False
            
            # Execute trades
            logger.info(f"Executing paper trades for {ticker}")
            results = self.rl_service.execute_signals(ticker, signals, **kwargs)
            
            if 'error' in results:
                logger.error(f"Error executing trades: {results['error']}")
                return False
            
            logger.info(f"Successfully executed {results['successful_trades']} trades for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing paper trades: {str(e)}")
            return False
    
    def run_command(self, args):
        """Run a command.
        
        Args:
            args: Command-line arguments
        """
        if args.command == 'import':
            self.import_data(
                symbols=args.symbols,
                force_refresh=args.force_refresh,
                analyze_volume=args.analyze_volume
            )
        elif args.command == 'notebooks':
            self.generate_notebooks(
                download_templates=args.download_templates,
                upload=args.upload,
                launch=args.launch
            )
        elif args.command == 'train':
            self.train_rl_model(
                ticker=args.ticker,
                days=args.days,
                agent_type=args.agent_type
            )
        elif args.command == 'trade':
            self.execute_paper_trades(
                ticker=args.ticker,
                days=args.days
            )
        else:
            logger.error(f"Unknown command: {args.command}")

def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Delphi - Financial Analysis Platform")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import data")
    import_parser.add_argument("--symbols", nargs="+", help="Symbols to import (default: from config)")
    import_parser.add_argument("--force-refresh", action="store_true", help="Force refresh from data source")
    import_parser.add_argument("--analyze-volume", action="store_true", help="Analyze volume patterns")
    
    # Notebooks command
    notebooks_parser = subparsers.add_parser("notebooks", help="Generate and launch notebooks")
    notebooks_parser.add_argument("--download-templates", action="store_true", help="Download notebook templates")
    notebooks_parser.add_argument("--upload", action="store_true", help="Upload notebooks to Google Drive")
    notebooks_parser.add_argument("--launch", action="store_true", help="Launch notebooks in browser")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a reinforcement learning model")
    train_parser.add_argument("--ticker", required=True, help="Stock symbol")
    train_parser.add_argument("--days", type=int, default=365, help="Number of days of data to use")
    train_parser.add_argument("--agent-type", choices=["dqn", "ppo"], default="dqn", help="Type of agent to train")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute paper trades")
    trade_parser.add_argument("--ticker", required=True, help="Stock symbol")
    trade_parser.add_argument("--days", type=int, default=30, help="Number of days of data to use")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run command
    if args.command:
        try:
            app = DelphiApp()
            app.run_command(args)
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return 1
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
