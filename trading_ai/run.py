"""
Main entry point for the Volume Intelligence Trading System.
"""
import logging
import asyncio
import argparse
import os
import sys
from datetime import datetime

from trading_ai.config import config_manager
from trading_ai.core.data_ingestion import DataIngestionManager
from trading_ai.core.bigquery_io import BigQueryStorage
from trading_ai.trading.paper_trader import PaperTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"trading_ai_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger(__name__)

async def run_data_ingestion(force_full: bool = False):
    """Run the data ingestion pipeline.
    
    Args:
        force_full: Whether to force a full data refresh
    """
    logger.info("Starting data ingestion...")
    
    # Initialize data ingestion manager
    ingestion_manager = DataIngestionManager()
    
    # Run full ingestion pipeline
    await ingestion_manager.run_full_ingestion()
    
    logger.info("Data ingestion completed")

async def run_volume_analysis(ticker: str = None):
    """Run volume analysis for a ticker or all tickers.
    
    Args:
        ticker: Stock symbol (None for all tickers)
    """
    logger.info(f"Starting volume analysis for {'all tickers' if ticker is None else ticker}...")
    
    # Initialize data ingestion manager
    ingestion_manager = DataIngestionManager()
    
    if ticker:
        # Process volume analysis for a single ticker
        await ingestion_manager.process_volume_analysis(ticker)
    else:
        # Process volume analysis for all tickers
        await ingestion_manager.process_all_volume_analysis()
    
    logger.info("Volume analysis completed")

async def run_sentiment_analysis(ticker: str = None):
    """Run sentiment analysis for a ticker or all tickers.
    
    Args:
        ticker: Stock symbol (None for all tickers)
    """
    logger.info(f"Starting sentiment analysis for {'all tickers' if ticker is None else ticker}...")
    
    # Initialize data ingestion manager
    ingestion_manager = DataIngestionManager()
    
    if ticker:
        # Process sentiment analysis for a single ticker
        await ingestion_manager.process_sentiment_analysis(ticker)
    else:
        # Process sentiment analysis for all tickers
        await ingestion_manager.process_all_sentiment_analysis()
    
    logger.info("Sentiment analysis completed")

async def generate_summary():
    """Generate master summary for all tickers."""
    logger.info("Generating master summary...")
    
    # Initialize data ingestion manager
    ingestion_manager = DataIngestionManager()
    
    # Generate master summary
    summary_df = await ingestion_manager.generate_master_summary()
    
    if not summary_df.empty:
        logger.info(f"Generated master summary for {len(summary_df)} tickers")
    else:
        logger.warning("Failed to generate master summary")

def execute_trade(ticker: str, direction: str, entry_price: float, position_size: float = None,
                 capital_percentage: float = None, stop_loss: float = None, take_profit: float = None,
                 trigger_reason: str = None, notes: str = None):
    """Execute a paper trade.
    
    Args:
        ticker: Stock symbol
        direction: Trade direction ('buy' or 'short')
        entry_price: Entry price
        position_size: Position size (number of shares or contracts)
        capital_percentage: Percentage of capital to allocate (alternative to position_size)
        stop_loss: Stop loss price
        take_profit: Take profit price
        trigger_reason: Reason for entering the trade
        notes: Additional notes
    """
    logger.info(f"Executing {direction} trade for {ticker}...")
    
    # Initialize paper trader
    paper_trader = PaperTrader()
    
    # Execute trade
    trade_id = paper_trader.execute_trade(
        ticker=ticker,
        direction=direction,
        entry_price=entry_price,
        position_size=position_size,
        capital_percentage=capital_percentage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trigger_reason=trigger_reason,
        notes=notes
    )
    
    if trade_id:
        logger.info(f"Successfully executed trade with ID {trade_id}")
    else:
        logger.warning("Failed to execute trade")

def close_trade(trade_id: str, exit_price: float, exit_reason: str = None, notes: str = None):
    """Close a paper trade.
    
    Args:
        trade_id: Trade ID
        exit_price: Exit price
        exit_reason: Reason for exiting the trade
        notes: Additional notes
    """
    logger.info(f"Closing trade {trade_id}...")
    
    # Initialize paper trader
    paper_trader = PaperTrader()
    
    # Close trade
    success = paper_trader.close_trade(
        trade_id=trade_id,
        exit_price=exit_price,
        exit_reason=exit_reason,
        notes=notes
    )
    
    if success:
        logger.info(f"Successfully closed trade {trade_id}")
    else:
        logger.warning(f"Failed to close trade {trade_id}")

def get_account_summary():
    """Get a summary of the paper trading account."""
    logger.info("Getting account summary...")
    
    # Initialize paper trader
    paper_trader = PaperTrader()
    
    # Get account summary
    summary = paper_trader.get_account_summary()
    
    # Print summary
    print("\nAccount Summary:")
    print(f"Initial Capital: ${summary.get('initial_capital', 0):.2f}")
    print(f"Current Capital: ${summary.get('current_capital', 0):.2f}")
    print(f"Unrealized P&L: ${summary.get('unrealized_pnl', 0):.2f}")
    print(f"Account Value: ${summary.get('account_value', 0):.2f}")
    print(f"Total Return: ${summary.get('total_return', 0):.2f} ({summary.get('total_return_pct', 0):.2f}%)")
    print(f"Open Trades: {summary.get('open_trades_count', 0)}")
    print(f"Closed Trades: {summary.get('closed_trades_count', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0) * 100:.2f}%")
    print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
    print(f"Average Profit: ${summary.get('avg_profit', 0):.2f}")
    print(f"Average Loss: ${summary.get('avg_loss', 0):.2f}")
    print(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Volume Intelligence Trading System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data ingestion command
    ingest_parser = subparsers.add_parser("ingest", help="Run data ingestion")
    ingest_parser.add_argument("--force-full", action="store_true", help="Force full data refresh")
    
    # Volume analysis command
    volume_parser = subparsers.add_parser("volume", help="Run volume analysis")
    volume_parser.add_argument("--ticker", help="Stock symbol (None for all tickers)")
    
    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser("sentiment", help="Run sentiment analysis")
    sentiment_parser.add_argument("--ticker", help="Stock symbol (None for all tickers)")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Generate master summary")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute a paper trade")
    trade_parser.add_argument("--ticker", required=True, help="Stock symbol")
    trade_parser.add_argument("--direction", required=True, choices=["buy", "short"], help="Trade direction")
    trade_parser.add_argument("--entry-price", required=True, type=float, help="Entry price")
    trade_parser.add_argument("--position-size", type=float, help="Position size (number of shares)")
    trade_parser.add_argument("--capital-percentage", type=float, help="Percentage of capital to allocate")
    trade_parser.add_argument("--stop-loss", type=float, help="Stop loss price")
    trade_parser.add_argument("--take-profit", type=float, help="Take profit price")
    trade_parser.add_argument("--trigger-reason", help="Reason for entering the trade")
    trade_parser.add_argument("--notes", help="Additional notes")
    
    # Close trade command
    close_parser = subparsers.add_parser("close", help="Close a paper trade")
    close_parser.add_argument("--trade-id", required=True, help="Trade ID")
    close_parser.add_argument("--exit-price", required=True, type=float, help="Exit price")
    close_parser.add_argument("--exit-reason", help="Reason for exiting the trade")
    close_parser.add_argument("--notes", help="Additional notes")
    
    # Account summary command
    account_parser = subparsers.add_parser("account", help="Get account summary")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "ingest":
        await run_data_ingestion(args.force_full)
    elif args.command == "volume":
        await run_volume_analysis(args.ticker)
    elif args.command == "sentiment":
        await run_sentiment_analysis(args.ticker)
    elif args.command == "summary":
        await generate_summary()
    elif args.command == "trade":
        execute_trade(
            ticker=args.ticker,
            direction=args.direction,
            entry_price=args.entry_price,
            position_size=args.position_size,
            capital_percentage=args.capital_percentage,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trigger_reason=args.trigger_reason,
            notes=args.notes
        )
    elif args.command == "close":
        close_trade(
            trade_id=args.trade_id,
            exit_price=args.exit_price,
            exit_reason=args.exit_reason,
            notes=args.notes
        )
    elif args.command == "account":
        get_account_summary()
    else:
        parser.print_help()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
