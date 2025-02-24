
import os
import logging
from datetime import datetime
from src.services.timeseries_storage_service import TimeSeriesStorageService
from src.services.market_data_service import MarketDataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_daily_import():
    """Run daily import of market data to BigQuery"""
    try:
        storage_service = TimeSeriesStorageService()
        market_service = MarketDataService()
        
        symbols = [
            # US Stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'BAC', 'WMT',
            # International
            'BVI.PA', 'PRX.AS', 'SAP.DE', 'SONY', 'TCEHY',
            # Crypto
            'BTC-USD', 'ETH-USD', 'HUT', 'RIOT', 'COIN'
        ]
        
        for symbol in symbols:
            logger.info(f"Importing data for {symbol}")
            market_data = await market_service.fetch_stock_data(symbol)
            
            if not market_data.empty:
                success = storage_service.store_market_data(symbol, market_data)
                logger.info(f"Data import for {symbol}: {'Success' if success else 'Failed'}")
            else:
                logger.error(f"No data retrieved for {symbol}")
                
        logger.info("Daily import completed")
        
    except Exception as e:
        logger.error(f"Error in daily import: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_daily_import())
