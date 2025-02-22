
from src.data.alpha_vantage_client import AlphaVantageClient
from src.services.timeseries_storage_service import TimeSeriesStorageService

def main():
    # Initialize services
    storage_service = TimeSeriesStorageService()
    av_client = AlphaVantageClient()
    
    # List of symbols to load
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        data = av_client.get_daily_adjusted(symbol)
        if data is not None:
            success = storage_service.store_market_data(symbol, data)
            print(f"Data load for {symbol}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()
