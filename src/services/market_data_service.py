
import os
import logging
from typing import Dict, List
import pandas as pd
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
    async def fetch_stock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch stock data from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.alpha_vantage_key}"
            response = requests.get(url)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                raise ValueError(f"No data found for symbol {symbol}")
                
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            
            return df.tail(days)
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
            
    async def fetch_institutional_flow(self, symbol: str) -> Dict:
        """Fetch institutional trading data from SEC filings"""
        try:
            # Implementation for SEC filings API
            pass
        except Exception as e:
            logger.error(f"Error fetching institutional data: {str(e)}")
            return {}
