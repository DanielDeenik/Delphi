"""
Batch downloader for retrieving stock data from Alpha Vantage in batches.
"""
import os
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from tqdm import tqdm

from volume_trading.config import config
from volume_trading.core.data_storage import DataStorage

# Configure logging
logger = logging.getLogger(__name__)

class BatchDownloader:
    """Downloader for retrieving stock data in batches."""
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 5, 
                 delay_between_batches: int = 60, max_retries: int = 3):
        """Initialize the batch downloader.
        
        Args:
            api_key: Alpha Vantage API key (optional, will use from config if not provided)
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay in seconds between batches
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key or config.get("alpha_vantage_api_key")
        self.base_url = "https://www.alphavantage.co/query"
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.max_retries = max_retries
        self.storage = DataStorage()
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
    
    def download_all_stocks(self, tickers: Optional[List[str]] = None, 
                           force_refresh: bool = False) -> Dict[str, bool]:
        """Download data for all stocks in batches.
        
        Args:
            tickers: List of tickers to download (None for all tracked tickers)
            force_refresh: Whether to force a full refresh
            
        Returns:
            Dictionary mapping tickers to success status
        """
        # Get tickers if not provided
        if tickers is None:
            tickers = config.get_all_tickers()
        
        logger.info(f"Downloading data for {len(tickers)} stocks in batches of {self.batch_size}...")
        
        # Split tickers into batches
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        logger.info(f"Split into {len(batches)} batches")
        
        # Process each batch
        results = {}
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {', '.join(batch)}")
            
            # Process batch
            batch_results = self._process_batch(batch, force_refresh)
            results.update(batch_results)
            
            # Wait between batches (except after the last batch)
            if i < len(batches) - 1:
                logger.info(f"Waiting {self.delay_between_batches} seconds before next batch...")
                time.sleep(self.delay_between_batches)
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Download completed: {success_count}/{len(tickers)} successful")
        
        return results
    
    def _process_batch(self, tickers: List[str], force_refresh: bool) -> Dict[str, bool]:
        """Process a batch of tickers.
        
        Args:
            tickers: List of tickers to process
            force_refresh: Whether to force a full refresh
            
        Returns:
            Dictionary mapping tickers to success status
        """
        results = {}
        
        # Create progress bar
        with tqdm(total=len(tickers), desc="Downloading", unit="stock") as pbar:
            for ticker in tickers:
                # Process ticker
                success = self._download_stock_data(ticker, force_refresh)
                results[ticker] = success
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"ticker": ticker, "success": success})
                
                # Add a small delay between requests to avoid hitting rate limits
                time.sleep(12)  # 5 requests per minute = 12 seconds between requests
        
        return results
    
    def _download_stock_data(self, ticker: str, force_refresh: bool) -> bool:
        """Download data for a single stock.
        
        Args:
            ticker: Stock symbol
            force_refresh: Whether to force a full refresh
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we already have data
        existing_df = self.storage.load_price_data(ticker)
        
        # Determine output size
        outputsize = "full" if force_refresh or existing_df.empty else "compact"
        
        # Try to download with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": outputsize,
                    "apikey": self.api_key
                }
                
                # Make request
                logger.info(f"Fetching {outputsize} data for {ticker} (attempt {attempt+1}/{self.max_retries})...")
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                # Check for errors
                if "Error Message" in data:
                    logger.error(f"Error fetching data for {ticker}: {data['Error Message']}")
                    time.sleep(2)  # Wait before retry
                    continue
                
                if "Note" in data:
                    logger.warning(f"API limit reached: {data['Note']}")
                    time.sleep(60)  # Wait a minute before retry
                    continue
                
                # Parse data
                if "Time Series (Daily)" not in data:
                    logger.error(f"Unexpected response format for {ticker}")
                    time.sleep(2)  # Wait before retry
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                
                # Rename columns
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                    "7. dividend amount": "dividend",
                    "8. split coefficient": "split_coefficient"
                })
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Add symbol column
                df["symbol"] = ticker
                
                # Convert index to datetime
                df.index = pd.to_datetime(df.index)
                
                # Sort by date (newest first)
                df = df.sort_index(ascending=False)
                
                # Save data
                if not df.empty:
                    self.storage.save_price_data(ticker, df)
                    logger.info(f"Successfully downloaded and saved {len(df)} rows for {ticker}")
                    return True
                else:
                    logger.warning(f"Empty DataFrame for {ticker}")
                    return False
                
            except Exception as e:
                logger.error(f"Error downloading data for {ticker} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                time.sleep(2)  # Wait before retry
        
        logger.error(f"Failed to download data for {ticker} after {self.max_retries} attempts")
        return False
    
    async def download_all_stocks_async(self, tickers: Optional[List[str]] = None, 
                                      force_refresh: bool = False) -> Dict[str, bool]:
        """Download data for all stocks in batches asynchronously.
        
        Args:
            tickers: List of tickers to download (None for all tracked tickers)
            force_refresh: Whether to force a full refresh
            
        Returns:
            Dictionary mapping tickers to success status
        """
        # Get tickers if not provided
        if tickers is None:
            tickers = config.get_all_tickers()
        
        logger.info(f"Downloading data for {len(tickers)} stocks in batches of {self.batch_size} asynchronously...")
        
        # Split tickers into batches
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        logger.info(f"Split into {len(batches)} batches")
        
        # Process each batch
        results = {}
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {', '.join(batch)}")
            
            # Process batch asynchronously
            batch_results = await self._process_batch_async(batch, force_refresh)
            results.update(batch_results)
            
            # Wait between batches (except after the last batch)
            if i < len(batches) - 1:
                logger.info(f"Waiting {self.delay_between_batches} seconds before next batch...")
                await asyncio.sleep(self.delay_between_batches)
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Download completed: {success_count}/{len(tickers)} successful")
        
        return results
    
    async def _process_batch_async(self, tickers: List[str], force_refresh: bool) -> Dict[str, bool]:
        """Process a batch of tickers asynchronously.
        
        Args:
            tickers: List of tickers to process
            force_refresh: Whether to force a full refresh
            
        Returns:
            Dictionary mapping tickers to success status
        """
        # Create tasks for each ticker
        tasks = []
        for ticker in tickers:
            task = asyncio.create_task(self._download_stock_data_async(ticker, force_refresh))
            tasks.append((ticker, task))
        
        # Wait for all tasks to complete
        results = {}
        for ticker, task in tasks:
            try:
                success = await task
                results[ticker] = success
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                results[ticker] = False
        
        return results
    
    async def _download_stock_data_async(self, ticker: str, force_refresh: bool) -> bool:
        """Download data for a single stock asynchronously.
        
        Args:
            ticker: Stock symbol
            force_refresh: Whether to force a full refresh
            
        Returns:
            True if successful, False otherwise
        """
        # Check if we already have data
        existing_df = self.storage.load_price_data(ticker)
        
        # Determine output size
        outputsize = "full" if force_refresh or existing_df.empty else "compact"
        
        # Try to download with retries
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": outputsize,
                    "apikey": self.api_key
                }
                
                # Make request
                logger.info(f"Fetching {outputsize} data for {ticker} (attempt {attempt+1}/{self.max_retries})...")
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        data = await response.json()
                
                # Check for errors
                if "Error Message" in data:
                    logger.error(f"Error fetching data for {ticker}: {data['Error Message']}")
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                
                if "Note" in data:
                    logger.warning(f"API limit reached: {data['Note']}")
                    await asyncio.sleep(60)  # Wait a minute before retry
                    continue
                
                # Parse data
                if "Time Series (Daily)" not in data:
                    logger.error(f"Unexpected response format for {ticker}")
                    await asyncio.sleep(2)  # Wait before retry
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
                
                # Rename columns
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. adjusted close": "adjusted_close",
                    "6. volume": "volume",
                    "7. dividend amount": "dividend",
                    "8. split coefficient": "split_coefficient"
                })
                
                # Convert to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col])
                
                # Add symbol column
                df["symbol"] = ticker
                
                # Convert index to datetime
                df.index = pd.to_datetime(df.index)
                
                # Sort by date (newest first)
                df = df.sort_index(ascending=False)
                
                # Save data
                if not df.empty:
                    self.storage.save_price_data(ticker, df)
                    logger.info(f"Successfully downloaded and saved {len(df)} rows for {ticker}")
                    return True
                else:
                    logger.warning(f"Empty DataFrame for {ticker}")
                    return False
                
            except Exception as e:
                logger.error(f"Error downloading data for {ticker} (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(2)  # Wait before retry
        
        logger.error(f"Failed to download data for {ticker} after {self.max_retries} attempts")
        return False
