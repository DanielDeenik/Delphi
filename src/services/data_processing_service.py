import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import asyncio
import aiohttp
from ratelimit import limits, sleep_and_retry
import dask.dataframe as dd
from src.data.alpha_vantage_client import AlphaVantageClient
from src.utils.indicators import FinancialIndicators

logger = logging.getLogger(__name__)

class DataProcessingService:
    """Service for processing high-frequency market data"""

    def __init__(self):
        self.alpha_vantage = AlphaVantageClient()
        self.indicators = FinancialIndicators()
        self.batch_size = 100  # Configurable batch size for processing

    async def process_market_data(self, symbols: List[str], start_date: Optional[str] = None) -> Dict:
        """Process market data for multiple symbols asynchronously"""
        try:
            logger.info(f"Starting market data processing for {len(symbols)} symbols")

            # Initialize results container
            results = {
                'timestamp': datetime.now().isoformat(),
                'processed_symbols': 0,
                'failed_symbols': [],
                'data': {}
            }

            # Create batches of symbols for rate limiting
            symbol_batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]

            for batch in symbol_batches:
                async with aiohttp.ClientSession() as session:
                    tasks = [self._process_single_symbol(symbol, session, start_date) for symbol in batch]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for symbol, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to process {symbol}: {str(result)}")
                            results['failed_symbols'].append(symbol)
                        else:
                            results['data'][symbol] = result
                            results['processed_symbols'] += 1

            logger.info(f"Completed processing {results['processed_symbols']} symbols")
            return results

        except Exception as e:
            logger.error(f"Error in market data processing: {str(e)}", exc_info=True)
            raise

    @sleep_and_retry
    @limits(calls=5, period=60)  # Rate limiting for API calls
    async def _process_single_symbol(self, symbol: str, session: aiohttp.ClientSession, 
                                   start_date: Optional[str]) -> Dict:
        """Process data for a single symbol"""
        try:
            # Fetch raw data
            raw_data = await self._fetch_data(symbol, session, start_date)

            # Convert to Dask DataFrame for efficient processing
            ddf = dd.from_pandas(raw_data, npartitions=4)

            # Calculate indicators
            processed_data = self._calculate_indicators(ddf)

            # Compute final results
            results = processed_data.compute()

            return {
                'last_update': datetime.now().isoformat(),
                'data': results.to_dict(orient='records')
            }

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            raise

    async def _fetch_data(self, symbol: str, session: aiohttp.ClientSession, 
                         start_date: Optional[str]) -> pd.DataFrame:
        """Fetch data for a symbol"""
        try:
            data = await self.alpha_vantage.async_fetch_daily_adjusted(symbol, session)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def _calculate_indicators(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """Calculate financial indicators using Dask"""
        try:
            # Volume indicators
            ddf['volume_sma'] = ddf['Volume'].rolling(window=20).mean()
            ddf['volume_ratio'] = ddf['Volume'] / ddf['volume_sma']

            # Volatility indicators
            ddf['returns'] = ddf['Close'].pct_change()
            ddf['volatility'] = ddf['returns'].rolling(window=20).std()

            # Advanced indicators (using traditional methods for now)
            ddf['rsi'] = self.indicators.calculate_rsi(ddf['Close'])
            ddf['obv'] = self.indicators.calculate_obv(ddf['Close'], ddf['Volume'])

            return ddf

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise