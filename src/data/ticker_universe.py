"""
Ticker Universe

This module provides access to the universe of available stock tickers
from various exchanges.
"""

import os
import logging
import requests
import csv
import time
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)

class TickerUniverse:
    """
    Provides access to the universe of available stock tickers.
    Supports downloading and caching ticker lists from various sources.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the ticker universe.
        
        Args:
            cache_dir: Directory to cache ticker lists (default: data/tickers)
        """
        if cache_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            self.cache_dir = os.path.join(data_dir, 'tickers')
        else:
            self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Sources for ticker lists
        self.sources = {
            'nasdaq': {
                'url': 'https://www.nasdaq.com/api/screener/stocks',
                'cache_file': os.path.join(self.cache_dir, 'nasdaq_tickers.csv'),
                'parser': self._parse_nasdaq_response
            },
            'nyse': {
                'url': 'https://www.nyse.com/api/quotes/filter',
                'cache_file': os.path.join(self.cache_dir, 'nyse_tickers.csv'),
                'parser': self._parse_nyse_response
            },
            # Fallback to a static list if APIs fail
            'default': {
                'url': None,
                'cache_file': os.path.join(self.cache_dir, 'default_tickers.csv'),
                'parser': None
            }
        }
        
        # Initialize cache with default tickers if needed
        self._init_default_tickers()
    
    def _init_default_tickers(self):
        """Initialize the default ticker list if it doesn't exist."""
        default_file = self.sources['default']['cache_file']
        
        if not os.path.exists(default_file):
            # Create a default list of popular tickers
            default_tickers = [
                # Technology
                {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Cyclical'},
                {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Cyclical'},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology'},
                # Financial
                {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'sector': 'Financial Services'},
                {'symbol': 'BAC', 'name': 'Bank of America Corporation', 'exchange': 'NYSE', 'sector': 'Financial Services'},
                {'symbol': 'WFC', 'name': 'Wells Fargo & Company', 'exchange': 'NYSE', 'sector': 'Financial Services'},
                # Healthcare
                {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'sector': 'Healthcare'},
                {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare'},
                {'symbol': 'UNH', 'name': 'UnitedHealth Group Incorporated', 'exchange': 'NYSE', 'sector': 'Healthcare'},
                # Consumer
                {'symbol': 'WMT', 'name': 'Walmart Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Defensive'},
                {'symbol': 'PG', 'name': 'The Procter & Gamble Company', 'exchange': 'NYSE', 'sector': 'Consumer Defensive'},
                {'symbol': 'KO', 'name': 'The Coca-Cola Company', 'exchange': 'NYSE', 'sector': 'Consumer Defensive'},
                # Energy
                {'symbol': 'XOM', 'name': 'Exxon Mobil Corporation', 'exchange': 'NYSE', 'sector': 'Energy'},
                {'symbol': 'CVX', 'name': 'Chevron Corporation', 'exchange': 'NYSE', 'sector': 'Energy'},
                # Communication
                {'symbol': 'VZ', 'name': 'Verizon Communications Inc.', 'exchange': 'NYSE', 'sector': 'Communication Services'},
                {'symbol': 'T', 'name': 'AT&T Inc.', 'exchange': 'NYSE', 'sector': 'Communication Services'},
                # Industrial
                {'symbol': 'BA', 'name': 'The Boeing Company', 'exchange': 'NYSE', 'sector': 'Industrials'},
                {'symbol': 'CAT', 'name': 'Caterpillar Inc.', 'exchange': 'NYSE', 'sector': 'Industrials'},
                {'symbol': 'GE', 'name': 'General Electric Company', 'exchange': 'NYSE', 'sector': 'Industrials'},
                # Utilities
                {'symbol': 'NEE', 'name': 'NextEra Energy, Inc.', 'exchange': 'NYSE', 'sector': 'Utilities'},
                {'symbol': 'DUK', 'name': 'Duke Energy Corporation', 'exchange': 'NYSE', 'sector': 'Utilities'},
                # Real Estate
                {'symbol': 'AMT', 'name': 'American Tower Corporation', 'exchange': 'NYSE', 'sector': 'Real Estate'},
                {'symbol': 'SPG', 'name': 'Simon Property Group, Inc.', 'exchange': 'NYSE', 'sector': 'Real Estate'},
                # Materials
                {'symbol': 'LIN', 'name': 'Linde plc', 'exchange': 'NYSE', 'sector': 'Materials'},
                {'symbol': 'FCX', 'name': 'Freeport-McMoRan Inc.', 'exchange': 'NYSE', 'sector': 'Materials'},
                # ETFs
                {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'exchange': 'NYSE', 'sector': 'ETF'},
                {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'exchange': 'NASDAQ', 'sector': 'ETF'},
                {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'exchange': 'NYSE', 'sector': 'ETF'},
                {'symbol': 'VTI', 'name': 'Vanguard Total Stock Market ETF', 'exchange': 'NYSE', 'sector': 'ETF'},
                {'symbol': 'GLD', 'name': 'SPDR Gold Shares', 'exchange': 'NYSE', 'sector': 'ETF'},
            ]
            
            # Save to CSV
            self._save_tickers_to_csv(default_tickers, default_file)
            logger.info(f"Created default ticker list with {len(default_tickers)} tickers")
    
    def _save_tickers_to_csv(self, tickers: List[Dict], file_path: str):
        """Save tickers to a CSV file."""
        try:
            with open(file_path, 'w', newline='') as f:
                if not tickers:
                    return
                
                # Get fieldnames from the first ticker
                fieldnames = tickers[0].keys()
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tickers)
                
            logger.info(f"Saved {len(tickers)} tickers to {file_path}")
        except Exception as e:
            logger.error(f"Error saving tickers to CSV: {str(e)}")
    
    def _load_tickers_from_csv(self, file_path: str) -> List[Dict]:
        """Load tickers from a CSV file."""
        tickers = []
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Ticker file not found: {file_path}")
                return []
            
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                tickers = list(reader)
                
            logger.info(f"Loaded {len(tickers)} tickers from {file_path}")
        except Exception as e:
            logger.error(f"Error loading tickers from CSV: {str(e)}")
        
        return tickers
    
    def _parse_nasdaq_response(self, response_data) -> List[Dict]:
        """Parse NASDAQ API response."""
        tickers = []
        
        try:
            for item in response_data.get('data', {}).get('rows', []):
                ticker = {
                    'symbol': item.get('symbol', ''),
                    'name': item.get('name', ''),
                    'exchange': item.get('exchange', 'NASDAQ'),
                    'sector': item.get('sector', ''),
                    'industry': item.get('industry', ''),
                    'market_cap': item.get('marketCap', 0)
                }
                tickers.append(ticker)
        except Exception as e:
            logger.error(f"Error parsing NASDAQ response: {str(e)}")
        
        return tickers
    
    def _parse_nyse_response(self, response_data) -> List[Dict]:
        """Parse NYSE API response."""
        tickers = []
        
        try:
            for item in response_data:
                ticker = {
                    'symbol': item.get('symbolTicker', ''),
                    'name': item.get('instrumentName', ''),
                    'exchange': 'NYSE',
                    'sector': item.get('sector', {}).get('sectorName', ''),
                    'industry': item.get('sector', {}).get('industry', {}).get('industryName', '')
                }
                tickers.append(ticker)
        except Exception as e:
            logger.error(f"Error parsing NYSE response: {str(e)}")
        
        return tickers
    
    def _download_tickers(self, source: str) -> List[Dict]:
        """Download tickers from a source."""
        if source not in self.sources:
            logger.error(f"Unknown ticker source: {source}")
            return []
        
        source_info = self.sources[source]
        
        if source_info['url'] is None:
            logger.warning(f"No URL defined for source: {source}")
            return []
        
        try:
            logger.info(f"Downloading tickers from {source}")
            
            response = requests.get(source_info['url'])
            response.raise_for_status()
            
            data = response.json()
            
            tickers = source_info['parser'](data)
            
            # Cache the results
            self._save_tickers_to_csv(tickers, source_info['cache_file'])
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error downloading tickers from {source}: {str(e)}")
            return []
    
    def get_tickers(self, source: str = 'default', refresh: bool = False) -> List[Dict]:
        """
        Get tickers from a source.
        
        Args:
            source: Source name ('nasdaq', 'nyse', or 'default')
            refresh: Whether to refresh the cache
            
        Returns:
            List of ticker dictionaries
        """
        if source not in self.sources:
            logger.error(f"Unknown ticker source: {source}")
            return []
        
        source_info = self.sources[source]
        cache_file = source_info['cache_file']
        
        # Check if we need to download fresh data
        if refresh or not os.path.exists(cache_file):
            tickers = self._download_tickers(source)
            if tickers:
                return tickers
        
        # Load from cache
        return self._load_tickers_from_csv(cache_file)
    
    def get_all_tickers(self, refresh: bool = False) -> List[Dict]:
        """
        Get all tickers from all sources.
        
        Args:
            refresh: Whether to refresh the cache
            
        Returns:
            List of ticker dictionaries
        """
        all_tickers = []
        seen_symbols = set()
        
        for source in ['nasdaq', 'nyse', 'default']:
            tickers = self.get_tickers(source, refresh)
            
            for ticker in tickers:
                symbol = ticker.get('symbol')
                if symbol and symbol not in seen_symbols:
                    all_tickers.append(ticker)
                    seen_symbols.add(symbol)
        
        logger.info(f"Got {len(all_tickers)} unique tickers from all sources")
        return all_tickers
    
    def get_symbols(self, source: str = 'default', refresh: bool = False) -> List[str]:
        """
        Get just the ticker symbols from a source.
        
        Args:
            source: Source name ('nasdaq', 'nyse', or 'default')
            refresh: Whether to refresh the cache
            
        Returns:
            List of ticker symbols
        """
        tickers = self.get_tickers(source, refresh)
        return [t.get('symbol') for t in tickers if t.get('symbol')]
    
    def get_all_symbols(self, refresh: bool = False) -> List[str]:
        """
        Get all ticker symbols from all sources.
        
        Args:
            refresh: Whether to refresh the cache
            
        Returns:
            List of ticker symbols
        """
        tickers = self.get_all_tickers(refresh)
        return [t.get('symbol') for t in tickers if t.get('symbol')]
    
    def filter_tickers(self, 
                      exchange: Optional[str] = None,
                      sector: Optional[str] = None,
                      min_market_cap: Optional[float] = None) -> List[Dict]:
        """
        Filter tickers based on criteria.
        
        Args:
            exchange: Filter by exchange (e.g., 'NYSE', 'NASDAQ')
            sector: Filter by sector
            min_market_cap: Minimum market cap (in billions)
            
        Returns:
            Filtered list of ticker dictionaries
        """
        all_tickers = self.get_all_tickers()
        filtered = []
        
        for ticker in all_tickers:
            # Apply exchange filter
            if exchange and ticker.get('exchange') != exchange:
                continue
            
            # Apply sector filter
            if sector and ticker.get('sector') != sector:
                continue
            
            # Apply market cap filter
            if min_market_cap is not None:
                market_cap = ticker.get('market_cap', 0)
                if isinstance(market_cap, str):
                    try:
                        market_cap = float(market_cap.replace(',', ''))
                    except:
                        market_cap = 0
                
                # Convert to billions
                market_cap_billions = market_cap / 1_000_000_000
                
                if market_cap_billions < min_market_cap:
                    continue
            
            filtered.append(ticker)
        
        return filtered


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize ticker universe
    universe = TickerUniverse()
    
    # Get default tickers
    default_tickers = universe.get_tickers()
    print(f"Default tickers: {len(default_tickers)}")
    
    # Get all symbols
    all_symbols = universe.get_all_symbols()
    print(f"All symbols: {len(all_symbols)}")
    
    # Print first 10 symbols
    print(f"First 10 symbols: {all_symbols[:10]}")
