"""
Correlation Analyzer

This module provides a class for analyzing correlations between symbols.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import functools
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Class for analyzing correlations between symbols.
    """
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the correlation analyzer.
        
        Args:
            cache_size: Size of the LRU cache for analyze method
        """
        # Apply caching to analyze method
        self.analyze = functools.lru_cache(maxsize=cache_size)(self._analyze_impl)
        
        logger.info("Initialized correlation analyzer")
    
    def _analyze_impl(self, data_hash: tuple) -> Dict[str, Any]:
        """
        Implementation of the analyze method.
        
        Args:
            data_hash: Tuple representation of data dictionary for caching
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        # Convert tuple back to dictionary of DataFrames
        data = {}
        for i in range(0, len(data_hash), 3):
            symbol = data_hash[i]
            index = data_hash[i+1]
            values = data_hash[i+2]
            df = pd.DataFrame(values, index=index)
            data[symbol] = df
        
        try:
            if len(data) < 2:
                logger.error("Not enough data for correlation analysis")
                return {"error": "Not enough data for correlation analysis", "success": False}
            
            # Create DataFrames for price and returns
            price_df = pd.DataFrame()
            return_df = pd.DataFrame()
            volume_df = pd.DataFrame()
            
            # Extract data for each symbol
            for symbol, df in data.items():
                # Add close price
                price_df[symbol] = df['close']
                
                # Add daily returns
                return_df[symbol] = df['close'].pct_change().fillna(0)
                
                # Add volume
                volume_df[symbol] = df['volume']
            
            # Calculate correlations
            price_corr = price_df.corr()
            return_corr = return_df.corr()
            volume_corr = volume_df.corr()
            
            # Calculate rolling correlations (30-day window)
            rolling_corrs = {}
            for symbol1 in data.keys():
                for symbol2 in data.keys():
                    if symbol1 != symbol2:
                        key = f"{symbol1}_{symbol2}"
                        rolling_corrs[key] = return_df[symbol1].rolling(30).corr(return_df[symbol2])
            
            # Calculate VIX-SPY correlation specifically if both exist
            vix_spy_corr = None
            if '^VIX' in return_df.columns and 'SPY' in return_df.columns:
                vix_spy_corr = return_df['^VIX'].corr(return_df['SPY'])
                logger.info(f"VIX-SPY correlation: {vix_spy_corr:.4f}")
            
            # Calculate VIX-PLTR correlation specifically if both exist
            vix_pltr_corr = None
            if '^VIX' in return_df.columns and 'PLTR' in return_df.columns:
                vix_pltr_corr = return_df['^VIX'].corr(return_df['PLTR'])
                logger.info(f"VIX-PLTR correlation: {vix_pltr_corr:.4f}")
            
            # Calculate SPY-PLTR correlation specifically if both exist
            spy_pltr_corr = None
            if 'SPY' in return_df.columns and 'PLTR' in return_df.columns:
                spy_pltr_corr = return_df['SPY'].corr(return_df['PLTR'])
                logger.info(f"SPY-PLTR correlation: {spy_pltr_corr:.4f}")
            
            # Generate signals
            signals = []
            
            # High correlation signals
            high_corr_pairs = []
            for symbol1 in data.keys():
                for symbol2 in data.keys():
                    if symbol1 < symbol2:  # Avoid duplicates
                        corr = return_corr.loc[symbol1, symbol2]
                        if abs(corr) > 0.8:
                            high_corr_pairs.append({
                                "pair": f"{symbol1}-{symbol2}",
                                "correlation": corr,
                                "type": "positive" if corr > 0 else "negative"
                            })
            
            if high_corr_pairs:
                signals.append({
                    "type": "HIGH_CORRELATION",
                    "description": f"High correlation detected between {len(high_corr_pairs)} pairs",
                    "pairs": high_corr_pairs,
                    "strength": "MEDIUM"
                })
            
            # VIX correlation signals
            if vix_spy_corr is not None and vix_spy_corr < -0.7:
                signals.append({
                    "type": "VIX_SPY_CORRELATION",
                    "description": f"Strong negative correlation between VIX and SPY ({vix_spy_corr:.4f})",
                    "strength": "HIGH"
                })
            
            if vix_pltr_corr is not None and abs(vix_pltr_corr) > 0.5:
                signals.append({
                    "type": "VIX_PLTR_CORRELATION",
                    "description": f"Significant correlation between VIX and PLTR ({vix_pltr_corr:.4f})",
                    "strength": "MEDIUM"
                })
            
            # Prepare results
            results = {
                "correlations": {
                    "price_correlation": price_corr.to_dict(),
                    "return_correlation": return_corr.to_dict(),
                    "volume_correlation": volume_corr.to_dict(),
                },
                "rolling_correlations": {k: v.dropna().to_dict() for k, v in rolling_corrs.items()},
                "key_correlations": {
                    "VIX_SPY": vix_spy_corr,
                    "VIX_PLTR": vix_pltr_corr,
                    "SPY_PLTR": spy_pltr_corr
                },
                "summary": {
                    "signals": signals,
                    "high_correlation_pairs": high_corr_pairs
                },
                "success": True
            }
            
            logger.info(f"Correlation analysis completed with {len(signals)} signals")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {"error": str(e), "success": False}
    
    def analyze(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze correlations between symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        # Convert dictionary to hashable format for caching
        data_hash = []
        for symbol, df in data.items():
            data_hash.append(symbol)
            data_hash.append(tuple(df.index))
            data_hash.append(tuple(map(tuple, df.values)))
        
        return self._analyze_impl(tuple(data_hash))
    
    def analyze_parallel(self, data: Dict[str, pd.DataFrame], max_workers: int = 4) -> Dict[str, Any]:
        """
        Analyze correlations between symbols with parallel processing.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            if len(data) < 2:
                logger.error("Not enough data for correlation analysis")
                return {"error": "Not enough data for correlation analysis", "success": False}
            
            # Create DataFrames for price and returns
            price_df = pd.DataFrame()
            return_df = pd.DataFrame()
            volume_df = pd.DataFrame()
            
            # Extract data for each symbol
            for symbol, df in data.items():
                # Add close price
                price_df[symbol] = df['close']
                
                # Add daily returns
                return_df[symbol] = df['close'].pct_change().fillna(0)
                
                # Add volume
                volume_df[symbol] = df['volume']
            
            # Calculate correlations
            price_corr = price_df.corr()
            return_corr = return_df.corr()
            volume_corr = volume_df.corr()
            
            # Calculate rolling correlations (30-day window) in parallel
            rolling_corrs = {}
            
            # Create pairs for parallel processing
            pairs = []
            for symbol1 in data.keys():
                for symbol2 in data.keys():
                    if symbol1 != symbol2:
                        pairs.append((symbol1, symbol2))
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary mapping futures to pairs
                future_to_pair = {
                    executor.submit(self._calculate_rolling_correlation, return_df, symbol1, symbol2): (symbol1, symbol2)
                    for symbol1, symbol2 in pairs
                }
                
                # Process completed futures
                for future in future_to_pair:
                    symbol1, symbol2 = future_to_pair[future]
                    try:
                        rolling_corr = future.result()
                        key = f"{symbol1}_{symbol2}"
                        rolling_corrs[key] = rolling_corr
                    except Exception as e:
                        logger.error(f"Error calculating rolling correlation for {symbol1}-{symbol2}: {str(e)}")
            
            # Calculate key correlations
            vix_spy_corr = None
            if '^VIX' in return_df.columns and 'SPY' in return_df.columns:
                vix_spy_corr = return_df['^VIX'].corr(return_df['SPY'])
                logger.info(f"VIX-SPY correlation: {vix_spy_corr:.4f}")
            
            vix_pltr_corr = None
            if '^VIX' in return_df.columns and 'PLTR' in return_df.columns:
                vix_pltr_corr = return_df['^VIX'].corr(return_df['PLTR'])
                logger.info(f"VIX-PLTR correlation: {vix_pltr_corr:.4f}")
            
            spy_pltr_corr = None
            if 'SPY' in return_df.columns and 'PLTR' in return_df.columns:
                spy_pltr_corr = return_df['SPY'].corr(return_df['PLTR'])
                logger.info(f"SPY-PLTR correlation: {spy_pltr_corr:.4f}")
            
            # Generate signals
            signals = []
            
            # High correlation signals
            high_corr_pairs = []
            for symbol1 in data.keys():
                for symbol2 in data.keys():
                    if symbol1 < symbol2:  # Avoid duplicates
                        corr = return_corr.loc[symbol1, symbol2]
                        if abs(corr) > 0.8:
                            high_corr_pairs.append({
                                "pair": f"{symbol1}-{symbol2}",
                                "correlation": corr,
                                "type": "positive" if corr > 0 else "negative"
                            })
            
            if high_corr_pairs:
                signals.append({
                    "type": "HIGH_CORRELATION",
                    "description": f"High correlation detected between {len(high_corr_pairs)} pairs",
                    "pairs": high_corr_pairs,
                    "strength": "MEDIUM"
                })
            
            # VIX correlation signals
            if vix_spy_corr is not None and vix_spy_corr < -0.7:
                signals.append({
                    "type": "VIX_SPY_CORRELATION",
                    "description": f"Strong negative correlation between VIX and SPY ({vix_spy_corr:.4f})",
                    "strength": "HIGH"
                })
            
            if vix_pltr_corr is not None and abs(vix_pltr_corr) > 0.5:
                signals.append({
                    "type": "VIX_PLTR_CORRELATION",
                    "description": f"Significant correlation between VIX and PLTR ({vix_pltr_corr:.4f})",
                    "strength": "MEDIUM"
                })
            
            # Prepare results
            results = {
                "correlations": {
                    "price_correlation": price_corr.to_dict(),
                    "return_correlation": return_corr.to_dict(),
                    "volume_correlation": volume_corr.to_dict(),
                },
                "rolling_correlations": {k: v.dropna().to_dict() for k, v in rolling_corrs.items()},
                "key_correlations": {
                    "VIX_SPY": vix_spy_corr,
                    "VIX_PLTR": vix_pltr_corr,
                    "SPY_PLTR": spy_pltr_corr
                },
                "summary": {
                    "signals": signals,
                    "high_correlation_pairs": high_corr_pairs
                },
                "success": True
            }
            
            logger.info(f"Correlation analysis completed with {len(signals)} signals")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return {"error": str(e), "success": False}
    
    def _calculate_rolling_correlation(self, return_df: pd.DataFrame, symbol1: str, symbol2: str, window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation between two symbols.
        
        Args:
            return_df: DataFrame with returns
            symbol1: First symbol
            symbol2: Second symbol
            window: Window size for rolling correlation
            
        Returns:
            pd.Series: Rolling correlation
        """
        return return_df[symbol1].rolling(window).corr(return_df[symbol2])
