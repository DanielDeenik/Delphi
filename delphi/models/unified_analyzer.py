"""
Unified Analyzer

This module provides a unified analyzer that combines volume and correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor

from .volume_analyzer import VolumeAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from ..utils.parallel import parallel_process

logger = logging.getLogger(__name__)

class UnifiedAnalyzer:
    """
    Unified analyzer that combines volume and correlation analysis.
    """
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the unified analyzer.
        
        Args:
            cache_size: Size of the LRU cache for analyze methods
        """
        self.volume_analyzer = VolumeAnalyzer(cache_size=cache_size)
        self.correlation_analyzer = CorrelationAnalyzer(cache_size=cache_size)
        
        logger.info("Initialized unified analyzer")
    
    def analyze(self, data: Dict[str, pd.DataFrame], max_workers: int = 4) -> Dict[str, Any]:
        """
        Analyze data with volume and correlation analysis.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            if not data:
                logger.error("No data to analyze")
                return {"error": "No data to analyze", "success": False}
            
            # Analyze volume patterns in parallel
            volume_results = {}
            
            # Define function for parallel processing
            def analyze_volume(symbol):
                try:
                    df = data[symbol]
                    result = self.volume_analyzer.analyze(df)
                    return symbol, result
                except Exception as e:
                    logger.error(f"Error analyzing volume patterns for {symbol}: {str(e)}")
                    return symbol, {"error": str(e), "success": False}
            
            # Analyze volume patterns in parallel
            results = parallel_process(list(data.keys()), analyze_volume, max_workers=max_workers)
            for symbol, result in results:
                volume_results[symbol] = result
            
            # Analyze correlations
            correlation_results = self.correlation_analyzer.analyze_parallel(data, max_workers=max_workers)
            
            # Combine results
            unified_results = {
                "volume_analysis": volume_results,
                "correlation_analysis": correlation_results,
                "success": True
            }
            
            # Generate unified signals
            signals = []
            
            # Add volume signals
            for symbol, result in volume_results.items():
                if 'summary' in result and 'signals' in result['summary']:
                    for signal in result['summary']['signals']:
                        signals.append({
                            "symbol": symbol,
                            "type": f"VOLUME_{signal['type']}",
                            "description": signal['description'],
                            "strength": signal['strength']
                        })
            
            # Add correlation signals
            if 'summary' in correlation_results and 'signals' in correlation_results['summary']:
                for signal in correlation_results['summary']['signals']:
                    signals.append({
                        "symbol": "MULTI",
                        "type": f"CORRELATION_{signal['type']}",
                        "description": signal['description'],
                        "strength": signal['strength']
                    })
            
            # Add combined signals
            # Example: Volume spike in multiple symbols
            volume_spike_symbols = []
            for symbol, result in volume_results.items():
                if 'summary' in result and 'signals' in result['summary']:
                    for signal in result['summary']['signals']:
                        if signal['type'] == 'VOLUME_SPIKE':
                            volume_spike_symbols.append(symbol)
            
            if len(volume_spike_symbols) > 1:
                signals.append({
                    "symbol": "MULTI",
                    "type": "COMBINED_VOLUME_SPIKE",
                    "description": f"Volume spike detected in multiple symbols: {', '.join(volume_spike_symbols)}",
                    "strength": "HIGH"
                })
            
            # Add signals to unified results
            unified_results["signals"] = signals
            
            logger.info(f"Unified analysis completed with {len(signals)} signals")
            return unified_results
            
        except Exception as e:
            logger.error(f"Error in unified analysis: {str(e)}")
            return {"error": str(e), "success": False}
