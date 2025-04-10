"""
Volume Analyzer

This module provides a class for analyzing volume patterns in stock data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import functools
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# Numba-optimized OBV calculation if available
if HAS_NUMBA:
    @numba.jit(nopython=True)
    def calculate_obv(close_values, volume_values):
        """
        Calculate On-Balance Volume (OBV) using Numba for performance.
        
        Args:
            close_values: Array of close prices
            volume_values: Array of volume values
            
        Returns:
            Array of OBV values
        """
        obv = np.zeros_like(volume_values)
        
        for i in range(1, len(close_values)):
            if close_values[i] > close_values[i-1]:
                obv[i] = obv[i-1] + volume_values[i]
            elif close_values[i] < close_values[i-1]:
                obv[i] = obv[i-1] - volume_values[i]
            else:
                obv[i] = obv[i-1]
        
        return obv

class VolumeAnalyzer:
    """
    Class for analyzing volume patterns in stock data.
    """
    
    def __init__(self, cache_size: int = 128):
        """
        Initialize the volume analyzer.
        
        Args:
            cache_size: Size of the LRU cache for analyze method
        """
        # Apply caching to analyze method
        self.analyze = functools.lru_cache(maxsize=cache_size)(self._analyze_impl)
        
        logger.info("Initialized volume analyzer")
    
    def _analyze_impl(self, df_tuple: tuple) -> Dict[str, Any]:
        """
        Implementation of the analyze method.
        
        Args:
            df_tuple: Tuple representation of DataFrame for caching
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Convert tuple back to DataFrame
        df = pd.DataFrame(df_tuple[1], index=df_tuple[0], columns=df_tuple[2])
        
        try:
            if df is None or df.empty:
                logger.error("No data to analyze")
                return {"error": "No data to analyze"}
            
            # Calculate volume moving averages
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ma50'] = df['volume'].rolling(window=50).mean()
            
            # Calculate volume ratios
            df['volume_ratio_5'] = df['volume'] / df['volume_ma5']
            df['volume_ratio_20'] = df['volume'] / df['volume_ma20']
            
            # Identify volume spikes (volume > 2x 20-day average)
            volume_spikes = df[df['volume_ratio_20'] > 2].copy()
            
            # Identify volume drops (volume < 0.5x 20-day average)
            volume_drops = df[df['volume_ratio_20'] < 0.5].copy()
            
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(periods=5)
            
            # Identify price-volume divergences
            # Price up, volume down
            price_up_volume_down = df[(df['price_change'] > 0) & (df['volume_ratio_20'] < 0.8)].copy()
            
            # Price down, volume up
            price_down_volume_up = df[(df['price_change'] < 0) & (df['volume_ratio_20'] > 1.5)].copy()
            
            # Calculate On-Balance Volume (OBV)
            if HAS_NUMBA:
                # Use Numba-optimized function
                df['obv'] = calculate_obv(df['close'].values, df['volume'].values)
            else:
                # Fallback to pandas implementation
                df['obv'] = 0
                for i in range(1, len(df)):
                    if df['close'].iloc[i] > df['close'].iloc[i-1]:
                        df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
                    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                        df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
                    else:
                        df['obv'].iloc[i] = df['obv'].iloc[i-1]
            
            # Calculate OBV moving average
            df['obv_ma20'] = df['obv'].rolling(window=20).mean()
            
            # Identify OBV divergences
            # Price up, OBV down
            obv_bearish_divergence = df[(df['price_change_5d'] > 0) & (df['obv'].diff(5) < 0)].copy()
            
            # Price down, OBV up
            obv_bullish_divergence = df[(df['price_change_5d'] < 0) & (df['obv'].diff(5) > 0)].copy()
            
            # Get recent volume patterns (last 5 days)
            recent_volume = df.iloc[-5:][['close', 'volume', 'volume_ratio_20']].copy() if len(df) >= 5 else pd.DataFrame()
            
            # Determine volume profile
            avg_volume_ratio = df['volume_ratio_20'].mean()
            if avg_volume_ratio > 1.2:
                volume_profile = "HIGH"
            elif avg_volume_ratio < 0.8:
                volume_profile = "LOW"
            else:
                volume_profile = "NORMAL"
            
            # Generate signals
            signals = []
            
            # Volume spike signals
            if len(volume_spikes) > 0 and volume_spikes.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "VOLUME_SPIKE",
                    "description": f"Recent volume spike detected ({volume_spikes['volume_ratio_20'].iloc[-1]:.2f}x average)",
                    "strength": "HIGH"
                })
            
            # Volume drop signals
            if len(volume_drops) > 0 and volume_drops.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "VOLUME_DROP",
                    "description": f"Recent volume drop detected ({volume_drops['volume_ratio_20'].iloc[-1]:.2f}x average)",
                    "strength": "MEDIUM"
                })
            
            # Price-volume divergence signals
            if len(price_up_volume_down) > 0 and price_up_volume_down.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "PRICE_UP_VOLUME_DOWN",
                    "description": "Price increasing on decreasing volume (potential weakness)",
                    "strength": "MEDIUM"
                })
            
            if len(price_down_volume_up) > 0 and price_down_volume_up.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "PRICE_DOWN_VOLUME_UP",
                    "description": "Price decreasing on increasing volume (potential capitulation)",
                    "strength": "HIGH"
                })
            
            # OBV divergence signals
            if len(obv_bearish_divergence) > 0 and obv_bearish_divergence.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "OBV_BEARISH_DIVERGENCE",
                    "description": "Price up but OBV down (potential reversal)",
                    "strength": "HIGH"
                })
            
            if len(obv_bullish_divergence) > 0 and obv_bullish_divergence.index[-1] >= df.index[-5]:
                signals.append({
                    "type": "OBV_BULLISH_DIVERGENCE",
                    "description": "Price down but OBV up (potential reversal)",
                    "strength": "HIGH"
                })
            
            # Store results
            results = {
                "data_points": len(df),
                "patterns": {
                    "volume_spikes": len(volume_spikes),
                    "volume_drops": len(volume_drops),
                    "price_up_volume_down": len(price_up_volume_down),
                    "price_down_volume_up": len(price_down_volume_up),
                    "obv_bearish_divergence": len(obv_bearish_divergence),
                    "obv_bullish_divergence": len(obv_bullish_divergence),
                    "patterns": [
                        {"type": "spike", "date": date, "value": row["volume_ratio_20"]}
                        for date, row in volume_spikes.iterrows()
                    ] + [
                        {"type": "drop", "date": date, "value": row["volume_ratio_20"]}
                        for date, row in volume_drops.iterrows()
                    ]
                },
                "metrics": {
                    "avg_volume": df['volume'].mean(),
                    "max_volume": df['volume'].max(),
                    "min_volume": df['volume'].min(),
                    "avg_volume_ratio": avg_volume_ratio,
                    "last_close": df['close'].iloc[-1] if not df.empty else None,
                    "last_volume": df['volume'].iloc[-1] if not df.empty else None,
                    "last_volume_ratio": df['volume_ratio_20'].iloc[-1] if not df.empty and not np.isnan(df['volume_ratio_20'].iloc[-1]) else None
                },
                "summary": {
                    "volume_profile": volume_profile,
                    "signals": signals
                },
                "success": True
            }
            
            logger.info(f"Analysis completed with {len(signals)} signals")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return {"error": str(e), "success": False}
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume patterns in the data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Convert DataFrame to hashable format for caching
        df_tuple = (tuple(df.index), tuple(map(tuple, df.values)), tuple(df.columns))
        return self._analyze_impl(df_tuple)
