
import numpy as np
import pandas as pd
import talib
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class MarketStructureAnalyzer:
    def __init__(self):
        self.bb_period = 20
        self.volume_threshold = 2.0  # Volume spike threshold
        self.compression_threshold = 0.02  # 2% band compression
        
    def analyze_market_structure(self, 
                               price_data: pd.DataFrame,
                               volume_data: pd.DataFrame) -> Dict:
        """Analyze market structure for breakouts and consolidation"""
        try:
            # Calculate Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                price_data['close'].values,
                timeperiod=self.bb_period
            )
            
            # Calculate ATR for volatility
            atr = talib.ATR(
                price_data['high'].values,
                price_data['low'].values,
                price_data['close'].values,
                timeperiod=14
            )
            
            # Detect compression
            band_width = (upper - lower) / middle
            is_compressed = band_width[-1] < self.compression_threshold
            
            # Check for breakout
            last_close = price_data['close'].iloc[-1]
            is_breakout = last_close > upper[-1]
            
            # Volume confirmation
            avg_volume = volume_data['volume'].rolling(20).mean()
            volume_spike = volume_data['volume'].iloc[-1] > self.volume_threshold * avg_volume.iloc[-1]
            
            return {
                'is_breakout': is_breakout,
                'is_compressed': is_compressed,
                'volume_confirmed': volume_spike,
                'breakout_strength': self._calculate_breakout_strength(
                    last_close, upper[-1], atr[-1]
                ),
                'stop_level': lower[-1],
                'confidence_score': self._calculate_confidence(
                    is_breakout, volume_spike, band_width[-1]
                )
            }
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return {}
            
    def _calculate_breakout_strength(self,
                                   close: float,
                                   upper_band: float,
                                   atr: float) -> float:
        """Calculate relative strength of breakout"""
        if close <= upper_band:
            return 0.0
        return (close - upper_band) / atr
        
    def _calculate_confidence(self,
                            is_breakout: bool,
                            volume_confirmed: bool,
                            band_width: float) -> float:
        """Calculate confidence score for the breakout"""
        if not is_breakout:
            return 0.0
            
        score = 0.5  # Base score for breakout
        if volume_confirmed:
            score += 0.3  # Volume confirmation
        if band_width < self.compression_threshold:
            score += 0.2  # Tight consolidation
            
        return score
