
import numpy as np
import pandas as pd
from typing import Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketRegimeAlignment:
    def __init__(self):
        self.vix_threshold = 20.0
        self.adx_threshold = 25.0
        self.ma_periods = [20, 50, 200]
        
    def analyze_market_conditions(self, 
                                market_data: pd.DataFrame,
                                institutional_data: Dict) -> Dict:
        """Analyze overall market conditions"""
        try:
            # Calculate VIX-based risk environment
            vix_environment = self._analyze_vix(market_data['vix'])
            
            # Calculate trend strength
            trend_metrics = self._calculate_trend_strength(market_data)
            
            # Analyze institutional flows
            flow_analysis = self._analyze_institutional_flow(institutional_data)
            
            # Calculate market breadth
            breadth = self._calculate_market_breadth(market_data)
            
            # Combine all metrics
            alignment_score = self._calculate_alignment_score(
                vix_environment['score'],
                trend_metrics['trend_strength'],
                flow_analysis['institutional_score'],
                breadth['breadth_score']
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_conditions': {
                    'vix_environment': vix_environment,
                    'trend_strength': trend_metrics,
                    'institutional_flow': flow_analysis,
                    'market_breadth': breadth,
                    'overall_alignment': alignment_score,
                    'is_favorable': alignment_score > 0.7
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}
            
    def _analyze_vix(self, vix_data: pd.Series) -> Dict:
        """Analyze VIX-based risk environment"""
        current_vix = vix_data.iloc[-1]
        vix_ma = vix_data.rolling(20).mean().iloc[-1]
        
        return {
            'current_vix': current_vix,
            'vix_ma': vix_ma,
            'is_low_volatility': current_vix < self.vix_threshold,
            'score': 1.0 - (current_vix / (2 * self.vix_threshold))
        }
        
    def _calculate_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Calculate trend strength using ADX and moving averages"""
        close = data['close']
        
        # Calculate moving averages
        mas = {period: close.rolling(period).mean().iloc[-1] 
               for period in self.ma_periods}
        
        # Calculate ADX
        adx = self._calculate_adx(data)
        
        # Determine trend alignment
        trend_aligned = all(mas[p1] > mas[p2] 
                          for p1, p2 in zip(self.ma_periods, self.ma_periods[1:]))
        
        return {
            'adx': adx,
            'moving_averages': mas,
            'trend_aligned': trend_aligned,
            'trend_strength': min(adx / self.adx_threshold, 1.0)
        }
        
    def _analyze_institutional_flow(self, flow_data: Dict) -> Dict:
        """Analyze institutional money flow"""
        buy_volume = flow_data.get('buy_volume', 0)
        sell_volume = flow_data.get('sell_volume', 0)
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'institutional_score': 0.5}
            
        buy_ratio = buy_volume / total_volume
        return {
            'buy_ratio': buy_ratio,
            'institutional_score': buy_ratio
        }
        
    def _calculate_market_breadth(self, market_data: pd.DataFrame) -> Dict:
        """Calculate market breadth metrics"""
        advances = market_data.get('advances', 0)
        declines = market_data.get('declines', 0)
        total = advances + declines
        
        if total == 0:
            return {'breadth_score': 0.5}
            
        breadth_ratio = advances / total
        return {
            'breadth_ratio': breadth_ratio,
            'breadth_score': breadth_ratio
        }
        
    def _calculate_alignment_score(self,
                                 vix_score: float,
                                 trend_score: float,
                                 institutional_score: float,
                                 breadth_score: float) -> float:
        """Calculate overall market alignment score"""
        weights = {
            'vix': 0.25,
            'trend': 0.30,
            'institutional': 0.25,
            'breadth': 0.20
        }
        
        return (weights['vix'] * vix_score +
                weights['trend'] * trend_score +
                weights['institutional'] * institutional_score +
                weights['breadth'] * breadth_score)
                
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Simplified ADX calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return min(atr * 100 / close.iloc[-1], 100)
