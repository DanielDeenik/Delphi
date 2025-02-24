
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RelativeStrengthAnalyzer:
    def __init__(self):
        self.lookback_periods = [5, 10, 20]  # Multiple timeframes
        self.min_confidence = 0.7
        
    def analyze_sector_strength(self, 
                              stock_data: pd.DataFrame,
                              sector_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze stock's relative strength vs sector"""
        try:
            strength_scores = {}
            
            for period in self.lookback_periods:
                # Calculate returns
                stock_returns = self._calculate_returns(stock_data, period)
                sector_returns = {
                    symbol: self._calculate_returns(data, period)
                    for symbol, data in sector_data.items()
                }
                
                # Rank stocks
                sector_rank = pd.Series(sector_returns).rank(ascending=False)
                strength_scores[f'{period}d_rank'] = sector_rank.get(stock_data['symbol'], 0)
                
            # Calculate institutional flows
            inst_flows = self._analyze_institutional_flows(stock_data['symbol'])
            
            return {
                'strength_scores': strength_scores,
                'institutional_flows': inst_flows,
                'is_sector_leader': self._is_sector_leader(strength_scores),
                'confidence': self._calculate_confidence(strength_scores, inst_flows)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector strength: {str(e)}")
            return {}
            
    def _calculate_returns(self, data: pd.DataFrame, period: int) -> float:
        """Calculate period returns"""
        try:
            return (data['Close'].pct_change(period).fillna(0).mean())
        except Exception:
            return 0.0
            
    def _analyze_institutional_flows(self, symbol: str) -> Dict:
        """Analyze institutional money flows"""
        try:
            return {
                'net_flow': 0.8,  # Placeholder for actual institutional flow data
                'options_flow': 0.6,
                'dark_pool_activity': 0.7
            }
        except Exception:
            return {}
            
    def _is_sector_leader(self, scores: Dict) -> bool:
        """Determine if stock is sector leader"""
        try:
            avg_rank = np.mean([rank for rank in scores.values()])
            return avg_rank <= 2  # Must be in top 2 of sector
        except Exception:
            return False
            
    def _calculate_confidence(self, scores: Dict, flows: Dict) -> float:
        """Calculate overall confidence score"""
        try:
            rank_score = 1 - (np.mean([rank for rank in scores.values()]) / 10)
            flow_score = np.mean([score for score in flows.values()])
            
            return np.clip(0.6 * rank_score + 0.4 * flow_score, 0, 1)
        except Exception:
            return 0.0
