
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradeAnalysis:
    def __init__(self):
        self.factor_weights = {
            'alpha': 0.4,
            'beta': 0.3,
            'sentiment': 0.2,
            'geopolitical': 0.1
        }

    def analyze_stop_loss_trigger(self, trade_data: Dict, market_data: pd.DataFrame) -> Dict:
        """Analyze why a stop loss was triggered"""
        try:
            # Calculate factor contributions
            alpha_impact = self._analyze_alpha_factors(trade_data)
            beta_impact = self._analyze_beta_factors(trade_data, market_data)
            sentiment_impact = self._analyze_sentiment_factors(trade_data)
            geo_impact = self._analyze_geopolitical_factors(trade_data)

            # Determine primary cause
            factor_impacts = {
                'alpha': alpha_impact,
                'beta': beta_impact,
                'sentiment': sentiment_impact,
                'geopolitical': geo_impact
            }

            primary_factor = max(factor_impacts.items(), key=lambda x: x[1])[0]

            return {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_data.get('trade_id'),
                'stop_loss_price': trade_data.get('stop_loss'),
                'factor_attribution': factor_impacts,
                'primary_factor': primary_factor,
                'recommendations': self._generate_recommendations(primary_factor, factor_impacts)
            }

        except Exception as e:
            logger.error(f"Error analyzing stop loss trigger: {str(e)}")
            return {}

    def _analyze_alpha_factors(self, trade_data: Dict) -> float:
        """Analyze company-specific factors"""
        try:
            earnings_impact = trade_data.get('earnings_surprise', 0)
            revenue_growth = trade_data.get('revenue_growth', 0)
            innovation_score = trade_data.get('innovation_score', 0)
            
            return np.mean([earnings_impact, revenue_growth, innovation_score])
        except Exception:
            return 0.0

    def _analyze_beta_factors(self, trade_data: Dict, market_data: pd.DataFrame) -> float:
        """Analyze market-related factors"""
        try:
            # Calculate correlation with index
            returns = pd.Series(trade_data.get('price_history', []))
            market_returns = market_data['Close'].pct_change()
            correlation = returns.corr(market_returns)
            
            return abs(correlation)
        except Exception:
            return 0.0

    def _analyze_sentiment_factors(self, trade_data: Dict) -> float:
        """Analyze sentiment impact"""
        try:
            sentiment_score = trade_data.get('sentiment_score', 0)
            news_impact = trade_data.get('news_impact', 0)
            
            return np.mean([sentiment_score, news_impact])
        except Exception:
            return 0.0

    def _analyze_geopolitical_factors(self, trade_data: Dict) -> float:
        """Analyze geopolitical impact"""
        try:
            geo_risk = trade_data.get('geopolitical_risk', 0)
            macro_impact = trade_data.get('macro_impact', 0)
            
            return np.mean([geo_risk, macro_impact])
        except Exception:
            return 0.0

    def _generate_recommendations(self, primary_factor: str, impacts: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if primary_factor == 'alpha':
            recommendations.append("Review company-specific analysis criteria")
            recommendations.append("Adjust earnings importance in trade selection")
        elif primary_factor == 'beta':
            recommendations.append("Consider reducing position size in high market correlation")
            recommendations.append("Implement sector-specific beta thresholds")
        elif primary_factor == 'sentiment':
            recommendations.append("Adjust sentiment analysis weights")
            recommendations.append("Implement sentiment confirmation delays")
        else:  # geopolitical
            recommendations.append("Enhance geopolitical risk monitoring")
            recommendations.append("Implement country-specific risk thresholds")
            
        return recommendations
