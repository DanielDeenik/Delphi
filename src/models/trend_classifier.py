from src.config.trend_analysis_config import TREND_THRESHOLDS, TREND_WEIGHTS, TREND_STAGES
from typing import Dict

class TrendClassifier:
    """Classifies trends using Chris Camillo's methodology"""

    def _calculate_trend_score(self, social: Dict, demand: Dict) -> float:
        """Calculate overall trend score"""
        social_score = social.get('sentiment_score', 0) * social.get('engagement_score', 0)
        demand_score = demand.get('demand_score', 0) * demand.get('trend_momentum', 0)

        return (social_score * TREND_WEIGHTS['SOCIAL_WEIGHT'] + 
                demand_score * TREND_WEIGHTS['DEMAND_WEIGHT']) * 100

    def _determine_stage(self, score: float) -> str:
        """Determine trend stage based on score thresholds"""
        if score > TREND_THRESHOLDS['VIRAL_THRESHOLD']:
            return TREND_STAGES['VIRAL']
        elif score > TREND_THRESHOLDS['EARLY_THRESHOLD']:
            return TREND_STAGES['EARLY']
        elif score < TREND_THRESHOLDS['FADING_THRESHOLD']:
            return TREND_STAGES['FADING']
        return TREND_STAGES['UNDEFINED']

    def classify(self, social: Dict, demand: Dict) -> str:
        """Classifies the trend based on social and demand data."""
        score = self._calculate_trend_score(social, demand)
        return self._determine_stage(score)