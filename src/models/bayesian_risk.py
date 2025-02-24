
import numpy as np
from scipy import stats
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class BayesianRiskAnalyzer:
    """Bayesian approach to risk assessment and position sizing"""
    
    def __init__(self):
        self.prior_alpha = 2.0
        self.prior_beta = 2.0
        self.min_confidence = 0.75
        
    def calculate_position_risk(self, 
                              signals: Dict,
                              historical_performance: List[float]) -> Dict:
        """Calculate position risk using Bayesian inference"""
        try:
            # Update beliefs based on historical performance
            successes = sum(1 for x in historical_performance if x > 0)
            failures = len(historical_performance) - successes
            
            posterior_alpha = self.prior_alpha + successes
            posterior_beta = self.prior_beta + failures
            
            # Calculate probability of success
            success_prob = stats.beta.mean(posterior_alpha, posterior_beta)
            
            # Calculate Kelly criterion for position sizing
            edge = signals.get('edge', 0.5)
            odds = signals.get('odds', 1.0)
            kelly_fraction = self._calculate_kelly(success_prob, edge, odds)
            
            return {
                'success_probability': success_prob,
                'position_size': min(kelly_fraction, 0.05),  # Cap at 5%
                'confidence_interval': stats.beta.interval(0.95, posterior_alpha, posterior_beta),
                'risk_score': 1 - success_prob
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian risk analysis: {str(e)}")
            return {}
            
    def _calculate_kelly(self, prob: float, edge: float, odds: float) -> float:
        """Calculate Kelly criterion for optimal position sizing"""
        q = 1 - prob
        return (prob * (1 + edge) - q) / odds if prob > self.min_confidence else 0
