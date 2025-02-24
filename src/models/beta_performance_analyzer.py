
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BetaPerformanceAnalyzer:
    def __init__(self):
        self.significance_threshold = 0.02
        self.min_data_points = 30
        
    def analyze_beta_performance(self,
                               stock_returns: np.ndarray,
                               market_returns: np.ndarray) -> Dict:
        """Analyze if stock moved differently than beta would predict"""
        try:
            # Add constant for regression
            X = sm.add_constant(market_returns)
            
            # Calculate beta through regression
            model = sm.OLS(stock_returns, X).fit()
            beta = model.params[1]
            r_squared = model.rsquared
            
            # Calculate expected returns and residuals
            expected_returns = beta * market_returns
            residuals = stock_returns - expected_returns
            
            # Analyze residuals
            mean_residual = np.abs(residuals).mean()
            is_unexpected = mean_residual > self.significance_threshold
            
            return {
                'beta': beta,
                'r_squared': r_squared,
                'mean_residual': mean_residual,
                'is_unexpected_move': is_unexpected,
                'confidence_score': 1 - (mean_residual / self.significance_threshold),
                'residuals': residuals.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in beta performance analysis: {str(e)}")
            return {}
