
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class MedallionAnalyzer:
    """Implements Renaissance-style statistical arbitrage models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.hmm_model = hmm.GaussianHMM(n_components=5, random_state=42)
        self.lookback_window = 252  # 1 year trading days
        self.min_data_points = 50
        
    def detect_stat_arb_opportunities(self, price_data: pd.DataFrame) -> Dict:
        """Detect statistical arbitrage opportunities using HMM"""
        try:
            # Prepare features
            returns = np.log(price_data['Close']).diff().dropna()
            volatility = returns.rolling(20).std()
            volume_ma = price_data['Volume'].rolling(20).mean()
            
            # Scale features
            X = np.column_stack([
                self.scaler.fit_transform(returns.values.reshape(-1, 1)),
                self.scaler.fit_transform(volatility.values.reshape(-1, 1)),
                self.scaler.fit_transform(volume_ma.values.reshape(-1, 1))
            ])
            
            # Fit HMM and predict states
            self.hmm_model.fit(X)
            hidden_states = self.hmm_model.predict(X)
            
            return {
                'current_regime': hidden_states[-1],
                'regime_probability': np.max(self.hmm_model.predict_proba(X)[-1]),
                'transition_matrix': self.hmm_model.transmat_,
                'confidence_score': self._calculate_confidence(X, hidden_states)
            }
            
        except Exception as e:
            logger.error(f"Error in statistical arbitrage detection: {str(e)}")
            return {}
            
    def _calculate_confidence(self, X: np.ndarray, states: np.ndarray) -> float:
        """Calculate confidence score based on state stability"""
        recent_states = states[-self.min_data_points:]
        state_stability = len(set(recent_states)) / len(recent_states)
        signal_strength = np.mean(np.abs(X[-self.min_data_points:]))
        return (1 - state_stability) * signal_strength
