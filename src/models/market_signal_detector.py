
import numpy as np
from typing import Dict, List
from filterpy.kalman import KalmanFilter
from hmmlearn import hmm
import logging

logger = logging.getLogger(__name__)

class MarketSignalDetector:
    """Detects market signals using HMM and Kalman filters"""
    
    def __init__(self, n_states: int = 3):
        self.hmm = hmm.GaussianHMM(n_components=n_states, random_state=42)
        self.kalman = self._init_kalman()
        
    def _init_kalman(self) -> KalmanFilter:
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([[0.], [0.]])  # State initialization
        kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
        kf.H = np.array([[1., 0.]])  # Measurement function
        kf.P *= 1000.  # Covariance matrix
        kf.R = 5  # Measurement noise
        kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])  # Process noise
        return kf
        
    def detect_regime(self, prices: np.ndarray) -> Dict:
        """Detect market regime using HMM"""
        returns = np.diff(np.log(prices)).reshape(-1, 1)
        self.hmm.fit(returns)
        states = self.hmm.predict(returns)
        
        return {
            'current_regime': int(states[-1]),
            'regime_probs': self.hmm.predict_proba(returns)[-1],
            'transition_matrix': self.hmm.transmat_
        }
        
    def filter_price(self, price: float) -> Dict:
        """Apply Kalman filter to price series"""
        self.kalman.predict()
        self.kalman.update(price)
        
        return {
            'filtered_price': float(self.kalman.x[0]),
            'velocity': float(self.kalman.x[1]),
            'uncertainty': float(self.kalman.P[0, 0])
        }
