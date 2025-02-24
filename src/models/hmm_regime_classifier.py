import numpy as np
from hmmlearn import hmm
from typing import Dict, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MarketRegimeClassifier:
    """HMM-based market regime classifier"""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        )
        self.regime_labels = ['Bearish', 'Sideways', 'Bullish']

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM"""
        returns = np.diff(np.log(data['close']))
        volatility = data['high'].div(data['low']).apply(np.log)
        volume = np.log(data['volume'])

        # Normalize features
        features = np.column_stack([
            returns,
            volatility[1:],
            volume[1:]
        ])
        return (features - features.mean()) / features.std()

    def fit(self, data: pd.DataFrame) -> None:
        """Train HMM model"""
        try:
            features = self.prepare_features(data)
            self.hmm.fit(features)
            logger.info("Successfully trained HMM model")
        except Exception as e:
            logger.error(f"Error training HMM: {str(e)}")
            raise

    def predict_regime(self, data: pd.DataFrame) -> Dict:
        """Predict market regime"""
        features = self.prepare_features(data)
        current_regime = self.hmm.predict(features)[-1]
        regime_probs = self.hmm.predict_proba(features)[-1]

        return {
            'regime': self.regime_labels[current_regime],
            'probability': float(regime_probs[current_regime]),
            'transition_matrix': self.hmm.transmat_.tolist(),
            'regime_probabilities': regime_probs.tolist()
        }