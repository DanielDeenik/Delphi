
import numpy as np
import pandas as pd
from typing import Dict, List
from hmmlearn import hmm
import pinecone
from src.models.rag_market_memory import RAGMarketMemory
from src.models.market_signal_detector import MarketSignalDetector

class IntegratedMarketPredictor:
    """Integrates HMM predictions with RAG memory for market analysis"""
    
    def __init__(self, pinecone_api_key: str, environment: str):
        self.market_memory = RAGMarketMemory(pinecone_api_key, environment)
        self.signal_detector = MarketSignalDetector(n_states=3)
        self.hmm = hmm.GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        )
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM analysis"""
        returns = np.diff(np.log(data['close']))
        volume = np.log(data['volume'][1:])
        features = np.column_stack([returns, volume])
        return (features - features.mean()) / features.std()
        
    def predict_and_store(self, market_data: pd.DataFrame) -> Dict:
        """Predict market state and store in RAG memory"""
        # Prepare features
        features = self.prepare_features(market_data)
        
        # Train HMM and predict current regime
        self.hmm.fit(features)
        current_regime = self.hmm.predict(features)[-1]
        regime_probs = self.hmm.predict_proba(features)[-1]
        
        # Get filtered price signals
        latest_price = market_data['close'].iloc[-1]
        filtered_signals = self.signal_detector.filter_price(latest_price)
        
        # Create market state
        state = {
            'returns': features[-1, 0],
            'volume': features[-1, 1],
            'regime': current_regime,
            'regime_probability': float(regime_probs[current_regime]),
            'filtered_price': filtered_signals['filtered_price'],
            'price_velocity': filtered_signals['velocity']
        }
        
        # Store in RAG memory
        self.market_memory.store_market_state(
            state=state,
            regime_data={'regime': current_regime, 'probability': regime_probs[current_regime]},
            performance={'price': latest_price, 'momentum': filtered_signals['velocity']}
        )
        
        return {
            'current_regime': current_regime,
            'regime_probability': float(regime_probs[current_regime]),
            'signals': filtered_signals,
            'similar_states': self.market_memory.query_similar_states(state, k=5)
        }
