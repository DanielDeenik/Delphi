
from typing import Dict, List, Tuple
import numpy as np
import pinecone
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RAGTradeService:
    def __init__(self):
        self.index_name = "trade-memory"
        
    def encode_trade_features(self, market_data: Dict) -> np.ndarray:
        """Convert market data into vector representation"""
        features = np.array([
            market_data.get('price_momentum', 0),
            market_data.get('volume_profile', 0),
            market_data.get('volatility', 0),
            market_data.get('sentiment_score', 0)
        ])
        return features / np.linalg.norm(features)
        
    def store_successful_trade(self, 
                             trade_data: Dict,
                             performance: float,
                             metadata: Dict) -> None:
        """Store successful trade setup in Pinecone"""
        try:
            trade_vector = self.encode_trade_features(trade_data)
            
            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'performance': float(performance),
                'success': 1 if performance > 0 else 0
            })
            
            self.index.upsert([
                (str(datetime.now().timestamp()), 
                 trade_vector.tolist(),
                 metadata)
            ])
            
        except Exception as e:
            logger.error(f"Error storing trade: {str(e)}")
            
    def find_similar_setups(self, 
                           current_setup: Dict,
                           k: int = 5) -> List[Dict]:
        """Find similar historical trade setups"""
        try:
            query_vector = self.encode_trade_features(current_setup)
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=k,
                include_metadata=True
            )
            return results.matches
            
        except Exception as e:
            logger.error(f"Error querying similar trades: {str(e)}")
            return []
