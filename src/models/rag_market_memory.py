
import pinecone
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RAGMarketMemory:
    """RAG-based market memory system using Pinecone"""
    
    def __init__(self, pinecone_api_key: str, environment: str):
        pinecone.init(api_key=pinecone_api_key, environment=environment)
        self.index_name = "market-regimes"
        self.index = pinecone.Index(self.index_name)
        
    def store_market_state(self, 
                          state: Dict,
                          regime_data: Dict,
                          performance: Dict) -> None:
        """Store market state in Pinecone"""
        vector = np.array([
            state['returns'],
            state['volatility'],
            state['volume'],
            regime_data['probability']
        ])
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'regime': regime_data['regime'],
            'performance': str(performance),
            'confidence': float(regime_data['probability'])
        }
        
        self.index.upsert(
            vectors=[(str(datetime.now().timestamp()), vector.tolist(), metadata)]
        )
        
    def query_similar_states(self, current_state: Dict, k: int = 5) -> List[Dict]:
        """Query similar market states"""
        vector = np.array([
            current_state['returns'],
            current_state['volatility'],
            current_state['volume'],
            current_state.get('confidence', 0.5)
        ])
        
        results = self.index.query(
            vector=vector.tolist(),
            top_k=k,
            include_metadata=True
        )
        
        return [match['metadata'] for match in results['matches']]
