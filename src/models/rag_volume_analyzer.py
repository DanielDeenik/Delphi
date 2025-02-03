import numpy as np
import pandas as pd
from typing import List, Dict

class RAGVolumeAnalyzer:
    """RAG-powered volume analysis system"""

    def __init__(self):
        self.volume_patterns_db = self._initialize_volume_patterns()

    def _initialize_volume_patterns(self) -> List[Dict]:
        """Initialize database of volume patterns and their descriptions"""
        return [
            {
                "pattern": "high_volume_breakout",
                "description": "Price breaks above resistance on 50%+ higher than average volume",
                "signal": "Potential start of new uptrend, especially if accompanied by positive news or earnings"
            },
            {
                "pattern": "volume_climax",
                "description": "Extremely high volume (3x+ average) with sharp price movement",
                "signal": "Possible exhaustion of current trend, watch for reversal"
            },
            {
                "pattern": "low_volume_pullback",
                "description": "Price retracement on lower than average volume",
                "signal": "Healthy consolidation within uptrend if volume stays low"
            }
        ]

    def analyze_volume_pattern(self, volume_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        insights = []
        lookback_period = 20

        # Calculate key metrics
        avg_volume = volume_data['Volume'].rolling(window=lookback_period).mean()
        volume_ratio = volume_data['Volume'] / avg_volume
        price_change = volume_data['Close'].pct_change()

        # Check for patterns
        if len(volume_data) > 0:
            latest_ratio = volume_ratio.iloc[-1]
            latest_price_change = price_change.iloc[-1]

            # High volume breakout
            if latest_ratio > 1.5 and latest_price_change > 0:
                insights.append({
                    'pattern': 'high_volume_breakout',
                    'description': self.volume_patterns_db[0]['description'],
                    'signal': self.volume_patterns_db[0]['signal'],
                    'metrics': {
                        'volume_ratio': float(latest_ratio),
                        'price_change': float(latest_price_change),
                        'lookback_period': lookback_period
                    }
                })

            # Volume climax
            if latest_ratio > 3:
                insights.append({
                    'pattern': 'volume_climax',
                    'description': self.volume_patterns_db[1]['description'],
                    'signal': self.volume_patterns_db[1]['signal'],
                    'metrics': {
                        'volume_ratio': float(latest_ratio),
                        'price_change': float(latest_price_change),
                        'lookback_period': lookback_period
                    }
                })

        return {
            'timestamp': volume_data.index[-1] if len(volume_data) > 0 else None,
            'insights': insights
        }

    def find_similar_patterns(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar volume patterns based on text query"""
        # Simple keyword matching for now
        results = []
        query = query.lower()

        for pattern in self.volume_patterns_db:
            # Calculate a simple similarity score based on word overlap
            pattern_text = f"{pattern['pattern']} {pattern['description']} {pattern['signal']}".lower()
            similarity = len(set(query.split()) & set(pattern_text.split())) / len(set(query.split()))

            if similarity > 0:
                results.append({
                    **pattern,
                    'similarity_score': float(similarity)
                })

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

# Export the class
__all__ = ['RAGVolumeAnalyzer']