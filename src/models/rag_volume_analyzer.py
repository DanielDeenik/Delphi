import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
import datetime

class RAGVolumeAnalyzer:
    """Enhanced RAG-powered volume analysis system with real-time data integration"""

    def __init__(self):
        self.volume_patterns_db = self._initialize_volume_patterns()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pattern_embeddings = None
        self._initialize_embeddings()

    def _initialize_volume_patterns(self) -> List[Dict]:
        """Initialize comprehensive database of volume patterns and their descriptions"""
        return [
            {
                "pattern": "high_volume_breakout",
                "description": "Price breaks above resistance on 50%+ higher than average volume",
                "signal": "Potential start of new uptrend, especially if accompanied by positive news or earnings",
                "context": "Often seen during earnings surprises or major announcements"
            },
            {
                "pattern": "institutional_accumulation",
                "description": "Steady volume increase with rising prices over multiple sessions",
                "signal": "Large players potentially building positions, bullish if sustained",
                "context": "Common during sector rotations or new fund allocations"
            },
            {
                "pattern": "volume_climax",
                "description": "Extremely high volume (3x+ average) with sharp price movement",
                "signal": "Possible exhaustion of current trend, watch for reversal",
                "context": "Often marks short-term tops or bottoms"
            },
            {
                "pattern": "low_volume_pullback",
                "description": "Price retracement on lower than average volume",
                "signal": "Healthy consolidation within uptrend if volume stays low",
                "context": "Common during bull market corrections"
            },
            {
                "pattern": "distribution_day",
                "description": "Higher volume on down days compared to up days",
                "signal": "Potential institutional selling, concerning if pattern persists",
                "context": "Watch for multiple distribution days within a short period"
            }
        ]

    def _initialize_embeddings(self):
        """Initialize FAISS index for pattern matching"""
        pattern_texts = [
            f"{p['pattern']} {p['description']} {p['signal']} {p['context']}"
            for p in self.volume_patterns_db
        ]
        embeddings = self.model.encode(pattern_texts)

        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.pattern_embeddings = embeddings

    def analyze_volume_pattern(self, volume_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns with enhanced context"""
        insights = []
        lookback_period = 20

        # Calculate key metrics
        avg_volume = volume_data['Volume'].rolling(window=lookback_period).mean()
        volume_ratio = volume_data['Volume'] / avg_volume
        price_change = volume_data['Close'].pct_change()

        # Advanced metrics
        volume_trend = volume_data['Volume'].pct_change().rolling(5).mean()
        price_volume_correlation = volume_data['Close'].pct_change().rolling(5).corr(
            volume_data['Volume'].pct_change()
        )

        # Check for patterns
        if len(volume_data) > lookback_period:
            latest_ratio = volume_ratio.iloc[-1]
            latest_price_change = price_change.iloc[-1]
            latest_volume_trend = volume_trend.iloc[-1]
            latest_correlation = price_volume_correlation.iloc[-1]

            # Enhanced pattern detection
            patterns_to_check = [
                (
                    'high_volume_breakout',
                    latest_ratio > 1.5 and latest_price_change > 0,
                    {"volume_ratio": latest_ratio, "price_change": latest_price_change}
                ),
                (
                    'institutional_accumulation',
                    latest_volume_trend > 0.1 and latest_correlation > 0.6,
                    {"volume_trend": latest_volume_trend, "price_vol_correlation": latest_correlation}
                ),
                (
                    'volume_climax',
                    latest_ratio > 3,
                    {"volume_ratio": latest_ratio, "price_change": latest_price_change}
                ),
                (
                    'distribution_day',
                    latest_ratio > 1.2 and latest_price_change < 0 and latest_correlation < -0.5,
                    {"volume_ratio": latest_ratio, "price_vol_correlation": latest_correlation}
                )
            ]

            for pattern_name, condition, metrics in patterns_to_check:
                if condition:
                    pattern_info = next(
                        p for p in self.volume_patterns_db 
                        if p['pattern'] == pattern_name
                    )
                    insights.append({
                        'pattern': pattern_name,
                        'description': pattern_info['description'],
                        'signal': pattern_info['signal'],
                        'context': pattern_info['context'],
                        'metrics': {
                            **metrics,
                            'lookback_period': lookback_period
                        }
                    })

        return {
            'timestamp': volume_data.index[-1] if len(volume_data) > 0 else None,
            'insights': insights,
            'metrics': {
                'volume_ratio': float(latest_ratio) if 'latest_ratio' in locals() else None,
                'price_change': float(latest_price_change) if 'latest_price_change' in locals() else None,
                'volume_trend': float(latest_volume_trend) if 'latest_volume_trend' in locals() else None,
                'price_volume_correlation': float(latest_correlation) if 'latest_correlation' in locals() else None
            }
        }

    def find_similar_patterns(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find similar volume patterns using FAISS"""
        # Encode the query
        query_embedding = self.model.encode([query])

        # Search in FAISS index
        D, I = self.index.search(query_embedding.astype('float32'), top_k)

        # Get results
        results = []
        for idx, score in zip(I[0], D[0]):
            pattern = self.volume_patterns_db[idx]
            results.append({
                **pattern,
                'similarity_score': float(1 / (1 + score))  # Convert distance to similarity
            })

        return results

    def get_contextual_insight(self, pattern: str, metrics: Dict) -> str:
        """Generate contextual insight based on pattern and metrics"""
        base_templates = {
            'high_volume_breakout': (
                "Volume surged {volume_ratio:.1f}x above average with a "
                "{price_change:+.1%} price move, suggesting strong buying interest"
            ),
            'institutional_accumulation': (
                "Steady accumulation detected with {volume_trend:+.1%} volume trend "
                "and {price_vol_correlation:.2f} price-volume correlation"
            ),
            'volume_climax': (
                "Extreme volume spike of {volume_ratio:.1f}x normal levels, "
                "potentially signaling a trend exhaustion"
            ),
            'distribution_day': (
                "Distribution pattern with {volume_ratio:.1f}x volume on "
                "declining prices, suggesting institutional selling"
            )
        }

        template = base_templates.get(pattern, "Volume pattern detected")
        try:
            return template.format(**metrics)
        except KeyError:
            return template

# Export the class
__all__ = ['RAGVolumeAnalyzer']