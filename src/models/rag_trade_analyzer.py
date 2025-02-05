import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class RAGTradeAnalyzer:
    """RAG-powered trade analysis and insight generation"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.trade_data = []
        self.pattern_embeddings = None

    def add_trade_data(self, trades: List[Dict]):
        """Add historical trade data to the index"""
        try:
            # Convert trade data to text representations
            trade_texts = [self._trade_to_text(trade) for trade in trades]

            # Generate embeddings
            self.pattern_embeddings = self.vectorizer.fit_transform(trade_texts)
            self.trade_data.extend(trades)

            logger.info(f"Successfully added {len(trades)} trades to the database")
        except Exception as e:
            logger.error(f"Error adding trade data: {str(e)}")
            raise

    def _trade_to_text(self, trade: Dict) -> str:
        """Convert trade data to text representation for embedding"""
        return (
            f"Trade on {trade.get('timestamp', '')} for {trade.get('symbol', '')} "
            f"Direction: {trade.get('direction', '')} "
            f"Entry: {trade.get('entry_price', '')} "
            f"Exit: {trade.get('exit_price', '')} "
            f"Volume: {trade.get('volume', '')} "
            f"Market Sentiment: {trade.get('market_sentiment', '')} "
            f"Technical Signals: {trade.get('technical_signals', '')}"
        )

    def get_similar_trades(self, current_conditions: Dict, k: int = 5) -> List[Dict]:
        """Find similar historical trades based on current market conditions"""
        try:
            if not self.trade_data:
                logger.warning("No historical trade data available")
                return []

            # Convert current conditions to text
            condition_text = (
                f"Market conditions: {current_conditions.get('market_regime', '')} "
                f"Sentiment: {current_conditions.get('sentiment', '')} "
                f"Volume Profile: {current_conditions.get('volume_profile', '')} "
                f"Technical Signals: {current_conditions.get('technical_signals', '')}"
            )

            # Generate embedding for query
            query_embedding = self.vectorizer.transform([condition_text])

            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_embedding, self.pattern_embeddings)[0]

            # Get top k similar trades
            top_indices = np.argsort(-similarity_scores)[:k]

            # Return similar trades with similarity scores
            similar_trades = []
            for idx in top_indices:
                trade = self.trade_data[idx].copy()
                trade['similarity_score'] = float(similarity_scores[idx])
                similar_trades.append(trade)

            return similar_trades

        except Exception as e:
            logger.error(f"Error finding similar trades: {str(e)}")
            return []

    def generate_trade_insights(self, 
                            current_conditions: Dict,
                            similar_trades: List[Dict]) -> Dict:
        """Generate AI-enhanced trade insights based on similar historical trades"""
        try:
            if not similar_trades:
                return self._generate_default_insights()

            # Analyze outcomes of similar trades
            successful_trades = [
                trade for trade in similar_trades 
                if trade.get('profit', 0) > 0
            ]
            success_rate = len(successful_trades) / len(similar_trades) if similar_trades else 0

            # Calculate average profit and risk metrics
            avg_profit = np.mean([
                trade.get('profit', 0) for trade in successful_trades
            ]) if successful_trades else 0

            avg_risk_ratio = np.mean([
                trade.get('risk_reward_ratio', 1) for trade in similar_trades
            ]) if similar_trades else 0

            # Generate insights
            insights = {
                'timestamp': datetime.now().isoformat(),
                'market_context': {
                    'regime': current_conditions.get('market_regime', 'UNKNOWN'),
                    'sentiment': current_conditions.get('sentiment', 'NEUTRAL'),
                    'volume_profile': current_conditions.get('volume_profile', 'NORMAL')
                },
                'historical_analysis': {
                    'similar_trades_count': len(similar_trades),
                    'success_rate': success_rate,
                    'avg_profit': avg_profit,
                    'avg_risk_ratio': avg_risk_ratio
                },
                'recommendations': self._generate_recommendations(
                    success_rate, avg_profit, avg_risk_ratio
                ),
                'confidence_score': self._calculate_confidence_score(
                    similar_trades, current_conditions
                )
            }

            return insights

        except Exception as e:
            logger.error(f"Error generating trade insights: {str(e)}")
            return self._generate_default_insights()

    def _generate_recommendations(self, 
                               success_rate: float, 
                               avg_profit: float,
                               risk_ratio: float) -> List[str]:
        """Generate specific trading recommendations"""
        recommendations = []

        if success_rate > 0.7:
            recommendations.append("Historical patterns suggest favorable conditions")
        elif success_rate < 0.3:
            recommendations.append("Exercise caution - historical success rate is low")

        if avg_profit > 0:
            recommendations.append(
                f"Similar trades averaged {avg_profit:.2%} profit"
            )

        if risk_ratio > 2:
            recommendations.append("Consider tighter stop-loss for risk management")
        elif risk_ratio < 1:
            recommendations.append("Potential for improved risk-reward ratio")

        return recommendations

    def _calculate_confidence_score(self, 
                                similar_trades: List[Dict],
                                current_conditions: Dict) -> float:
        """Calculate confidence score based on historical similarity"""
        if not similar_trades:
            return 0.0

        # Weight factors
        similarity_weight = 0.4
        success_weight = 0.3
        volume_weight = 0.3

        # Calculate similarity confidence
        avg_similarity = np.mean([
            trade.get('similarity_score', 0) for trade in similar_trades
        ])
        similarity_confidence = avg_similarity

        # Calculate success confidence
        success_rate = len([
            trade for trade in similar_trades if trade.get('profit', 0) > 0
        ]) / len(similar_trades) if similar_trades else 0
        success_confidence = success_rate

        # Calculate volume confidence
        volume_profile = current_conditions.get('volume_profile', 'NORMAL')
        volume_confidence = {
            'HIGH': 0.8,
            'NORMAL': 0.6,
            'LOW': 0.4
        }.get(volume_profile, 0.5)

        # Combine confidence scores
        total_confidence = (
            similarity_weight * similarity_confidence +
            success_weight * success_confidence +
            volume_weight * volume_confidence
        )

        return min(1.0, max(0.0, total_confidence))

    def _generate_default_insights(self) -> Dict:
        """Generate default insights when no similar trades are found"""
        return {
            'timestamp': datetime.now().isoformat(),
            'market_context': {
                'regime': 'UNKNOWN',
                'sentiment': 'NEUTRAL',
                'volume_profile': 'NORMAL'
            },
            'historical_analysis': {
                'similar_trades_count': 0,
                'success_rate': 0.0,
                'avg_profit': 0.0,
                'avg_risk_ratio': 0.0
            },
            'recommendations': [
                "Insufficient historical data for comparison",
                "Consider gathering more market data",
                "Monitor market conditions closely"
            ],
            'confidence_score': 0.0
        }