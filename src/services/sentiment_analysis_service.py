import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import os
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """Service for analyzing market sentiment from social media data"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_mock = True  # Flag to indicate if we're using mock data

    def get_sentiment_analysis(self, symbol: str) -> Dict:
        """Get sentiment analysis for a given symbol"""
        try:
            # Check cache first
            if symbol in self.cache:
                last_update, data = self.cache[symbol]
                if datetime.now() - last_update < timedelta(seconds=self.cache_timeout):
                    return data

            # Get sentiment data (mock for now)
            sentiment_data = self._generate_mock_sentiment(symbol)

            # Cache the results
            self.cache[symbol] = (datetime.now(), sentiment_data)

            return sentiment_data

        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {str(e)}")
            return self._get_default_sentiment()

    def _generate_mock_sentiment(self, symbol: str) -> Dict:
        """Generate realistic mock sentiment data"""
        try:
            # Base sentiment affected by time of day and random market factors
            hour = datetime.now().hour
            market_hours = 9 <= hour <= 16
            base_sentiment = random.uniform(0.4, 0.8) if market_hours else random.uniform(0.3, 0.6)

            # Volume change based on market hours
            volume_change = random.uniform(1.5, 3.0) if market_hours else random.uniform(0.5, 1.5)

            # Generate trending topics with realistic patterns
            trends = self._generate_realistic_trends(symbol)

            # Calculate source-specific sentiment
            twitter_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.1, 0.1)))
            reddit_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.15, 0.15)))
            stocktwits_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.05, 0.05)))

            sentiment_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'sentiment_metrics': {
                    'overall_score': base_sentiment,
                    'sentiment_change_24h': random.uniform(-0.15, 0.15),
                    'confidence': random.uniform(0.7, 0.9),
                    'volume_change': volume_change
                },
                'trending_topics': trends,
                'source_breakdown': {
                    'twitter': twitter_sentiment,
                    'reddit': reddit_sentiment,
                    'stocktwits': stocktwits_sentiment
                },
                'recent_mentions': self._generate_realistic_mentions(symbol, base_sentiment),
                'sentiment_signals': self._generate_sentiment_signals(base_sentiment)
            }

            logger.info(f"Generated mock sentiment data for {symbol}: {sentiment_data['sentiment_metrics']}")
            return sentiment_data

        except Exception as e:
            logger.error(f"Error generating mock sentiment: {str(e)}")
            return self._get_default_sentiment()

    def _generate_realistic_trends(self, symbol: str) -> List[Dict]:
        """Generate realistic trending topics based on market context"""
        base_topics = [
            ('earnings', 0.8),
            ('market analysis', 0.6),
            ('technical analysis', 0.7),
            ('price target', 0.75),
            ('trading volume', 0.65),
            ('market sentiment', 0.7),
            ('stock analysis', 0.72),
            ('market outlook', 0.68),
            ('trading strategy', 0.67),
            ('market update', 0.71)
        ]

        # Select 4-6 topics with realistic strength values
        selected_count = random.randint(4, 6)
        selected_topics = random.sample(base_topics, selected_count)

        return [
            {
                'topic': topic,
                'strength': strength * random.uniform(0.8, 1.2),  # Add some variation
                'sentiment': random.uniform(0.4, 0.8)  # Slightly optimistic bias
            }
            for topic, strength in selected_topics
        ]

    def _generate_realistic_mentions(self, symbol: str, base_sentiment: float) -> List[Dict]:
        """Generate realistic-looking social media mentions"""
        templates = [
            "#{symbol} showing strong momentum with increasing volume 📈",
            "Technical analysis suggests potential breakout for {symbol} 🚀",
            "Interesting price action on {symbol} today, watching closely 👀",
            "Volume analysis indicates accumulation in {symbol} 📊",
            "Market sentiment turning positive for {symbol} 💹",
            "Keep an eye on {symbol} for potential entry points 🎯",
            "Volatility increasing in {symbol}, exercise caution ⚠️",
            "Strong support level holding for {symbol} 💪",
            "Analyzing {symbol} price patterns for trading opportunities 📝",
            "Market makers active in {symbol} today 🏦"
        ]

        mentions = []
        for _ in range(5):
            template = random.choice(templates)
            sentiment_variation = random.uniform(-0.2, 0.2)
            mentions.append({
                'text': template.format(symbol=symbol),
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
                'sentiment': max(0, min(1, base_sentiment + sentiment_variation)),
                'engagement': int(random.gauss(500, 200))  # Normal distribution for engagement
            })

        return mentions

    def _generate_sentiment_signals(self, sentiment_score: float) -> Dict:
        """Generate actionable trading signals based on sentiment analysis"""
        signal_strength = abs(sentiment_score - 0.5) * 2

        if sentiment_score > 0.7:
            signal = 'BULLISH'
            action = "Strong positive sentiment suggests potential upside"
        elif sentiment_score > 0.55:
            signal = 'BULLISH'
            action = "Mild positive sentiment detected"
        elif sentiment_score < 0.3:
            signal = 'BEARISH'
            action = "Significant negative sentiment detected"
        elif sentiment_score < 0.45:
            signal = 'BEARISH'
            action = "Mild negative sentiment suggests caution"
        else:
            signal = 'NEUTRAL'
            action = "Mixed sentiment signals, monitor for clarity"

        return {
            'signal': signal,
            'strength': signal_strength,
            'confidence': max(0.5, min(0.9, 1 - abs(0.5 - sentiment_score))),
            'suggested_action': action,
            'risk_level': 'HIGH' if signal_strength > 0.7 else 'MEDIUM' if signal_strength > 0.4 else 'LOW'
        }

    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment data in case of errors"""
        return {
            'sentiment_metrics': {
                'overall_score': 0.5,
                'sentiment_change_24h': 0.0,
                'confidence': 0.5,
                'volume_change': 0.0
            },
            'trending_topics': [],
            'source_breakdown': {
                'twitter': 0.5,
                'reddit': 0.5,
                'stocktwits': 0.5
            },
            'recent_mentions': [],
            'sentiment_signals': {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.5,
                'suggested_action': "Insufficient data for analysis",
                'risk_level': "LOW"
            }
        }