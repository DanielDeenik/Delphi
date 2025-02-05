import logging
from typing import Dict, List
from datetime import datetime, timedelta
import pandas as pd
import os
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import trafilatura
import requests

logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """Service for analyzing market sentiment using STEPPS framework"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.is_mock = True  # Flag to indicate if we're using mock data

    def get_sentiment_analysis(self, symbol: str) -> Dict:
        """Get sentiment analysis for a given symbol using STEPPS framework"""
        try:
            # Check cache first
            if symbol in self.cache:
                last_update, data = self.cache[symbol]
                if datetime.now() - last_update < timedelta(seconds=self.cache_timeout):
                    return data

            # Get sentiment data (mock for now)
            sentiment_data = self._generate_stepps_sentiment(symbol)

            # Cache the results
            self.cache[symbol] = (datetime.now(), sentiment_data)

            return sentiment_data

        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {str(e)}")
            return self._get_default_sentiment()

    def _generate_stepps_sentiment(self, symbol: str) -> Dict:
        """Generate STEPPS framework based sentiment analysis"""
        try:
            # Base sentiment affected by time of day and random market factors
            hour = datetime.now().hour
            market_hours = 9 <= hour <= 16
            base_sentiment = random.uniform(0.4, 0.8) if market_hours else random.uniform(0.3, 0.6)

            # STEPPS components
            stepps_metrics = {
                'social_currency': self._calculate_social_currency(symbol),
                'triggers': self._generate_market_triggers(symbol),
                'emotion': self._analyze_emotional_content(symbol),
                'public': self._analyze_public_visibility(symbol),
                'practical_value': self._analyze_practical_value(symbol),
                'stories': self._analyze_narrative_impact(symbol)
            }

            # Calculate source-specific sentiment
            twitter_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.1, 0.1)))
            reddit_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.15, 0.15)))
            stocktwits_sentiment = max(0, min(1, base_sentiment + random.uniform(-0.05, 0.05)))

            # Viral coefficient calculation
            viral_score = (stepps_metrics['social_currency'] * 0.3 +
                         stepps_metrics['emotion'] * 0.2 +
                         stepps_metrics['public'] * 0.2 +
                         stepps_metrics['practical_value'] * 0.15 +
                         stepps_metrics['stories'] * 0.15)

            sentiment_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'stepps_analysis': stepps_metrics,
                'sentiment_metrics': {
                    'overall_score': base_sentiment,
                    'viral_coefficient': viral_score,
                    'sentiment_change_24h': random.uniform(-0.15, 0.15),
                    'confidence': random.uniform(0.7, 0.9)
                },
                'source_breakdown': {
                    'twitter': twitter_sentiment,
                    'reddit': reddit_sentiment,
                    'stocktwits': stocktwits_sentiment
                },
                'viral_signals': self._generate_viral_signals(viral_score)
            }

            logger.info(f"Generated STEPPS sentiment data for {symbol}")
            return sentiment_data

        except Exception as e:
            logger.error(f"Error generating STEPPS sentiment: {str(e)}")
            return self._get_default_sentiment()

    def _calculate_social_currency(self, symbol: str) -> float:
        """Calculate social currency score based on engagement metrics"""
        mentions = random.randint(100, 1000)
        engagement_rate = random.uniform(0.01, 0.05)
        influencer_ratio = random.uniform(0.1, 0.3)

        return min(1.0, (mentions / 1000) * 0.4 + engagement_rate * 0.3 + influencer_ratio * 0.3)

    def _generate_market_triggers(self, symbol: str) -> Dict:
        """Identify market events triggering discussion"""
        triggers = {
            'earnings_related': random.uniform(0, 1) > 0.7,
            'news_impact': random.uniform(0, 1),
            'technical_triggers': random.uniform(0, 1),
            'macro_events': random.uniform(0, 1) > 0.8
        }
        return triggers

    def _analyze_emotional_content(self, symbol: str) -> float:
        """Analyze emotional content of social media posts"""
        return random.uniform(0, 1)

    def _analyze_public_visibility(self, symbol: str) -> float:
        """Analyze public visibility and reach"""
        return random.uniform(0, 1)

    def _analyze_practical_value(self, symbol: str) -> float:
        """Analyze practical trading value in discussions"""
        return random.uniform(0, 1)

    def _analyze_narrative_impact(self, symbol: str) -> float:
        """Analyze impact of market narratives"""
        return random.uniform(0, 1)

    def _generate_viral_signals(self, viral_score: float) -> Dict:
        """Generate viral trend signals based on STEPPS analysis"""
        signal_strength = viral_score * random.uniform(0.8, 1.2)

        if viral_score > 0.8:
            signal = "STRONG_VIRAL"
            action = "High viral potential detected, monitor for volume surge"
        elif viral_score > 0.6:
            signal = "EMERGING_VIRAL"
            action = "Emerging viral trend, watch for acceleration"
        elif viral_score > 0.4:
            signal = "POTENTIAL_VIRAL"
            action = "Some viral indicators present"
        else:
            signal = "NO_VIRAL"
            action = "No significant viral trends detected"

        return {
            'signal': signal,
            'strength': signal_strength,
            'confidence': max(0.5, min(0.9, viral_score)),
            'suggested_action': action,
            'risk_level': 'HIGH' if signal_strength > 0.7 else 'MEDIUM' if signal_strength > 0.4 else 'LOW'
        }

    def _get_default_sentiment(self) -> Dict:
        """Return default sentiment data in case of errors"""
        return {
            'sentiment_metrics': {
                'overall_score': 0.5,
                'viral_coefficient': 0.0,
                'sentiment_change_24h': 0.0,
                'confidence': 0.5
            },
            'stepps_analysis': {
                'social_currency': 0.0,
                'triggers': {},
                'emotion': 0.0,
                'public': 0.0,
                'practical_value': 0.0,
                'stories': 0.0
            },
            'source_breakdown': {
                'twitter': 0.5,
                'reddit': 0.5,
                'stocktwits': 0.5
            },
            'viral_signals': {
                'signal': 'NO_VIRAL',
                'strength': 0.0,
                'confidence': 0.5,
                'suggested_action': "Insufficient data for analysis",
                'risk_level': "LOW"
            }
        }