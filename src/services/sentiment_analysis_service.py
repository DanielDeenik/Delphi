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

    TREND_TYPES = {
        'FUNDAMENTAL': 'Trend driven by fundamental business changes',
        'SPECULATIVE': 'Trend driven by speculative retail interest',
        'MIXED': 'Combination of fundamental and speculative factors',
        'UNDEFINED': 'Insufficient data to classify trend type'
    }

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

    def classify_trend_type(self, symbol: str, stepps_metrics: Dict, inst_flow_data: Dict = None) -> Dict:
        """Classify whether a trend is fundamental or speculative using STEPPS framework"""
        try:
            # Calculate component scores
            fundamental_score = (
                stepps_metrics['practical_value'] * 0.4 +  # Highest weight on practical value
                stepps_metrics['triggers'] * 0.3 +         # External factors
                stepps_metrics['stories'] * 0.3            # Narrative strength
            )

            speculative_score = (
                stepps_metrics['social_currency'] * 0.4 +  # Social hype
                stepps_metrics['emotion'] * 0.4 +          # Emotional reaction
                stepps_metrics['public'] * 0.2             # Public trends
            )

            # Adjust scores based on institutional flows if available
            if inst_flow_data:
                inst_positioning = inst_flow_data.get('institutional_positioning', 0.5)
                fundamental_score *= (0.7 + 0.6 * inst_positioning)
                speculative_score *= (1.3 - 0.6 * inst_positioning)

            # Normalize scores
            total = fundamental_score + speculative_score
            if total > 0:
                fundamental_score /= total
                speculative_score /= total

            # Classify trend type
            if abs(fundamental_score - speculative_score) < 0.2:
                trend_type = 'MIXED'
            elif fundamental_score > speculative_score:
                trend_type = 'FUNDAMENTAL'
            else:
                trend_type = 'SPECULATIVE'

            return {
                'trend_type': trend_type,
                'classification': self.TREND_TYPES[trend_type],
                'scores': {
                    'fundamental': float(fundamental_score),
                    'speculative': float(speculative_score)
                },
                'confidence': float(abs(fundamental_score - speculative_score)),
                'components': {
                    'fundamental_drivers': {
                        'practical_value': float(stepps_metrics['practical_value']),
                        'triggers': float(stepps_metrics['triggers']),
                        'stories': float(stepps_metrics['stories'])
                    },
                    'speculative_drivers': {
                        'social_currency': float(stepps_metrics['social_currency']),
                        'emotion': float(stepps_metrics['emotion']),
                        'public': float(stepps_metrics['public'])
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error classifying trend type: {str(e)}")
            return {
                'trend_type': 'UNDEFINED',
                'classification': self.TREND_TYPES['UNDEFINED'],
                'scores': {'fundamental': 0.0, 'speculative': 0.0},
                'confidence': 0.0,
                'components': {
                    'fundamental_drivers': {},
                    'speculative_drivers': {}
                }
            }

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
                'triggers': self._analyze_market_triggers(symbol),
                'emotion': self._analyze_emotional_content(symbol),
                'public': self._analyze_public_visibility(symbol),
                'practical_value': self._analyze_practical_value(symbol),
                'stories': self._analyze_narrative_impact(symbol)
            }

            # Mock institutional flow data
            inst_flow_data = {
                'institutional_positioning': random.uniform(0, 1),
                'flow_strength': random.uniform(0, 1),
                'conviction_level': random.uniform(0, 1)
            }

            # Calculate trend classification
            trend_analysis = self.classify_trend_type(symbol, stepps_metrics, inst_flow_data)

            # Source-specific sentiment calculation
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
                'trend_classification': trend_analysis,
                'institutional_flows': inst_flow_data,
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
                }
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

    def _analyze_market_triggers(self, symbol: str) -> float:
        """Analyze market events and macro factors triggering discussion"""
        macro_impact = random.uniform(0, 1)
        news_relevance = random.uniform(0, 1)
        event_significance = random.uniform(0, 1)

        return (macro_impact * 0.4 + news_relevance * 0.3 + event_significance * 0.3)

    def _analyze_emotional_content(self, symbol: str) -> float:
        """Analyze emotional content and sentiment of social media posts"""
        return random.uniform(0, 1)

    def _analyze_public_visibility(self, symbol: str) -> float:
        """Analyze public visibility and cultural relevance"""
        return random.uniform(0, 1)

    def _analyze_practical_value(self, symbol: str) -> float:
        """Analyze fundamental business value and product offerings"""
        return random.uniform(0, 1)

    def _analyze_narrative_impact(self, symbol: str) -> float:
        """Analyze the strength and credibility of company narrative"""
        return random.uniform(0, 1)

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
                'triggers': 0.0,
                'emotion': 0.0,
                'public': 0.0,
                'practical_value': 0.0,
                'stories': 0.0
            },
            'trend_classification': {
                'trend_type': 'UNDEFINED',
                'classification': self.TREND_TYPES['UNDEFINED'],
                'scores': {'fundamental': 0.0, 'speculative': 0.0},
                'confidence': 0.0,
                'components': {}
            },
            'institutional_flows': {
                'institutional_positioning': 0.0,
                'flow_strength': 0.0,
                'conviction_level': 0.0
            },
            'source_breakdown': {
                'twitter': 0.5,
                'reddit': 0.5,
                'stocktwits': 0.5
            }
        }