import json
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
import logging
import requests
import os
from concurrent.futures import ThreadPoolExecutor

# Import models
from src.models.volume_spike_predictor import LSTMVolumePredictor
from src.models.volume_autoencoder import VolumeAutoencoder
from src.models.hmm_regime_classifier import MarketRegimeClassifier
from src.models.ml_volume_analyzer import MLVolumeAnalyzer
from src.models.rag_trade_analyzer import RAGTradeAnalyzer
from src.models.custom_volume_patterns import CustomVolumePatterns

# Import services
from src.services.sentiment_analysis_service import SentimentAnalysisService

logger = logging.getLogger(__name__)

class VolumeAnalysisService:
    """Enhanced service for coordinated volume analysis and alert generation"""

    def __init__(self):
        logger.info("Initializing VolumeAnalysisService")
        self.volume_predictor = LSTMVolumePredictor()
        self.volume_autoencoder = VolumeAutoencoder()
        self.regime_classifier = MarketRegimeClassifier()
        self.ml_analyzer = MLVolumeAnalyzer()
        self.sentiment_analyzer = SentimentAnalysisService()
        self.rag_analyzer = RAGTradeAnalyzer()
        self.custom_patterns = CustomVolumePatterns()  # Add new custom patterns analyzer
        self.is_trained = False

    def train_models(self, historical_data: pd.DataFrame):
        """Train all models with enhanced error handling"""
        try:
            logger.info("Starting model training pipeline")

            # Train ML Volume Analyzer first
            logger.info("Training ML Volume Analyzer...")
            self.ml_analyzer.train(historical_data)

            logger.info("Training LSTM Volume Predictor...")
            self.volume_predictor.train(historical_data)

            logger.info("Training Volume Autoencoder...")
            self.volume_autoencoder.train(historical_data)

            logger.info("Training Market Regime Classifier...")
            self.regime_classifier.fit_predict_regimes(historical_data)

            self.is_trained = True
            logger.info("Successfully completed training all models")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise

    def analyze_volume_patterns(self, current_data: pd.DataFrame) -> Dict:
        """Enhanced comprehensive volume analysis combining all models"""
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return {"error": "Models not trained yet"}

        try:
            logger.info("Starting enhanced volume pattern analysis")

            # Initialize default response structure
            default_analysis = {
                'timestamp': datetime.now().isoformat(),
                'ml_patterns': [],
                'custom_patterns': {'patterns': {}},
                'volume_decomposition': {
                    'accumulation_score': 0.0,
                    'distribution_score': 0.0,
                    'dominant_flow': 'NEUTRAL',
                    'flow_strength': 0.0
                },
                'predicted_spikes': [],
                'volume_anomalies': [],
                'market_regime': {
                    'current_regime': 'NEUTRAL',
                    'regime_type': 'NEUTRAL',
                    'regime_probability': 0.5
                },
                'sentiment': 'NEUTRAL',
                'recent_pattern': {
                    'type': 'NEUTRAL',
                    'strength': 0.0,
                    'confidence': 0.5
                },
                'volume_profile': 'NORMAL',
                'pattern_strength': {
                    'ml_strength': 0.0,
                    'custom_strength': 0.0,
                    'combined_strength': 0.0
                },
                'alerts': []
            }

            # Safely get ML pattern analysis
            try:
                ml_analysis = self.ml_analyzer.detect_volume_patterns(current_data)
                volume_decomposition = self.ml_analyzer.decompose_volume(current_data)
            except Exception as e:
                logger.error(f"Error in ML analysis: {str(e)}")
                ml_analysis = {'patterns': [], 'recent_pattern': {}, 'volume_profile': 'NORMAL'}
                volume_decomposition = default_analysis['volume_decomposition']

            # Safely get custom pattern analysis
            try:
                custom_patterns = self.custom_patterns.detect_all_patterns(current_data)
            except Exception as e:
                logger.error(f"Error in custom pattern analysis: {str(e)}")
                custom_patterns = {'patterns': {}, 'overall_strength': 0.0}

            # Safely get predictions and anomalies
            try:
                predicted_spikes = self.volume_predictor.predict_volume_spikes(current_data)
                volume_anomalies = self.volume_autoencoder.detect_anomalies(current_data)
                regime_analysis = self.regime_classifier.fit_predict_regimes(current_data)
            except Exception as e:
                logger.error(f"Error in predictions and anomalies: {str(e)}")
                predicted_spikes = []
                volume_anomalies = []
                regime_analysis = {'current_regime': 'NEUTRAL', 'regime_probabilities': [0.5]}

            # Safely get sentiment analysis
            try:
                sentiment_data = self.sentiment_analyzer.get_sentiment_analysis(current_data.index[-1])
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                sentiment_data = {'overall_sentiment': 'NEUTRAL'}

            # Combine all analyses with safe dictionary access
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'ml_patterns': ml_analysis.get('patterns', []),
                'custom_patterns': custom_patterns.get('patterns', {}),
                'volume_decomposition': volume_decomposition,
                'predicted_spikes': predicted_spikes or [],
                'volume_anomalies': volume_anomalies or [],
                'market_regime': {
                    'current_regime': regime_analysis.get('current_regime', 'NEUTRAL'),
                    'regime_type': regime_analysis.get('regime_stats', [{}])[0].get('regime_type', 'NEUTRAL'),
                    'regime_probability': max(regime_analysis.get('regime_probabilities', [0.5]))
                },
                'sentiment': sentiment_data.get('overall_sentiment', 'NEUTRAL'),
                'recent_pattern': ml_analysis.get('recent_pattern', default_analysis['recent_pattern']),
                'volume_profile': ml_analysis.get('volume_profile', 'NORMAL'),
                'pattern_strength': {
                    'ml_strength': ml_analysis.get('overall_strength', 0.0),
                    'custom_strength': custom_patterns.get('overall_strength', 0.0),
                    'combined_strength': (
                        ml_analysis.get('overall_strength', 0.0) + 
                        custom_patterns.get('overall_strength', 0.0)
                    ) / 2
                },
                'alerts': []
            }

            # Generate enhanced alerts including custom patterns
            if (predicted_spikes or volume_anomalies or 
                analysis['ml_patterns'] or analysis['custom_patterns']):
                try:
                    analysis['alerts'] = self._generate_enhanced_alerts(
                        predicted_spikes or [],
                        volume_anomalies or [],
                        analysis['ml_patterns'],
                        regime_analysis,
                        analysis['custom_patterns'],
                        current_data
                    )
                    # Send webhook notifications for high-priority alerts
                    self._send_webhook_notifications(analysis['alerts'])
                except Exception as e:
                    logger.error(f"Error generating alerts: {str(e)}")

            return analysis

        except Exception as e:
            logger.error(f"Error during volume pattern analysis: {str(e)}", exc_info=True)
            return default_analysis

    def _generate_enhanced_alerts(
        self,
        predicted_spikes: List[Dict],
        volume_anomalies: List[Dict],
        ml_patterns: List[Dict],
        regime_analysis: Dict,
        custom_patterns: Dict,
        current_data: pd.DataFrame
    ) -> List[Dict]:
        """Generate enhanced actionable alerts from all analysis sources"""
        alerts = []

        try:
            # Process ML-detected patterns
            for pattern in ml_patterns:
                if pattern['confidence'] > 0.7:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': f'ML_PATTERN_{pattern["pattern"]}',
                        'priority': 'HIGH' if pattern['confidence'] > 0.85 else 'MEDIUM',
                        'message': (
                            f"ML detected {pattern['pattern']} pattern "
                            f"(confidence: {pattern['confidence']:.1%})"
                        ),
                        'suggested_action': pattern['suggested_action'],
                        'risk_level': pattern['risk_level'],
                        'confidence': pattern['confidence'],
                        'details': pattern
                    }
                    alerts.append(alert)

            # Process custom patterns
            for pattern_type, patterns in custom_patterns.items():
                if isinstance(patterns, list):
                    for pattern in patterns:
                        if pattern.get('strength', 0) > 0.6:
                            alert = {
                                'timestamp': datetime.now().isoformat(),
                                'type': f'CUSTOM_PATTERN_{pattern["pattern"]}',
                                'priority': 'HIGH' if pattern['strength'] > 0.8 else 'MEDIUM',
                                'message': (
                                    f"Detected {pattern['pattern']} "
                                    f"(strength: {pattern['strength']:.1%})"
                                ),
                                'details': pattern
                            }
                            alerts.append(alert)

            # Process predicted spikes
            for spike in predicted_spikes:
                if spike['volume_ratio'] > 3.0:  # High-priority spike
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'PREDICTED_VOLUME_SPIKE',
                        'priority': 'HIGH',
                        'message': (
                            f"Potential {spike['volume_ratio']:.1f}x volume spike "
                            f"predicted in {spike['predicted_period']}"
                        ),
                        'confidence': spike['confidence_score'],
                        'details': spike
                    }
                    alerts.append(alert)
                    logger.info(f"Generated high-priority spike alert: {alert['message']}")

            # Process volume anomalies
            for anomaly in volume_anomalies:
                if anomaly['anomaly_score'] > 2.0:  # Significant anomaly
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'VOLUME_ANOMALY',
                        'priority': 'HIGH' if anomaly['anomaly_score'] > 3.0 else 'MEDIUM',
                        'message': (
                            f"Unusual volume pattern detected "
                            f"(score: {anomaly['anomaly_score']:.1f})"
                        ),
                        'confidence': anomaly['confidence'],
                        'details': anomaly
                    }
                    alerts.append(alert)
                    logger.info(f"Generated volume anomaly alert: {alert['message']}")

            return alerts

        except Exception as e:
            logger.error(f"Error generating alerts: {str(e)}")
            return []

    def _send_webhook_notifications(self, alerts: List[Dict]):
        """Send webhook notifications for high-priority alerts"""
        high_priority_alerts = [
            alert for alert in alerts 
            if alert['priority'] == 'HIGH'
        ]

        if not high_priority_alerts:
            return

        # Make webhook (if configured)
        make_webhook_url = os.getenv('MAKE_WEBHOOK_URL')
        if make_webhook_url:
            try:
                logger.info("Sending alerts to Make webhook")
                requests.post(
                    make_webhook_url,
                    json={'alerts': high_priority_alerts},
                    timeout=5
                )
                logger.info("Successfully sent alerts to Make webhook")
            except Exception as e:
                logger.error(f"Error sending Make webhook: {str(e)}")

        # Zapier webhook (if configured)
        zapier_webhook_url = os.getenv('ZAPIER_WEBHOOK_URL')
        if zapier_webhook_url:
            try:
                logger.info("Sending alerts to Zapier webhook")
                requests.post(
                    zapier_webhook_url,
                    json={'alerts': high_priority_alerts},
                    timeout=5
                )
                logger.info("Successfully sent alerts to Zapier webhook")
            except Exception as e:
                logger.error(f"Error sending Zapier webhook: {str(e)}")

    def get_external_insights(self, symbol: str, alerts: List[Dict]) -> Dict:
        """Get insights from external APIs and sentiment analysis"""
        try:
            logger.info(f"Fetching external insights for {symbol}")
            insights = {
                'fundamental_catalysts': [],
                'social_sentiment': {},
                'news_impact': []
            }

            # Get sentiment analysis
            sentiment_data = self.sentiment_analyzer.get_sentiment_analysis(symbol)
            insights['social_sentiment'] = sentiment_data['sentiment_metrics']
            insights['social_sentiment']['trending_topics'] = [
                trend['topic'] for trend in sentiment_data['trending_topics']
            ]
            insights['social_sentiment']['source_breakdown'] = sentiment_data['source_breakdown']

            # Add sentiment signals
            insights['sentiment_signals'] = sentiment_data['sentiment_signals']

            # Mock fundamental catalysts (unchanged)
            insights['fundamental_catalysts'] = self._mock_bigdata_insights(symbol)

            logger.info(f"Successfully gathered external insights for {symbol}")
            return insights

        except Exception as e:
            logger.error(f"Error fetching external insights: {str(e)}", exc_info=True)
            return {}

    def _mock_bigdata_insights(self, symbol: str) -> List[Dict]:
        """Mock BigData.com API response"""
        return [
            {
                'type': 'EARNINGS',
                'date': (datetime.now()).isoformat(),
                'description': 'Quarterly earnings report expected',
                'impact': 'HIGH'
            },
            {
                'type': 'ANALYST_RATING',
                'date': (datetime.now()).isoformat(),
                'description': 'Multiple analyst upgrades',
                'impact': 'MEDIUM'
            }
        ]

    def _mock_finchat_sentiment(self, symbol: str) -> Dict:
        """Mock FinChat.io API response"""
        return {
            'overall_sentiment': 0.75,
            'sentiment_change_24h': 0.15,
            'social_volume_change': 2.5,
            'trending_topics': [
                'earnings',
                'product launch',
                'market expansion'
            ],
            'source_breakdown': {
                'twitter': 0.8,
                'reddit': 0.7,
                'stocktwits': 0.75
            }
        }