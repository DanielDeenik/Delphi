
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import pinecone
from src.models.institutional_tracker import InstitutionalTracker
from src.models.custom_volume_patterns import VolumePatternDetector
import logging

logger = logging.getLogger(__name__)

class MosaicOutlierDetector:
    def __init__(self):
        self.institutional_tracker = InstitutionalTracker()
        self.volume_detector = VolumePatternDetector()
        self.pinecone_index = pinecone.Index("market-trends")
        
    def detect_outliers(self, market_data: Dict, sentiment_data: Dict) -> Dict:
        """Detect outliers using mosaic theory approach"""
        try:
            # Analyze institutional behavior
            options_flow = self.institutional_tracker.analyze_options_flow(
                market_data['options_chain']
            )
            
            # Detect volume patterns
            volume_patterns = self.volume_detector.detect_smart_money_patterns(
                market_data['price_data']
            )
            
            # Calculate sentiment outliers
            sentiment_score = self._analyze_sentiment_outliers(sentiment_data)
            
            # Combine signals
            combined_signals = self._combine_signals(
                options_flow,
                volume_patterns,
                sentiment_score
            )
            
            return {
                'timestamp': datetime.now().isoformat(),
                'signals': combined_signals,
                'confidence_score': self._calculate_confidence(combined_signals)
            }
            
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return {}
            
    def _analyze_sentiment_outliers(self, sentiment_data: Dict) -> float:
        """Detect unusual sentiment patterns"""
        try:
            # Calculate z-score of sentiment
            sentiment_values = np.array(sentiment_data['historical_sentiment'])
            z_score = (sentiment_data['current_sentiment'] - np.mean(sentiment_values)) / np.std(sentiment_values)
            
            return float(z_score)
            
        except Exception:
            return 0.0
            
    def _combine_signals(self, options_flow: Dict, volume_patterns: List, sentiment_score: float) -> Dict:
        """Combine different signals into actionable insights"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'overall_direction': 'neutral'
        }
        
        # Check for bullish convergence
        if (options_flow.get('unusual_calls', 0) > options_flow.get('unusual_puts', 0) and
            sentiment_score > 2.0 and
            any(p['pattern'] == 'SMART_MONEY_ACCUMULATION' for p in volume_patterns)):
            signals['buy_signals'].append({
                'strength': 'strong',
                'reason': 'Bullish options flow with positive sentiment and accumulation'
            })
            signals['overall_direction'] = 'bullish'
            
        # Check for bearish convergence
        elif (options_flow.get('unusual_puts', 0) > options_flow.get('unusual_calls', 0) and
              sentiment_score < -2.0 and
              any(p['pattern'] == 'SMART_MONEY_DISTRIBUTION' for p in volume_patterns)):
            signals['sell_signals'].append({
                'strength': 'strong',
                'reason': 'Bearish options flow with negative sentiment and distribution'
            })
            signals['overall_direction'] = 'bearish'
            
        return signals
        
    def _calculate_confidence(self, signals: Dict) -> float:
        """Calculate confidence score for signals"""
        if signals['overall_direction'] == 'neutral':
            return 0.0
            
        signal_strength = len(signals['buy_signals']) + len(signals['sell_signals'])
        return min(0.95, signal_strength * 0.2)  # Cap at 95% confidence
