
from typing import Dict, Any
import numpy as np
from .agent_framework import BaseAgent
from .mosaic_agent import MosaicTheoryAgent
from src.models.rag_trade_analyzer import RAGTradeAnalyzer

class TradeDetectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("TradeDetectionAgent")
        self.mosaic_agent = MosaicTheoryAgent()
        self.rag_analyzer = RAGTradeAnalyzer()
        self.outlier_detector = MosaicOutlierDetector()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect trading opportunities using multiple analysis methods"""
        try:
            # Analyze using Mosaic Theory
            mosaic_analysis = self.mosaic_agent.analyze_market_sentiment(
                context['news_data']
            )
            
            # Get similar historical trades
            similar_trades = self.rag_analyzer.find_similar_trades(
                context['market_data']
            )
            
            # Detect outliers and generate signals
            outlier_signals = self.outlier_detector.detect_outliers(
                context['market_data'],
                mosaic_analysis['sentiment']
            )
            
            # Generate trade signals
            signals = self._generate_trade_signals(
                mosaic_analysis,
                similar_trades,
                outlier_signals,
                context
            )
            
            return {
                'trade_signals': signals,
                'confidence_scores': self._calculate_confidence(signals)
            }
            
        except Exception as e:
            logger.error(f"Error in TradeDetectionAgent: {str(e)}")
            return {'error': str(e)}
            
    def _calculate_confidence(self, signals: Dict) -> float:
        """Calculate confidence score for trade signals"""
        weights = {
            'mosaic_score': 0.4,
            'historical_success': 0.3,
            'sentiment_score': 0.3
        }
        
        return np.average(
            [score for score in signals.values()],
            weights=list(weights.values())
        )
