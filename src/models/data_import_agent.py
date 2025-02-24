
from typing import Dict, Any
import pandas as pd
from .agent_framework import BaseAgent
from src.services.sentiment_analysis_service import SentimentAnalysisService
from src.services.volume_analysis_service import VolumeAnalysisService

class DataImportAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataImportAgent")
        self.sentiment_service = SentimentAnalysisService()
        self.volume_service = VolumeAnalysisService()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Import market data, sentiment, and volume analysis"""
        try:
            # Import market data
            market_data = await self._fetch_market_data(context['symbols'])
            
            # Get sentiment data
            sentiment_data = await self.sentiment_service.analyze_market_sentiment(
                context.get('news_data', [])
            )
            
            # Analyze volume
            volume_data = self.volume_service.analyze_volume_patterns(market_data)
            
            return {
                'market_data': market_data,
                'sentiment_data': sentiment_data,
                'volume_data': volume_data
            }
            
        except Exception as e:
            logger.error(f"Error in DataImportAgent: {str(e)}")
            return {'error': str(e)}
