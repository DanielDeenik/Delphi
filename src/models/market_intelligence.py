from typing import Dict, List
import pandas as pd
from datetime import datetime
import pinecone
from src.models.rag_trade_analyzer import RAGTradeAnalyzer
from src.models.hmm_regime_classifier import MarketRegimeClassifier
from src.models.lstm_price_predictor import LSTMPricePredictor

class RAGMarketMemory:
    def __init__(self, pinecone_api_key: str, environment: str):
        pinecone.init(api_key=pinecone_api_key, environment=environment)
        self.index_name = "market-memory"
        self.index = pinecone.Index(self.index_name)

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def query(self, query_vector, top_k=5):
        results = self.index.query(vector=query_vector, top_k=top_k)
        return results['matches']


class MarketIntelligence:
    def __init__(self, pinecone_api_key: str, environment: str):
        pinecone.init(api_key=pinecone_api_key, environment=environment)
        self.index_name = "market-intelligence"
        self.index = pinecone.Index(self.index_name)
        self.rag_analyzer = RAGTradeAnalyzer()
        self.regime_classifier = MarketRegimeClassifier(n_regimes=3)
        self.market_memory = RAGMarketMemory(pinecone_api_key, environment)
        self.price_predictor = LSTMPricePredictor()

    async def analyze_market_conditions(self, 
                                     market_data: pd.DataFrame,
                                     sentiment_data: Dict,
                                     institutional_data: Dict) -> Dict:
        """Analyze market conditions using multiple data sources"""
        try:
            # Get market regime prediction
            regime = self.regime_classifier.predict_regime(market_data)

            # Get price predictions
            price_forecast = self.price_predictor.predict_next_movement(market_data)

            # Get similar historical patterns
            similar_patterns = self.rag_analyzer.get_similar_trades({
                'market_regime': regime,
                'sentiment': sentiment_data.get('overall_sentiment'),
                'volume_profile': market_data['volume'].tail(5).mean()
            })

            # Generate trade insights
            insights = self.rag_analyzer.generate_trade_insights(
                current_conditions={
                    'regime': regime,
                    'forecast': price_forecast,
                    'sentiment': sentiment_data
                },
                similar_trades=similar_patterns
            )

            return {
                'timestamp': datetime.now().isoformat(),
                'market_regime': regime,
                'price_forecast': price_forecast,
                'sentiment_analysis': sentiment_data,
                'institutional_flow': institutional_data,
                'trade_insights': insights
            }

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {}