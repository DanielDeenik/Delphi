
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
from .relative_strength_analyzer import RelativeStrengthAnalyzer
from .beta_performance_analyzer import BetaPerformanceAnalyzer
from .market_structure_analyzer import MarketStructureAnalyzer
from .market_regime_alignment import MarketRegimeAlignment

logger = logging.getLogger(__name__)

class TradeReviewSystem:
    def __init__(self):
        self.strength_analyzer = RelativeStrengthAnalyzer()
        self.beta_analyzer = BetaPerformanceAnalyzer()
        self.structure_analyzer = MarketStructureAnalyzer()
        self.regime_analyzer = MarketRegimeAlignment()
        
    async def review_trade(self, 
                         trade_data: Dict,
                         market_data: Dict,
                         sector_data: Dict) -> Dict:
        """Perform comprehensive trade review"""
        try:
            # Check relative strength
            strength_analysis = await self.strength_analyzer.analyze_sector_strength(
                trade_data['symbol'],
                sector_data
            )
            
            # Analyze beta behavior
            beta_analysis = self.beta_analyzer.analyze_beta_alignment(
                trade_data,
                market_data
            )
            
            # Check breakout validity
            structure_analysis = self.structure_analyzer.analyze_market_structure(
                market_data['price'],
                market_data['volume']
            )
            
            # Verify market conditions
            regime_analysis = self.regime_analyzer.analyze_market_conditions(
                market_data,
                market_data.get('institutional_data', {})
            )
            
            review_results = {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_data.get('trade_id'),
                'symbol': trade_data.get('symbol'),
                'diagnostics': {
                    'sector_strength': {
                        'passed': strength_analysis['is_sector_leader'],
                        'alternative': strength_analysis.get('stronger_alternatives', []),
                        'score': strength_analysis['relative_strength_score']
                    },
                    'beta_behavior': {
                        'passed': beta_analysis['aligned_with_beta'],
                        'deviation': beta_analysis['beta_deviation'],
                        'sentiment_impact': beta_analysis['sentiment_impact']
                    },
                    'breakout_quality': {
                        'passed': structure_analysis['is_valid_breakout'],
                        'compression_quality': structure_analysis.get('compression_quality', 0),
                        'volume_confirmation': structure_analysis.get('volume_confirmed', False)
                    },
                    'market_conditions': {
                        'passed': regime_analysis['market_conditions']['is_favorable'],
                        'alignment_score': regime_analysis['market_conditions']['overall_alignment'],
                        'regime': regime_analysis['market_conditions']['vix_environment']
                    }
                },
                'overall_score': self._calculate_overall_score(
                    strength_analysis,
                    beta_analysis,
                    structure_analysis,
                    regime_analysis
                ),
                'recommendations': self._generate_recommendations(
                    strength_analysis,
                    beta_analysis,
                    structure_analysis,
                    regime_analysis
                )
            }
            
            # Store review for future optimization
            await self._store_review(review_results)
            
            return review_results
            
        except Exception as e:
            logger.error(f"Error in trade review: {str(e)}")
            return {'error': str(e)}
            
    def _calculate_overall_score(self,
                               strength: Dict,
                               beta: Dict,
                               structure: Dict,
                               regime: Dict) -> float:
        """Calculate overall trade quality score"""
        weights = {
            'strength': 0.3,
            'beta': 0.2,
            'structure': 0.3,
            'regime': 0.2
        }
        
        scores = {
            'strength': strength['relative_strength_score'],
            'beta': 1 - abs(beta['beta_deviation']),
            'structure': structure.get('breakout_score', 0),
            'regime': regime['market_conditions']['overall_alignment']
        }
        
        return sum(scores[k] * weights[k] for k in weights)
        
    def _generate_recommendations(self,
                                strength: Dict,
                                beta: Dict,
                                structure: Dict,
                                regime: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not strength['is_sector_leader']:
            recommendations.append(
                f"Consider stronger alternatives: {', '.join(strength['stronger_alternatives'][:3])}"
            )
            
        if abs(beta['beta_deviation']) > 0.2:
            recommendations.append(
                "Review news and sentiment impact on price behavior"
            )
            
        if not structure['is_valid_breakout']:
            recommendations.append(
                "Enhance breakout confirmation criteria with volume analysis"
            )
            
        if not regime['market_conditions']['is_favorable']:
            recommendations.append(
                "Implement stricter entry criteria during unfavorable conditions"
            )
            
        return recommendations
        
    async def _store_review(self, review: Dict) -> None:
        """Store trade review for strategy optimization"""
        try:
            # TODO: Implement storage in Pinecone or similar vector DB
            pass
        except Exception as e:
            logger.error(f"Error storing trade review: {str(e)}")
