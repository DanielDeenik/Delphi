
from typing import Dict, Optional
import pandas as pd
from datetime import datetime
import logging
from src.config.institutional_config import FILING_THRESHOLDS, OPTIONS_THRESHOLDS, SHORT_THRESHOLDS

logger = logging.getLogger(__name__)

class InstitutionalTracker:
    """Tracks institutional behavior through various indicators"""
    
    def analyze_options_flow(self, options_data: pd.DataFrame) -> Dict:
        """Analyze options flow for unusual activity"""
        try:
            volume_ratio = options_data['volume'] / options_data['avg_volume']
            unusual_activity = volume_ratio > OPTIONS_THRESHOLDS['unusual_volume']
            
            return {
                'unusual_calls': options_data[unusual_activity & (options_data['type'] == 'call')].shape[0],
                'unusual_puts': options_data[unusual_activity & (options_data['type'] == 'put')].shape[0],
                'put_call_ratio': options_data[options_data['type'] == 'put'].shape[0] / \
                                 options_data[options_data['type'] == 'call'].shape[0]
            }
        except Exception as e:
            logger.error(f"Error analyzing options flow: {str(e)}")
            return {}

    def analyze_13f_filings(self, filing_data: pd.DataFrame) -> Dict:
        """Analyze 13F filings for institutional sentiment"""
        try:
            significant_holdings = filing_data[
                filing_data['holding_ratio'] > FILING_THRESHOLDS['significant_holding']
            ]
            
            return {
                'institutional_sentiment': self._calculate_institutional_sentiment(significant_holdings),
                'major_holders': significant_holdings['institution'].unique().tolist(),
                'avg_position_change': significant_holdings['position_change'].mean()
            }
        except Exception as e:
            logger.error(f"Error analyzing 13F filings: {str(e)}")
            return {}

    def analyze_short_interest(self, short_data: pd.DataFrame) -> Dict:
        """Analyze short interest data"""
        try:
            return {
                'short_ratio': short_data['short_interest'].iloc[-1] / short_data['float'].iloc[-1],
                'days_to_cover': short_data['short_interest'].iloc[-1] / short_data['avg_volume'].iloc[-1],
                'is_high_short': (short_data['short_interest'].iloc[-1] / short_data['float'].iloc[-1]) > \
                                SHORT_THRESHOLDS['high_short_interest']
            }
        except Exception as e:
            logger.error(f"Error analyzing short interest: {str(e)}")
            return {}

    def _calculate_institutional_sentiment(self, holdings: pd.DataFrame) -> float:
        """Calculate institutional sentiment score"""
        if holdings.empty:
            return 0.0
            
        position_changes = holdings['position_change']
        weighted_sentiment = (position_changes * holdings['holding_ratio']).sum() / holdings['holding_ratio'].sum()
        return max(min(weighted_sentiment, 1.0), -1.0)
