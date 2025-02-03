import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CustomVolumePatterns:
    """Advanced volume pattern recognition system"""
    
    def __init__(self):
        self.pattern_definitions = {
            'SMART_MONEY_ACCUMULATION': {
                'description': 'High volume on dips with price stabilization',
                'strength_threshold': 0.7
            },
            'SMART_MONEY_DISTRIBUTION': {
                'description': 'High volume on rallies with price weakness',
                'strength_threshold': 0.7
            },
            'VOLUME_CLIMAX': {
                'description': 'Extreme volume spike with price exhaustion',
                'strength_threshold': 0.8
            },
            'CHAIKIN_ACCUMULATION': {
                'description': 'Positive money flow with rising prices',
                'strength_threshold': 0.6
            },
            'CHAIKIN_DISTRIBUTION': {
                'description': 'Negative money flow with falling prices',
                'strength_threshold': 0.6
            }
        }
    
    def calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow indicator"""
        try:
            # Money Flow Multiplier
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0)
            
            # Money Flow Volume
            mf_volume = mf_multiplier * df['Volume']
            
            # Chaikin Money Flow
            cmf = mf_volume.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
            return cmf
            
        except Exception as e:
            logger.error(f"Error calculating Chaikin Money Flow: {str(e)}")
            return pd.Series(0, index=df.index)

    def calculate_force_index(self, df: pd.DataFrame, period: int = 13) -> pd.Series:
        """Calculate Force Index indicator"""
        try:
            force = df['Close'].diff() * df['Volume']
            force_index = force.ewm(span=period, adjust=False).mean()
            return force_index
            
        except Exception as e:
            logger.error(f"Error calculating Force Index: {str(e)}")
            return pd.Series(0, index=df.index)

    def detect_smart_money_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect smart money patterns using volume and price action"""
        try:
            patterns = []
            
            # Calculate required metrics
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['price_ma'] = df['Close'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            df['price_change'] = df['Close'].pct_change()
            
            # Smart Money Accumulation
            accumulation = (
                (df['volume_ratio'] > 1.5) &  # High volume
                (df['Close'] < df['price_ma']) &  # Price below MA
                (df['Close'] > df['Low'])  # Price holding above lows
            )
            
            if accumulation.any():
                patterns.append({
                    'pattern': 'SMART_MONEY_ACCUMULATION',
                    'strength': float(df.loc[accumulation, 'volume_ratio'].mean()),
                    'price_level': float(df.loc[accumulation, 'Close'].iloc[-1]),
                    'volume_ratio': float(df.loc[accumulation, 'volume_ratio'].iloc[-1])
                })
            
            # Smart Money Distribution
            distribution = (
                (df['volume_ratio'] > 1.5) &  # High volume
                (df['Close'] > df['price_ma']) &  # Price above MA
                (df['Close'] < df['High'])  # Price failing at highs
            )
            
            if distribution.any():
                patterns.append({
                    'pattern': 'SMART_MONEY_DISTRIBUTION',
                    'strength': float(df.loc[distribution, 'volume_ratio'].mean()),
                    'price_level': float(df.loc[distribution, 'Close'].iloc[-1]),
                    'volume_ratio': float(df.loc[distribution, 'volume_ratio'].iloc[-1])
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting smart money patterns: {str(e)}")
            return []

    def detect_volume_climax(self, df: pd.DataFrame) -> Dict:
        """Detect volume climax patterns"""
        try:
            # Calculate volume and price metrics
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_std'] = df['Volume'].rolling(window=20).std()
            df['price_change'] = df['Close'].pct_change()
            
            # Climax conditions
            volume_spike = df['Volume'] > (df['volume_ma'] + 3 * df['volume_std'])
            price_exhaustion = abs(df['price_change']) > df['price_change'].std() * 2
            
            if volume_spike.any() and price_exhaustion.any():
                return {
                    'pattern': 'VOLUME_CLIMAX',
                    'strength': float(df.loc[volume_spike, 'Volume'].iloc[-1] / df['volume_ma'].iloc[-1]),
                    'price_change': float(df.loc[price_exhaustion, 'price_change'].iloc[-1]),
                    'volume_ratio': float(df.loc[volume_spike, 'Volume'].iloc[-1] / df['volume_ma'].iloc[-1])
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error detecting volume climax: {str(e)}")
            return {}

    def analyze_chaikin_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze patterns using Chaikin Money Flow"""
        try:
            patterns = []
            
            # Calculate CMF
            cmf = self.calculate_chaikin_money_flow(df)
            force_index = self.calculate_force_index(df)
            
            # Chaikin Accumulation
            accumulation = (
                (cmf > 0.2) &  # Strong positive money flow
                (force_index > 0)  # Positive force index
            )
            
            if accumulation.any():
                patterns.append({
                    'pattern': 'CHAIKIN_ACCUMULATION',
                    'strength': float(cmf[accumulation].mean()),
                    'cmf_value': float(cmf.iloc[-1]),
                    'force_index': float(force_index.iloc[-1])
                })
            
            # Chaikin Distribution
            distribution = (
                (cmf < -0.2) &  # Strong negative money flow
                (force_index < 0)  # Negative force index
            )
            
            if distribution.any():
                patterns.append({
                    'pattern': 'CHAIKIN_DISTRIBUTION',
                    'strength': float(abs(cmf[distribution].mean())),
                    'cmf_value': float(cmf.iloc[-1]),
                    'force_index': float(force_index.iloc[-1])
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing Chaikin patterns: {str(e)}")
            return []

    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect all custom volume patterns"""
        try:
            all_patterns = {
                'smart_money': self.detect_smart_money_patterns(df),
                'volume_climax': self.detect_volume_climax(df),
                'chaikin_patterns': self.analyze_chaikin_patterns(df)
            }
            
            # Calculate overall pattern strength
            pattern_strengths = []
            
            for pattern_type, patterns in all_patterns.items():
                if isinstance(patterns, list):
                    for pattern in patterns:
                        if 'strength' in pattern:
                            pattern_strengths.append(pattern['strength'])
                elif isinstance(patterns, dict) and 'strength' in patterns:
                    pattern_strengths.append(patterns['strength'])
            
            overall_strength = np.mean(pattern_strengths) if pattern_strengths else 0.0
            
            return {
                'patterns': all_patterns,
                'overall_strength': float(overall_strength),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {str(e)}")
            return {
                'patterns': {},
                'overall_strength': 0.0,
                'timestamp': pd.Timestamp.now().isoformat()
            }
