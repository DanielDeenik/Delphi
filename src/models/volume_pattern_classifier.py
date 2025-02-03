import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

class VolumePatternClassifier:
    """Machine Learning-based volume pattern classifier"""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.pattern_labels = {
            0: 'ACCUMULATION',
            1: 'DISTRIBUTION',
            2: 'BREAKOUT',
            3: 'FAKE_BREAKOUT',
            4: 'CLIMAX'
        }
        
    def extract_pattern_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for pattern classification"""
        features = pd.DataFrame()
        
        # Volume ratios
        features['volume_ma5_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
        features['volume_ma20_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        features['volume_ma50_ratio'] = df['Volume'] / df['Volume'].rolling(window=50).mean()
        
        # Price-volume relationships
        features['price_change'] = df['Close'].pct_change()
        features['volume_change'] = df['Volume'].pct_change()
        features['price_volume_corr'] = (
            features['price_change'].rolling(5)
            .corr(features['volume_change'])
        )
        
        # Trend features
        features['volume_trend'] = (
            df['Volume'].rolling(10).mean() - 
            df['Volume'].rolling(20).mean()
        ) / df['Volume'].rolling(20).mean()
        
        features['price_trend'] = (
            df['Close'].rolling(10).mean() - 
            df['Close'].rolling(20).mean()
        ) / df['Close'].rolling(20).mean()
        
        # Volatility features
        features['volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
        features['price_volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
        
        return self.scaler.fit_transform(features.fillna(0))
        
    def classify_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Classify volume patterns in the data"""
        features = self.extract_pattern_features(df)
        
        # Use rolling window to classify patterns
        window_size = 5
        patterns = []
        
        for i in range(window_size, len(features)):
            window = features[i-window_size:i]
            pattern_features = np.mean(window, axis=0)
            
            # Rule-based pattern classification
            pattern = self._classify_pattern(pattern_features)
            confidence = self._calculate_confidence(pattern_features)
            
            patterns.append({
                'timestamp': df.index[i],
                'pattern': pattern,
                'confidence': confidence,
                'metrics': {
                    'volume_ratio': float(pattern_features[0]),  # volume_ma5_ratio
                    'price_volume_corr': float(pattern_features[5]),  # price_volume_corr
                    'trend_strength': float(pattern_features[6])  # volume_trend
                }
            })
            
        return patterns
        
    def _classify_pattern(self, features: np.ndarray) -> str:
        """Classify pattern based on features"""
        volume_ratio = features[0]  # volume_ma5_ratio
        price_volume_corr = features[5]  # price_volume_corr
        trend = features[6]  # volume_trend
        
        if volume_ratio > 2.0 and price_volume_corr > 0.7:
            return 'BREAKOUT'
        elif volume_ratio > 2.0 and price_volume_corr < -0.7:
            return 'FAKE_BREAKOUT'
        elif volume_ratio > 3.0:
            return 'CLIMAX'
        elif trend > 0.1 and price_volume_corr > 0.5:
            return 'ACCUMULATION'
        elif trend < -0.1 and price_volume_corr < -0.5:
            return 'DISTRIBUTION'
        else:
            return 'NEUTRAL'
            
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for pattern classification"""
        # Use feature magnitudes to determine confidence
        volume_ratio = abs(features[0])
        correlation = abs(features[5])
        trend = abs(features[6])
        
        # Weighted average of key metrics
        confidence = (
            0.4 * min(1.0, volume_ratio / 3.0) +  # Volume ratio contribution
            0.3 * correlation +  # Price-volume correlation contribution
            0.3 * min(1.0, trend * 5.0)  # Trend strength contribution
        )
        
        return min(1.0, max(0.1, confidence))  # Bound between 0.1 and 1.0
