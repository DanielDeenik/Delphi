import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

class VolumeAnalyzer:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

    def detect_volume_anomalies(self, volume_data):
        """Detect anomalies in volume data using Isolation Forest"""
        scaled_data = self.scaler.fit_transform(volume_data.reshape(-1, 1))
        predictions = self.anomaly_detector.fit_predict(scaled_data)
        return predictions

    def calculate_vwap(self, df):
        """Calculate Volume Weighted Average Price"""
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        return df

    def relative_volume(self, current_volume, avg_volume):
        """Calculate Relative Volume (RVOL)"""
        return current_volume / avg_volume if avg_volume != 0 else 0

    def detect_volume_patterns(self, df):
        """Detect volume patterns based on O'Neil's principles"""
        # Calculate rolling averages and standard deviations
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_std'] = df['Volume'].rolling(window=20).std()
        df['price_change'] = df['Close'].pct_change()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']

        # Define pattern conditions
        patterns = {
            'accumulation': (
                (df['Volume'] > 1.2 * df['volume_ma']) & 
                (df['Close'] > df['Close'].shift(1))
            ),
            'distribution': (
                (df['Volume'] > 1.2 * df['volume_ma']) & 
                (df['Close'] < df['Close'].shift(1))
            ),
            'breakout': (
                (df['Volume'] > 1.5 * df['volume_ma']) & 
                (df['Close'] > df['Close'].rolling(20).max().shift(1))
            ),
            'exhaustion': (
                (df['Volume'] > 3 * df['volume_ma']) & 
                (abs(df['price_change'].rolling(3).sum()) > 0.10)
            )
        }

        # Add climax indicators
        df['is_climax_top'] = (
            (df['price_change'].rolling(3).sum() > 0.10) & 
            (df['volume_ratio'] > 3)
        )
        df['is_panic_selling'] = (
            (df['price_change'].rolling(3).sum() < -0.10) & 
            (df['volume_ratio'] > 3)
        )

        return patterns

    def analyze_market_trend(self, df, window=50):
        """Analyze market trend using accumulation/distribution ratio"""
        patterns = self.detect_volume_patterns(df)
        df['accumulation_days'] = patterns['accumulation'].astype(int)
        df['distribution_days'] = patterns['distribution'].astype(int)

        # Calculate rolling ratio of accumulation vs distribution days
        acc_dist_ratio = (
            df['accumulation_days'].rolling(window).sum() / 
            df['distribution_days'].rolling(window).sum()
        )

        return acc_dist_ratio > 1  # True for bullish, False for bearish