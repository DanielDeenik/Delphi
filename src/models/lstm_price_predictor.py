import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class LSTMPricePredictor:
    """Price prediction model using Random Forest instead of LSTM"""

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model with proper validation and cleaning"""
        # Create copy to avoid modifying original data
        df = df.copy()

        # Calculate features with proper error handling
        df['returns'] = df['Close'].pct_change().clip(-1, 1)  # Clip extreme returns

        # Calculate volume MA with error handling
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['volume_ma_ratio'] = (df['Volume'] / volume_ma).clip(0, 10)  # Clip extreme ratios

        # Calculate price MA with error handling
        price_ma = df['Close'].rolling(window=20).mean()
        df['price_ma_ratio'] = (df['Close'] / price_ma).clip(0, 10)  # Clip extreme ratios

        # Calculate safer volume trend
        df['volume_trend'] = df['Volume'].pct_change().clip(-1, 1)
        df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)

        # Replace any remaining infinities with 0
        df = df.replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values, then fill remaining with 0
        df = df.fillna(method='ffill').fillna(0)

        # Scale features with bounds checking
        price_features = df[['Close', 'returns', 'price_ma_ratio']].values
        volume_features = df[['Volume', 'volume_ma_ratio', 'volume_trend']].values

        # Ensure no NaN or inf values before scaling
        assert not np.any(np.isnan(price_features)), "NaN values in price features"
        assert not np.any(np.isnan(volume_features)), "NaN values in volume features"

        scaled_prices = self.price_scaler.fit_transform(price_features)
        scaled_volumes = self.volume_scaler.fit_transform(volume_features)

        # Combine features
        features = np.column_stack([scaled_prices, scaled_volumes])

        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)].flatten())  # Flatten for Random Forest
            y.append(features[i + self.sequence_length, 0])  # Predict next close price

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train the model with error handling"""
        try:
            X, y = self.prepare_data(df)
            if len(X) > 0:
                self.model.fit(X, y)
            else:
                raise ValueError("Not enough data points after preparation")
        except Exception as e:
            import logging
            logging.error(f"Error during model training: {str(e)}")
            raise

    def predict_next_price(self, df: pd.DataFrame) -> Dict:
        """Predict next price movement with confidence score"""
        # Prepare recent data
        X, _ = self.prepare_data(df)
        if len(X) == 0:
            return {
                'predicted_price': df['Close'].iloc[-1],
                'confidence_score': 0.0,
                'volume_impact': self._analyze_volume_impact(df),
                'prediction_time': datetime.now().isoformat(),
                'forecast_horizon': '1 day'
            }

        # Get last sequence
        last_sequence = X[-1:]

        # Make prediction
        predicted_scaled = self.model.predict(last_sequence)[0]

        # Inverse transform prediction
        predicted_features = np.zeros((1, 3))
        predicted_features[0, 0] = predicted_scaled
        predicted_price = self.price_scaler.inverse_transform(predicted_features)[0, 0]

        # Calculate confidence score
        confidence = self._calculate_confidence(df, predicted_price)

        # Calculate volume impact
        volume_impact = self._analyze_volume_impact(df)

        return {
            'predicted_price': float(predicted_price),
            'confidence_score': confidence,
            'volume_impact': volume_impact,
            'prediction_time': datetime.now().isoformat(),
            'forecast_horizon': '1 day'
        }

    def _calculate_confidence(self, df: pd.DataFrame, predicted_price: float) -> float:
        """Calculate confidence score based on recent prediction accuracy"""
        recent_volatility = df['Close'].pct_change().tail(20).std()
        price_range = df['Close'].tail(20).max() - df['Close'].tail(20).min()

        # Lower confidence if volatility is high
        confidence = max(0.1, 1 - (recent_volatility * 10))

        # Adjust confidence based on price range
        if abs(predicted_price - df['Close'].iloc[-1]) > price_range:
            confidence *= 0.8

        return min(1.0, max(0.1, confidence))

    def _analyze_volume_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze recent volume patterns for price impact"""
        recent_volume = df['Volume'].tail(5).values
        avg_volume = df['Volume'].tail(20).mean()

        volume_trend = recent_volume.mean() / avg_volume

        return {
            'trend': 'increasing' if volume_trend > 1.1 else 'decreasing' if volume_trend < 0.9 else 'stable',
            'strength': min(1.0, volume_trend / 2),
            'avg_volume_ratio': float(volume_trend)
        }