import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import tensorflow as tf

class LSTMPricePredictor:
    """Price prediction model using LSTM with enhanced momentum features"""

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(self.sequence_length, 6), 
                               return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(30, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model with proper validation and cleaning"""
        # Create copy to avoid modifying original data
        df = df.copy()

        # Calculate features with proper error handling
        df['returns'] = df['Close'].pct_change().clip(-1, 1)  # Clip extreme returns

        # Enhanced volume features
        volume_ma = df['Volume'].rolling(window=20).mean()
        df['volume_ma_ratio'] = (df['Volume'] / volume_ma).clip(0, 10)
        df['volume_trend'] = df['Volume'].pct_change().clip(-1, 1)
        df['volume_momentum'] = df['volume_ma_ratio'].rolling(5).mean()

        # Enhanced price momentum features
        price_ma = df['Close'].rolling(window=20).mean()
        df['price_ma_ratio'] = (df['Close'] / price_ma).clip(0, 10)
        df['momentum_score'] = df['returns'].rolling(5).mean() * df['volume_momentum']

        # Additional technical features
        df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)

        # Replace infinities and handle NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)

        # Scale features
        price_features = df[['Close', 'returns', 'price_ma_ratio']].values
        volume_features = df[['Volume', 'volume_ma_ratio', 'volume_trend']].values

        assert not np.any(np.isnan(price_features)), "NaN values in price features"
        assert not np.any(np.isnan(volume_features)), "NaN values in volume features"

        scaled_prices = self.price_scaler.fit_transform(price_features)
        scaled_volumes = self.volume_scaler.fit_transform(volume_features)

        # Combine features
        features = np.column_stack([scaled_prices, scaled_volumes])

        X, y = [], []
        for i in range(len(features) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(features[i + self.sequence_length, 0])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train the model with enhanced momentum features"""
        try:
            X, y = self.prepare_data(df)

            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                raise ValueError("NaN values detected in training data")

            if len(X) > 0:
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=0.001,
                    clipnorm=1.0,
                    clipvalue=0.5
                )
                self.model.compile(
                    optimizer=optimizer,
                    loss='huber',
                    metrics=['mae']
                )

                # Add callbacks for better training
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.0001
                )

                self.model.fit(
                    X, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1
                )
            else:
                raise ValueError("Not enough data points after preparation")

        except Exception as e:
            import logging
            logging.error(f"Error during model training: {str(e)}")
            raise

    def predict_next_price(self, df: pd.DataFrame) -> Dict:
        """Predict next price movement with enhanced momentum scoring"""
        # Prepare recent data
        X, _ = self.prepare_data(df)
        if len(X) == 0:
            return {
                'predicted_price': df['Close'].iloc[-1],
                'confidence_score': 0.0,
                'volume_impact': self._analyze_volume_impact(df),
                'prediction_time': datetime.now().isoformat(),
                'forecast_horizon': '1 day',
                'momentum_score': 0.0
            }

        # Get last sequence
        last_sequence = X[-1:]

        # Make prediction
        predicted_scaled = self.model.predict(last_sequence)[0, 0]

        # Inverse transform prediction
        predicted_features = np.zeros((1, 3))
        predicted_features[0, 0] = predicted_scaled
        predicted_price = self.price_scaler.inverse_transform(predicted_features)[0, 0]

        # Calculate enhanced confidence score
        confidence = self._calculate_confidence(df, predicted_price)

        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(df)

        # Calculate volume impact
        volume_impact = self._analyze_volume_impact(df)

        return {
            'predicted_price': float(predicted_price),
            'confidence_score': confidence,
            'volume_impact': volume_impact,
            'prediction_time': datetime.now().isoformat(),
            'forecast_horizon': '1 day',
            'momentum_score': momentum_score
        }

    def _calculate_confidence(self, df: pd.DataFrame, predicted_price: float) -> float:
        """Calculate enhanced confidence score based on momentum and volume"""
        recent_volatility = df['Close'].pct_change().tail(20).std()
        price_range = df['Close'].tail(20).max() - df['Close'].tail(20).min()

        # Volume-based confidence adjustment
        volume_ratio = df['Volume'].iloc[-1] / df['Volume'].tail(20).mean()
        volume_confidence = min(1.0, volume_ratio / 3.0)

        # Price momentum confidence
        returns = df['Close'].pct_change()
        momentum_strength = abs(returns.tail(5).mean()) / returns.tail(20).std()
        momentum_confidence = min(1.0, momentum_strength)

        # Combine confidence metrics
        base_confidence = max(0.1, 1 - (recent_volatility * 10))
        if abs(predicted_price - df['Close'].iloc[-1]) > price_range:
            base_confidence *= 0.8

        # Weight the components
        confidence = (
            0.4 * base_confidence + 
            0.3 * volume_confidence + 
            0.3 * momentum_confidence
        )

        return min(1.0, max(0.1, confidence))

    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (0-100) based on multiple factors"""
        # Price momentum
        returns = df['Close'].pct_change()
        price_momentum = returns.tail(5).mean() / returns.tail(20).std()

        # Volume momentum
        volume_ratio = df['Volume'].tail(5).mean() / df['Volume'].tail(20).mean()
        volume_trend = (volume_ratio - 1) / df['Volume'].tail(20).std()

        # Combine and normalize to 0-100 scale
        raw_score = (0.6 * price_momentum + 0.4 * volume_trend)
        normalized_score = (np.tanh(raw_score) + 1) * 50

        return float(normalized_score)

    def _analyze_volume_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze recent volume patterns for momentum impact"""
        recent_volume = df['Volume'].tail(5).values
        avg_volume = df['Volume'].tail(20).mean()

        volume_trend = recent_volume.mean() / avg_volume

        # Calculate volume momentum
        volume_change = df['Volume'].pct_change().tail(5).mean()
        price_correlation = df['Close'].pct_change().tail(5).corr(
            df['Volume'].pct_change().tail(5)
        )

        momentum_strength = volume_change * price_correlation

        trend_classification = (
            'accelerating' if volume_trend > 1.2 and momentum_strength > 0
            else 'fading' if volume_trend < 0.8 or momentum_strength < 0
            else 'stable'
        )

        return {
            'trend': trend_classification,
            'strength': min(1.0, abs(momentum_strength)),
            'avg_volume_ratio': float(volume_trend)
        }