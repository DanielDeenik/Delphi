import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple
from datetime import datetime

class LSTMVolumePredictor:
    """LSTM-based volume spike prediction model"""

    def __init__(self, sequence_length: int = 20, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.volume_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()

        # Initialize LSTM model
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 7), 
                               return_sequences=True, name='lstm_layer1'),
            tf.keras.layers.Dropout(0.2, name='dropout_layer1'),
            tf.keras.layers.LSTM(32, name='lstm_layer2'),
            tf.keras.layers.Dense(self.prediction_horizon, name='output_layer')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for LSTM model"""
        features = pd.DataFrame()

        # Volume features
        features['volume'] = df['Volume']
        features['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        features['volume_ma20'] = df['Volume'].rolling(window=20).mean()
        features['volume_std'] = df['Volume'].rolling(window=20).std()

        # Price-volume relationships
        features['price'] = df['Close']
        features['returns'] = df['Close'].pct_change()
        features['volume_price_corr'] = df['Volume'].rolling(20).corr(df['Close'])

        # Scale features
        scaled_volume = self.volume_scaler.fit_transform(features[['volume', 'volume_ma5', 'volume_ma20', 'volume_std']])
        scaled_price = self.price_scaler.fit_transform(features[['price', 'returns', 'volume_price_corr']])

        # Combine scaled features
        scaled_features = np.hstack([scaled_volume, scaled_price[:, [0, 1, 2]]])

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_features[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Train the LSTM model"""
        X, y = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("Not enough data points to train the model")

        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        return history

    def predict_volume_spikes(self, df: pd.DataFrame, threshold: float = 2.0) -> List[Dict]:
        """Predict potential volume spikes"""
        X, _ = self.prepare_features(df)

        if len(X) == 0:
            return []

        # Get last sequence
        last_sequence = X[-1:]

        # Predict future volumes
        predicted_volumes = self.model.predict(last_sequence)[0]

        # Inverse transform predictions
        predicted_volumes_rescaled = self.volume_scaler.inverse_transform(
            np.hstack([predicted_volumes.reshape(-1, 1), np.zeros((len(predicted_volumes), 3))])
        )[:, 0]

        # Current average volume
        current_avg_volume = df['Volume'].tail(20).mean()

        # Detect potential spikes
        spikes = []
        for i, volume in enumerate(predicted_volumes_rescaled, 1):
            volume_ratio = volume / current_avg_volume
            if volume_ratio > threshold:
                spikes.append({
                    'timestamp': datetime.now().isoformat(),
                    'predicted_period': f"t+{i}",
                    'predicted_volume': float(volume),
                    'volume_ratio': float(volume_ratio),
                    'confidence_score': self._calculate_confidence(volume_ratio),
                    'current_avg_volume': float(current_avg_volume)
                })

        return spikes

    def _calculate_confidence(self, volume_ratio: float) -> float:
        """Calculate confidence score based on volume ratio"""
        # Higher ratio = lower confidence (more extreme prediction)
        base_confidence = 1.0 - min(0.5, (volume_ratio - 2.0) / 8.0)

        # Bound between 0.1 and 0.9
        return max(0.1, min(0.9, base_confidence))