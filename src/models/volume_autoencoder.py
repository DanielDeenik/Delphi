import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
from datetime import datetime

class VolumeAutoencoder:
    """Autoencoder for unsupervised volume anomaly detection"""

    def __init__(self, sequence_length: int = 20, encoding_dim: int = 8):
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.reconstruction_error_threshold = None

        # Initialize autoencoder model
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build autoencoder architecture"""
        # Input layer
        input_layer = tf.keras.layers.Input(shape=(self.sequence_length, 5), name='encoder_input')

        # Encoder
        x = tf.keras.layers.LSTM(32, return_sequences=True, name='encoder_lstm1')(input_layer)
        x = tf.keras.layers.Dropout(0.2, name='encoder_dropout1')(x)
        x = tf.keras.layers.LSTM(16, name='encoder_lstm2')(x)
        encoded = tf.keras.layers.Dense(self.encoding_dim, activation='relu', name='encoder_output')(x)

        # Decoder
        x = tf.keras.layers.Dense(16, activation='relu', name='decoder_dense1')(encoded)
        x = tf.keras.layers.RepeatVector(self.sequence_length, name='decoder_repeat')(x)
        x = tf.keras.layers.LSTM(16, return_sequences=True, name='decoder_lstm1')(x)
        x = tf.keras.layers.LSTM(32, return_sequences=True, name='decoder_lstm2')(x)
        decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5), name='decoder_output')(x)

        # Create model
        model = tf.keras.Model(input_layer, decoded)
        model.compile(optimizer='adam', loss='mse')

        return model

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for autoencoder"""
        features = pd.DataFrame()

        # Volume features
        features['volume'] = df['Volume']
        features['volume_ma5'] = df['Volume'].rolling(window=5).mean()
        features['price_volume_ratio'] = df['Close'] * df['Volume']
        features['volume_trend'] = df['Volume'].pct_change()
        features['volatility'] = df['Close'].pct_change().rolling(window=20).std()

        # Scale features
        scaled_features = self.scaler.fit_transform(features)

        # Create sequences
        X = []
        for i in range(len(scaled_features) - self.sequence_length + 1):
            X.append(scaled_features[i:(i + self.sequence_length)])

        return np.array(X)

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """Train the autoencoder model"""
        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("Not enough data points to train the model")

        # Train model
        history = self.model.fit(
            X, X,  # Autoencoder tries to reconstruct its input
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

        # Calculate reconstruction error threshold
        predictions = self.model.predict(X)
        reconstruction_errors = np.mean(np.abs(X - predictions), axis=(1, 2))
        self.reconstruction_error_threshold = np.percentile(reconstruction_errors, 95)

        return history

    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume anomalies using autoencoder reconstruction error"""
        if self.reconstruction_error_threshold is None:
            raise ValueError("Model must be trained before detecting anomalies")

        X = self.prepare_features(df)

        if len(X) == 0:
            return []

        # Get reconstructions
        reconstructions = self.model.predict(X)

        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.abs(X - reconstructions), axis=(1, 2))

        # Detect anomalies
        anomalies = []
        for i, error in enumerate(reconstruction_errors):
            if error > self.reconstruction_error_threshold:
                anomaly_score = (error - self.reconstruction_error_threshold) / self.reconstruction_error_threshold
                anomalies.append({
                    'timestamp': df.index[i + self.sequence_length - 1],
                    'reconstruction_error': float(error),
                    'anomaly_score': float(anomaly_score),
                    'confidence': self._calculate_confidence(anomaly_score),
                    'pattern_start': df.index[i].isoformat(),
                    'pattern_end': df.index[i + self.sequence_length - 1].isoformat()
                })

        return anomalies

    def _calculate_confidence(self, anomaly_score: float) -> float:
        """Calculate confidence score based on anomaly score"""
        # Higher anomaly score = higher confidence it's a true anomaly
        base_confidence = min(0.9, 0.5 + anomaly_score / 4.0)

        # Bound between 0.1 and 0.9
        return max(0.1, base_confidence)

    def get_pattern_importance(self, sequence: np.ndarray) -> Dict:
        """Analyze which features contributed most to the anomaly"""
        # Get reconstruction
        reconstruction = self.model.predict(sequence.reshape(1, self.sequence_length, 5))[0]

        # Calculate feature-wise reconstruction errors
        feature_errors = np.mean(np.abs(sequence - reconstruction), axis=0)

        # Feature names
        feature_names = ['volume', 'volume_ma5', 'price_volume_ratio', 'volume_trend', 'volatility']

        # Calculate importance scores
        total_error = np.sum(feature_errors)
        importance_scores = {
            name: float(error / total_error)
            for name, error in zip(feature_names, feature_errors)
        }

        return importance_scores