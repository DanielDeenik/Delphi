import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Dict, List, Tuple
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class MomentumPredictor:
    """Predict trend sustainability using LSTM and HMM"""

    REGIME_TYPES = {
        0: "SUSTAINED_MOMENTUM",
        1: "PEAK_HYPE",
        2: "REVERSAL_PHASE",
        3: "DISTRIBUTION"
    }

    def __init__(self, 
                lookback_window: int = 10,
                prediction_horizon: int = 5,
                n_components: int = 4):
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.n_components = n_components
        
        # Initialize models
        self.lstm_model = self._build_lstm_model()
        self.hmm_model = GaussianHMM(n_components=n_components, 
                                    covariance_type="full", 
                                    n_iter=100)
        self.is_trained = False

    def _build_lstm_model(self) -> Sequential:
        """Build LSTM model for momentum prediction"""
        model = Sequential([
            LSTM(64, input_shape=(self.lookback_window, 5), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(self.prediction_horizon)  # Predict next n days
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        return model

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction"""
        features = np.column_stack([
            data['momentum_score'].values,       # Sentiment momentum
            data['volume'].values,               # Trading volume
            data['volatility'].values,           # Price volatility
            data['inst_flows'].values,           # Institutional flows
            data['social_score'].values          # Social sentiment score
        ])
        return features

    def train(self, historical_data: pd.DataFrame):
        """Train both LSTM and HMM models"""
        try:
            features = self.prepare_features(historical_data)
            
            # Prepare sequences for LSTM
            X, y = [], []
            for i in range(len(features) - self.lookback_window - self.prediction_horizon):
                X.append(features[i:(i + self.lookback_window)])
                y.append(features[i + self.lookback_window:
                                i + self.lookback_window + self.prediction_horizon, 0])
            
            X = np.array(X)
            y = np.array(y)

            # Train LSTM
            self.lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

            # Train HMM
            self.hmm_model.fit(features)
            
            self.is_trained = True

        except Exception as e:
            raise Exception(f"Error training momentum predictor: {str(e)}")

    def predict_momentum(self, current_data: pd.DataFrame) -> Dict:
        """Predict momentum trajectory and regime"""
        try:
            if not self.is_trained:
                raise ValueError("Models not trained yet")

            features = self.prepare_features(current_data)
            
            # Get recent window for prediction
            recent_window = features[-self.lookback_window:]
            
            # LSTM prediction
            momentum_pred = self.lstm_model.predict(
                recent_window.reshape(1, self.lookback_window, 5)
            )[0]
            
            # HMM regime prediction
            current_regime = self.hmm_model.predict(recent_window.reshape(-1, 5))[-1]
            regime_probs = self.hmm_model.predict_proba(recent_window.reshape(-1, 5))[-1]
            
            # Calculate trend continuation probability
            trend_direction = np.mean(momentum_pred - momentum_pred[0])
            continuation_prob = float(
                np.exp(-0.5 * abs(trend_direction)) if trend_direction < 0
                else (1 - np.exp(-0.5 * abs(trend_direction)))
            )

            # Determine prediction label
            if continuation_prob > 0.7 and trend_direction > 0:
                prediction = "MOMENTUM_UP"
                action = "Strong retail/institutional synergy, bullish continuation"
            elif continuation_prob > 0.6 and trend_direction < 0:
                prediction = "TREND_REVERSAL"
                action = "Sentiment cooling, institutional selling increasing"
            else:
                prediction = "FADING_MOMENTUM"
                action = "Hype peaking, likely near a top"

            return {
                'momentum_prediction': {
                    'values': momentum_pred.tolist(),
                    'horizon_days': self.prediction_horizon,
                    'continuation_probability': float(continuation_prob),
                    'prediction': prediction,
                    'suggested_action': action
                },
                'regime_analysis': {
                    'current_regime': self.REGIME_TYPES[current_regime],
                    'regime_probabilities': {
                        self.REGIME_TYPES[i]: float(prob)
                        for i, prob in enumerate(regime_probs)
                    }
                },
                'confidence_metrics': {
                    'prediction_confidence': float(max(regime_probs)),
                    'trend_strength': float(abs(trend_direction)),
                    'regime_stability': float(1 - np.std(regime_probs))
                }
            }

        except Exception as e:
            raise Exception(f"Error predicting momentum: {str(e)}")

    def analyze_trend_sustainability(self, 
                                  data: pd.DataFrame,
                                  sentiment_score: float,
                                  volume_profile: Dict) -> Dict:
        """Analyze if current trend is sustainable"""
        try:
            prediction = self.predict_momentum(data)
            
            # Combine momentum prediction with current sentiment
            sustainability_score = (
                prediction['momentum_prediction']['continuation_probability'] * 0.4 +
                sentiment_score * 0.3 +
                volume_profile.get('strength', 0) * 0.3
            )

            return {
                'prediction': prediction,
                'sustainability': {
                    'score': float(sustainability_score),
                    'is_sustainable': sustainability_score > 0.6,
                    'confidence': float(prediction['confidence_metrics']['prediction_confidence'])
                },
                'warning_signals': {
                    'divergence': abs(sentiment_score - sustainability_score) > 0.3,
                    'overextended': sustainability_score < 0.3 and sentiment_score > 0.7,
                    'weakening': prediction['momentum_prediction']['prediction'] == 'FADING_MOMENTUM'
                }
            }

        except Exception as e:
            raise Exception(f"Error analyzing trend sustainability: {str(e)}")
