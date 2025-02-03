from typing import Dict, Optional
import pandas as pd
from datetime import datetime
import logging
import numpy as np
from src.models.hmm_regime_classifier import MarketRegimeClassifier
from src.models.lstm_price_predictor import LSTMPricePredictor

logger = logging.getLogger(__name__)

class TradingSignalService:
    """Service for generating trading signals and stop recommendations"""

    def __init__(self):
        logger.info("Initializing TradingSignalService")
        self.regime_classifier = MarketRegimeClassifier()
        self.price_predictor = LSTMPricePredictor()
        self.is_trained = False

        # Enhanced regime types
        self.REGIME_TYPES = {
            0: "STRONG_BULLISH",
            1: "WEAK_BULLISH",
            2: "NEUTRAL",
            3: "WEAK_BEARISH",
            4: "STRONG_BEARISH",
            5: "HIGH_VOLATILITY",
            6: "LOW_VOLATILITY",
            7: "ACCUMULATION",
            8: "DISTRIBUTION"
        }

    def train_models(self, historical_data: pd.DataFrame):
        """Train both HMM and LSTM models"""
        try:
            if historical_data is None or 'Close' not in historical_data.columns:
                logger.error("Invalid historical data provided")
                raise ValueError("Invalid historical data provided")

            logger.info("Starting model training pipeline")

            # Prepare features for regime classification
            self._prepare_technical_features(historical_data)

            # Train LSTM price predictor
            logger.info("Training LSTM price predictor...")
            self.price_predictor.train(historical_data)

            # Fit HMM classifier with enhanced features
            logger.info("Training HMM regime classifier...")
            self.regime_classifier.fit_predict_regimes(historical_data)

            self.is_trained = True
            logger.info("Successfully completed training all models")

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise

    def _prepare_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced technical features for regime classification"""
        # Price momentum
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()

        # Volume analysis
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']

        # Trend strength
        df['adx'] = self._calculate_adx(df)

        # Clean up NaN values
        return df.dropna()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = low.diff()

        # True range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth with Wilder's smoothing
        smoothed_tr = tr.rolling(period).sum()
        smoothed_plus_dm = plus_dm.rolling(period).sum()
        smoothed_minus_dm = minus_dm.rolling(period).sum()

        # Calculate directional indicators
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def generate_trading_signals(self, current_data: pd.DataFrame) -> Dict:
        """Generate enhanced trading signals and stop recommendations"""
        if not self.is_trained:
            logger.warning("Models not trained yet")
            return {"error": "Models not trained yet"}

        if current_data is None or 'Close' not in current_data.columns:
            logger.error("Invalid data provided for signal generation")
            return {"error": "Invalid data provided"}

        try:
            logger.info("Generating trading signals")

            # Get enhanced market regime analysis
            regime_analysis = self.regime_classifier.fit_predict_regimes(current_data)
            current_regime = self.REGIME_TYPES.get(regime_analysis.get('current_regime', 2), "NEUTRAL")
            logger.info(f"Current market regime: {current_regime}")

            # Ensure regime analysis has all required fields with defaults
            regime_analysis = {
                'current_regime': regime_analysis.get('current_regime', 2),
                'regime_strength': regime_analysis.get('regime_strength', 0.5),
                'regime_duration': regime_analysis.get('regime_duration', 1),
                'regime_volatility': regime_analysis.get('regime_volatility', 0.0),
                'transitions': regime_analysis.get('transitions', {}),
                'next_state_probabilities': regime_analysis.get('next_state_probabilities', [0.2] * 5)
            }

            # Calculate regime transition probabilities
            transition_probs = self._calculate_transition_probabilities(regime_analysis)

            # Get price prediction with confidence intervals
            price_prediction = self.price_predictor.predict_next_price(current_data)
            logger.info(f"Price prediction generated with confidence: {price_prediction['confidence_score']:.2f}")

            # Calculate enhanced volume profile
            volume_profile = self._calculate_volume_profile(current_data)

            # Get optimal stop distance with volatility adjustment
            current_price = current_data['Close'].iloc[-1]
            current_volatility = current_data['Close'].pct_change().std()

            stop_recommendation = self.regime_classifier.get_optimal_stop_distance(
                regime_analysis['current_regime'],
                current_price,
                current_volatility,
                volume_profile
            )

            # Calculate regime stability score
            regime_stability = self._calculate_regime_stability(regime_analysis)

            # Generate combined insights with enhanced metrics
            signals = {
                'timestamp': datetime.now().isoformat(),
                'market_regime': {
                    'regime_type': current_regime,
                    'regime_strength': regime_analysis['regime_strength'],
                    'regime_duration': regime_analysis['regime_duration'],
                    'regime_stability': regime_stability,
                    'regime_probability': max(regime_analysis['next_state_probabilities']),
                    'transition_probabilities': transition_probs,
                    'next_state_probabilities': regime_analysis['next_state_probabilities']
                },
                'price_forecast': {
                    'predicted_price': price_prediction.get('predicted_price', current_price),
                    'confidence_score': price_prediction.get('confidence_score', 0.5),
                    'prediction_interval': price_prediction.get('prediction_interval', {'lower': current_price * 0.95, 'upper': current_price * 1.05}),
                    'volume_impact': self._analyze_volume_impact(volume_profile)
                },
                'stop_recommendation': {
                    'optimal_distance_percent': stop_recommendation.get('stop_distance', 0.02),
                    'suggested_stop_price': current_price * (1 - stop_recommendation.get('stop_distance', 0.02)),
                    'confidence_level': stop_recommendation.get('confidence', 0.5),
                    'volatility_adjustment': stop_recommendation.get('volatility_adjustment', 1.0),
                    'components': stop_recommendation.get('components', {})
                },
                'risk_analysis': {
                    'current_volatility': float(current_volatility),
                    'regime_volatility': regime_analysis['regime_volatility'],
                    'volume_profile': volume_profile,
                    'risk_score': self._calculate_risk_score(
                        current_volatility,
                        regime_analysis,
                        volume_profile
                    )
                }
            }

            logger.info("Successfully generated enhanced trading signals")
            return signals

        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to generate signals: {str(e)}",
                "market_regime": {
                    "regime_type": "NEUTRAL",
                    "regime_probability": 0.5,
                    "regime_strength": 0.5
                }
            }

    def _calculate_transition_probabilities(self, regime_analysis: Dict) -> Dict:
        """Calculate enhanced regime transition probabilities"""
        transitions = regime_analysis.get('transitions', {})
        current_regime = regime_analysis['current_regime']

        probs = {}
        for next_regime in self.REGIME_TYPES.keys():
            if (current_regime, next_regime) in transitions:
                prob = transitions[(current_regime, next_regime)]
                probs[self.REGIME_TYPES[next_regime]] = float(prob)
            else:
                probs[self.REGIME_TYPES[next_regime]] = 0.0

        return probs

    def _calculate_regime_stability(self, regime_analysis: Dict) -> float:
        """Calculate enhanced regime stability score"""
        duration = regime_analysis.get('regime_duration', 1)
        strength = regime_analysis.get('regime_strength', 0.5)
        volatility = regime_analysis.get('regime_volatility', 0.0)

        # Normalize duration (max 30 days)
        duration_factor = min(duration / 30.0, 1.0)

        # Calculate stability score
        stability = (
            0.4 * duration_factor +
            0.4 * strength +
            0.2 * (1.0 - min(volatility * 10, 1.0))
        )

        return float(stability)

    def _calculate_risk_score(self, volatility: float, regime_analysis: Dict, volume_profile: Dict) -> float:
        """Calculate comprehensive risk score"""
        # Normalize inputs
        vol_score = min(volatility * 10, 1.0)
        regime_risk = {
            "STRONG_BULLISH": 0.2,
            "WEAK_BULLISH": 0.4,
            "NEUTRAL": 0.5,
            "WEAK_BEARISH": 0.6,
            "STRONG_BEARISH": 0.8,
            "HIGH_VOLATILITY": 0.9,
            "LOW_VOLATILITY": 0.3,
            "ACCUMULATION": 0.4,
            "DISTRIBUTION": 0.7
        }.get(self.REGIME_TYPES[regime_analysis['current_regime']], 0.5)

        volume_risk = min(volume_profile['volume_strength'], 1.0)

        # Calculate weighted risk score
        risk_score = (
            0.4 * vol_score +
            0.4 * regime_risk +
            0.2 * volume_risk
        )

        return float(risk_score)

    def _analyze_volume_impact(self, volume_profile: Dict) -> Dict:
        """Analyze volume impact on price action"""
        volume_trend = volume_profile['volume_trend']
        volume_strength = volume_profile['volume_strength']

        if volume_strength < 0.2:
            impact = "NEGLIGIBLE"
        elif volume_trend > 0.5 and volume_strength > 0.7:
            impact = "STRONG_BULLISH"
        elif volume_trend < -0.5 and volume_strength > 0.7:
            impact = "STRONG_BEARISH"
        elif volume_trend > 0.2:
            impact = "MODERATE_BULLISH"
        elif volume_trend < -0.2:
            impact = "MODERATE_BEARISH"
        else:
            impact = "NEUTRAL"

        return {
            'trend': impact,
            'strength': float(volume_strength),
            'momentum': float(volume_trend)
        }

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate enhanced volume profile metrics"""
        try:
            if 'Volume' not in df.columns:
                logger.warning("No volume data available")
                return {
                    'volume_trend': 0.0,
                    'volume_strength': 0.0,
                    'volume_momentum': 0.0,
                    'status': 'NO_VOLUME_DATA'
                }

            # Calculate volume metrics
            recent_volume = df['Volume'].tail(5).values
            avg_volume = df['Volume'].tail(20).mean()

            # Enhanced metrics
            volume_trend = (recent_volume.mean() / avg_volume) - 1 if avg_volume > 0 else 0.0
            volume_strength = min(1.0, abs(volume_trend))

            # Volume momentum (acceleration)
            volume_changes = np.diff(recent_volume) / recent_volume[:-1]
            volume_momentum = np.mean(volume_changes) if len(volume_changes) > 0 else 0.0

            logger.debug(f"Calculated volume profile - trend: {volume_trend:.2f}, strength: {volume_strength:.2f}")
            return {
                'volume_trend': float(volume_trend),
                'volume_strength': float(volume_strength),
                'volume_momentum': float(volume_momentum),
                'status': 'VALID'
            }
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}", exc_info=True)
            return {
                'volume_trend': 0.0,
                'volume_strength': 0.0,
                'volume_momentum': 0.0,
                'status': f'ERROR: {str(e)}'
            }
    def get_entry_signals(self, data: pd.DataFrame) -> Optional[Dict]:
        """Generate entry signals based on market analysis"""
        try:
            if not self.is_trained:
                logger.warning("Models not trained yet")
                return None

            # Get current trading signals
            signals = self.generate_trading_signals(data)
            if 'error' in signals:
                return None

            # Extract key metrics
            regime_info = signals['market_regime']
            price_forecast = signals['price_forecast']
            current_price = data['Close'].iloc[-1]

            # Initialize signal components
            signal_type = None
            confidence = 0.0
            reasoning = []
            expected_return = 0.0

            # Analyze regime conditions
            if regime_info['regime_type'] in ['STRONG_BULLISH', 'WEAK_BULLISH']:
                signal_type = 'LONG'
                confidence = regime_info['regime_stability'] * 0.6 + price_forecast['confidence_score'] * 0.4
                reasoning.append(f"Market regime is {regime_info['regime_type']}")

                # Calculate expected return
                predicted_price = price_forecast['predicted_price']
                expected_return = (predicted_price - current_price) / current_price

            elif regime_info['regime_type'] in ['STRONG_BEARISH', 'WEAK_BEARISH']:
                signal_type = 'SHORT'
                confidence = regime_info['regime_stability'] * 0.6 + price_forecast['confidence_score'] * 0.4
                reasoning.append(f"Market regime is {regime_info['regime_type']}")

                # Calculate expected return (for short positions)
                predicted_price = price_forecast['predicted_price']
                expected_return = (current_price - predicted_price) / current_price

            # Add volume impact to reasoning
            if price_forecast['volume_impact']['trend'] != 'NEUTRAL':
                reasoning.append(f"Volume analysis shows {price_forecast['volume_impact']['trend']} trend")

            # Generate entry signal if conditions are met
            if signal_type and confidence > 0.6:
                suggested_stop = (
                    signals['stop_recommendation']['suggested_stop_price']
                    if 'stop_recommendation' in signals
                    else None
                )

                return {
                    'signal': signal_type,
                    'confidence': confidence,
                    'expected_return': expected_return,
                    'suggested_stop': suggested_stop,
                    'reasoning': reasoning
                }

            return None

        except Exception as e:
            logger.error(f"Error generating entry signals: {str(e)}", exc_info=True)
            return None