import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime

class MarketRegimeClassifier:
    """Advanced Hidden Markov Model-based market regime classifier"""

    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,  # Increased iterations for better convergence
            random_state=42
        )
        self.scaler = StandardScaler()
        self.regime_types = {
            0: 'ACCUMULATION',
            1: 'DISTRIBUTION',
            2: 'BREAKOUT',
            3: 'HIGH_VOLATILITY',
            4: 'CONSOLIDATION'
        }

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare enhanced features for HMM analysis"""
        features = pd.DataFrame(index=df.index)

        # Price-based features
        features['returns'] = df['Close'].pct_change()
        features['volatility'] = features['returns'].rolling(window=20).std()
        features['price_trend'] = df['Close'].rolling(window=5).mean().pct_change()

        # Enhanced volume features
        if 'Volume' in df.columns:
            # Volume trend features
            features['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            features['volume_trend'] = df['Volume'].pct_change()
            features['volume_volatility'] = df['Volume'].pct_change().rolling(window=20).std()

            # Price-volume relationship
            features['price_volume_correlation'] = (
                features['returns'].rolling(5)
                .corr(df['Volume'].pct_change())
            )

            # Volume pressure
            features['buying_pressure'] = (
                (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            ) * df['Volume']
        else:
            # Default values if volume data is missing
            features['volume_ma_ratio'] = 1.0
            features['volume_trend'] = 0.0
            features['volume_volatility'] = 0.0
            features['price_volume_correlation'] = 0.0
            features['buying_pressure'] = 0.0

        # Fill missing values
        features = features.fillna(method='bfill')

        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features

    def fit_predict_regimes(self, df: pd.DataFrame) -> Dict:
        """Fit HMM and predict market regimes with enhanced analysis"""
        features = self.prepare_features(df)

        # Fit HMM and predict states
        self.hmm.fit(features)
        states = self.hmm.predict(features)
        state_probs = self.hmm.predict_proba(features)

        # Calculate regime characteristics
        regime_stats = self.analyze_regime_characteristics(df, states)

        # Calculate transition matrix for next state prediction
        next_state_probs = self.hmm.transmat_[states[-1]]

        # Calculate regime stability
        regime_stability = self._calculate_regime_stability(states)

        return {
            'current_regime': int(states[-1]),
            'regime_probabilities': state_probs[-1].tolist(),
            'next_state_probabilities': next_state_probs.tolist(),
            'regime_stats': regime_stats,
            'historical_states': states.tolist(),
            'regime_stability': regime_stability,
            'timestamp': datetime.now().isoformat()
        }

    def analyze_regime_characteristics(
        self, df: pd.DataFrame, states: np.ndarray
    ) -> List[Dict]:
        """Analyze detailed characteristics of each regime"""
        regime_stats = []

        for i in range(self.n_states):
            regime_mask = states == i
            regime_data = df[regime_mask]

            if len(regime_data) > 0:
                # Basic statistics
                returns = regime_data['Close'].pct_change()

                stats = {
                    'regime_id': i,
                    'regime_type': self.regime_types[i],
                    'avg_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe_ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
                    'occurrence_count': regime_mask.sum(),
                    'avg_duration': self._calculate_avg_duration(states, i)
                }

                # Volume-based statistics if available
                if 'Volume' in df.columns:
                    volume_change = regime_data['Volume'].pct_change()
                    stats.update({
                        'avg_volume': regime_data['Volume'].mean(),
                        'volume_trend': volume_change.mean(),
                        'volume_volatility': volume_change.std(),
                        'volume_profile': self._analyze_volume_profile(regime_data)
                    })
                else:
                    stats.update({
                        'avg_volume': 0.0,
                        'volume_trend': 0.0,
                        'volume_volatility': 0.0,
                        'volume_profile': 'NO_VOLUME_DATA'
                    })

                regime_stats.append(stats)

        return regime_stats

    def get_optimal_stop_distance(
        self, current_regime: int, price: float, volatility: float, volume_profile: Dict = None
    ) -> Dict:
        """Calculate optimal trailing stop distance with volume considerations"""
        # Base distances calibrated by regime type
        base_distances = {
            'ACCUMULATION': 0.02,    # 2% for accumulation phases
            'DISTRIBUTION': 0.03,    # 3% for distribution phases
            'BREAKOUT': 0.025,       # 2.5% for breakouts
            'HIGH_VOLATILITY': 0.04, # 4% for high volatility
            'CONSOLIDATION': 0.015   # 1.5% for consolidation
        }

        regime_type = self.regime_types[current_regime]
        base_distance = base_distances[regime_type]

        # Volatility adjustment
        volatility_adjustment = min(volatility * 2, 0.02)

        # Volume-based adjustment
        volume_adjustment = 0.0
        if volume_profile:
            if volume_profile.get('volume_trend', 0) > 0.5:  # Strong buying pressure
                volume_adjustment = 0.005  # Tighten stop
            elif volume_profile.get('volume_trend', 0) < -0.5:  # Strong selling pressure
                volume_adjustment = 0.01   # Widen stop

        final_distance = base_distance + volatility_adjustment + volume_adjustment

        return {
            'stop_distance': final_distance,
            'components': {
                'base_distance': base_distance,
                'volatility_adjustment': volatility_adjustment,
                'volume_adjustment': volume_adjustment
            },
            'regime_type': regime_type,
            'confidence': self._calculate_stop_confidence(volatility, volume_profile)
        }

    def _calculate_regime_stability(self, states: np.ndarray) -> float:
        """Calculate the stability of the current regime"""
        if len(states) < 20:
            return 0.5

        recent_states = states[-20:]
        current_state = states[-1]

        # Calculate the proportion of the same state in recent history
        stability = np.mean(recent_states == current_state)
        return float(stability)

    def _calculate_avg_duration(self, states: np.ndarray, regime_id: int) -> float:
        """Calculate average duration of a regime in periods"""
        durations = []
        current_duration = 0

        for state in states:
            if state == regime_id:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def _analyze_volume_profile(self, regime_data: pd.DataFrame) -> str:
        """Analyze volume profile characteristics"""
        if 'Volume' not in regime_data.columns:
            return 'NO_VOLUME_DATA'

        avg_volume = regime_data['Volume'].mean()
        volume_trend = regime_data['Volume'].pct_change().mean()

        if volume_trend > 0.02 and avg_volume > regime_data['Volume'].rolling(20).mean().mean():
            return 'INCREASING_VOLUME'
        elif volume_trend < -0.02:
            return 'DECREASING_VOLUME'
        else:
            return 'STABLE_VOLUME'

    def _calculate_stop_confidence(self, volatility: float, volume_profile: Dict = None) -> float:
        """Calculate confidence score for stop-loss recommendation"""
        # Base confidence inversely related to volatility
        base_confidence = max(0.3, 1 - (volatility * 5))

        # Adjust based on volume profile if available
        if volume_profile:
            volume_trend = volume_profile.get('volume_trend', 0)
            if abs(volume_trend) > 0.5:  # Strong volume trend
                base_confidence *= 1.2
            elif abs(volume_trend) < 0.1:  # Weak volume trend
                base_confidence *= 0.8

        return min(1.0, max(0.1, base_confidence))