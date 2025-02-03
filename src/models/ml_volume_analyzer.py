import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)

class MLVolumeAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.volume_clusterer = KMeans(n_clusters=5, random_state=42)
        self.volume_classifier = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.explainer = None
        self.is_trained = False
        self.feature_names = []  # Initialize feature names list

        # Enhanced pattern types with driver attribution
        self.PATTERN_TYPES = {
            0: "ACCUMULATION",
            1: "DISTRIBUTION",
            2: "BREAKOUT",
            3: "EXHAUSTION",
            4: "CLIMAX",
            5: "CONSOLIDATION",
            6: "TREND_CONTINUATION",
            7: "REVERSAL",
            8: "VOLUME_SPIKE",
            9: "LOW_VOLUME_PULLBACK"
        }

    def _extract_advanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract advanced features with named columns for SHAP analysis"""
        try:
            logger.info("Starting feature extraction with enhanced preprocessing")

            # Create copy to avoid modifying original data
            df = df.copy()

            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)

            features_dict = {}

            # Safe rolling calculation function
            def safe_rolling_calc(series, window, func):
                try:
                    result = func(series.rolling(window=window))
                    return result.fillna(method='ffill').fillna(0)  # Forward fill then fill remaining with 0
                except Exception as e:
                    logger.warning(f"Error in rolling calculation: {str(e)}")
                    return pd.Series(0, index=series.index)

            # Volume metrics with safe calculations
            features_dict['volume_ma5'] = safe_rolling_calc(df['Volume'], 5, lambda x: x.mean())
            features_dict['volume_ma20'] = safe_rolling_calc(df['Volume'], 20, lambda x: x.mean())
            features_dict['volume_std20'] = safe_rolling_calc(df['Volume'], 20, lambda x: x.std())

            # Calculate relative volume safely
            volume_ma20 = features_dict['volume_ma20'].replace(0, np.nan)
            features_dict['rel_volume'] = (df['Volume'] / volume_ma20).fillna(1.0)

            # Price-volume relationship
            features_dict['price_change'] = df['Close'].pct_change().fillna(0)
            features_dict['volume_change'] = df['Volume'].pct_change().fillna(0)

            # Safe correlation calculation
            def safe_rolling_corr(x, y, window):
                try:
                    return x.rolling(window).corr(y).fillna(0)
                except Exception:
                    return pd.Series(0, index=x.index)

            features_dict['price_volume_corr'] = safe_rolling_corr(
                df['Close'].pct_change(),
                df['Volume'].pct_change(),
                5
            )
            
            # Momentum features with bounds
            features_dict['volume_momentum'] = df['Volume'].diff(5).fillna(0)
            features_dict['volume_acceleration'] = df['Volume'].diff().diff().fillna(0)
            features_dict['price_momentum'] = df['Close'].diff(5).fillna(0)

            # Trend features
            features_dict['price_trend'] = safe_rolling_calc(df['Close'], 20, lambda x: x.mean().pct_change())
            features_dict['volume_trend'] = safe_rolling_calc(df['Volume'], 20, lambda x: x.mean().pct_change())

            # Volatility features
            features_dict['price_volatility'] = safe_rolling_calc(df['Close'].pct_change(), 20, lambda x: x.std())
            features_dict['volume_volatility'] = safe_rolling_calc(df['Volume'].pct_change(), 20, lambda x: x.std())

            # Create DataFrame and handle remaining NaN values
            features_df = pd.DataFrame(features_dict)
            features_df = features_df.fillna(0)

            # Store feature names
            self.feature_names = list(features_df.columns)
            
            # Clip extreme values (beyond 5 std from mean)
            for column in features_df.columns:
                mean = features_df[column].mean()
                std = features_df[column].std()
                features_df[column] = features_df[column].clip(
                    lower=mean - 5*std,
                    upper=mean + 5*std
                )


            # Scale features
            scaled_features = self.scaler.fit_transform(features_df)

            logger.info(f"Successfully extracted {len(self.feature_names)} features")
            return scaled_features, self.feature_names

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

    def train(self, df: pd.DataFrame):
        """Train the ML models with SHAP-based feature attribution"""
        try:
            logger.info("Starting MLVolumeAnalyzer training")
            features, feature_names = self._extract_advanced_features(df)
            patterns = self._generate_training_labels(df)

            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                features, patterns, test_size=0.2, random_state=42
            )

            # Train XGBoost classifier
            self.volume_classifier.fit(X_train, y_train)

            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.volume_classifier)

            # Evaluate model
            y_pred = self.volume_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )

            logger.info(f"Model Performance - Accuracy: {accuracy:.2f}, F1: {f1:.2f}")

            # Train anomaly detector and clusterer
            self.anomaly_detector.fit(features)
            self.volume_clusterer.fit(features)

            self.is_trained = True
            logger.info("MLVolumeAnalyzer training completed successfully")

            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'feature_importance': dict(zip(
                    feature_names,
                    self.volume_classifier.feature_importances_
                ))
            }

        except Exception as e:
            logger.error(f"Error during MLVolumeAnalyzer training: {str(e)}")
            raise

    def explain_volume_pattern(self, features: np.ndarray) -> Dict:
        """Generate SHAP-based explanations for volume patterns"""
        if not self.is_trained or self.explainer is None:
            raise ValueError("Model not trained. Call train() first.")

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)

            # Get pattern prediction
            pattern_idx = self.volume_classifier.predict(features)[0]
            pattern_type = self.PATTERN_TYPES[pattern_idx]

            # Get top contributing features
            feature_importance = np.abs(shap_values[0]).mean(axis=0)
            top_features_idx = np.argsort(-feature_importance)[:5]

            # Create explanations with proper feature name indexing
            explanations = {}
            for i, idx in enumerate(top_features_idx):
                if idx < len(self.feature_names):  # Ensure index is valid
                    explanations[f'driver_{i+1}'] = {
                        'feature': self.feature_names[idx],
                        'importance': float(feature_importance[idx]),
                        'impact': 'positive' if shap_values[0][idx] > 0 else 'negative'
                    }

            return {
                'pattern': pattern_type,
                'driver_attribution': explanations
            }

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            return {
                'pattern': 'NEUTRAL',
                'driver_attribution': {}
            }

    def decompose_volume(self, df: pd.DataFrame) -> Dict:
        """Decompose volume into accumulation vs distribution components"""
        try:
            features, _ = self._extract_advanced_features(df)
            predictions = self.volume_classifier.predict(features)
            probabilities = self.volume_classifier.predict_proba(features)

            # Calculate accumulation vs distribution scores
            accumulation_score = np.mean([
                prob[0] for pred, prob in zip(predictions, probabilities)
                if pred in [0, 2, 6]  # Accumulation, Breakout, Trend Continuation
            ])

            distribution_score = np.mean([
                prob[0] for pred, prob in zip(predictions, probabilities)
                if pred in [1, 3, 7]  # Distribution, Exhaustion, Reversal
            ])

            return {
                'accumulation_score': float(accumulation_score),
                'distribution_score': float(distribution_score),
                'dominant_flow': 'ACCUMULATION' if accumulation_score > distribution_score else 'DISTRIBUTION',
                'flow_strength': float(abs(accumulation_score - distribution_score))
            }

        except Exception as e:
            logger.error(f"Error decomposing volume: {str(e)}")
            raise

    def _generate_training_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate enhanced training labels for pattern classification"""
        try:
            labels = np.zeros(len(df))

            # Calculate required metrics
            volume_ma20 = df['Volume'].rolling(window=20).mean()
            price_ma20 = df['Close'].rolling(window=20).mean()
            volume_std = df['Volume'].rolling(window=20).std()

            # Accumulation pattern
            labels[
                (df['Volume'] > 1.5 * volume_ma20) & 
                (df['Close'] > df['Close'].shift(1)) &
                (df['Close'] > price_ma20)
            ] = 0

            # Distribution pattern
            labels[
                (df['Volume'] > 1.5 * volume_ma20) & 
                (df['Close'] < df['Close'].shift(1)) &
                (df['Close'] < price_ma20)
            ] = 1

            # Breakout pattern
            labels[
                (df['Volume'] > 2.0 * volume_ma20) & 
                (abs(df['Close'] - df['Close'].shift(1)) > 2 * df['Close'].rolling(20).std())
            ] = 2

            # Exhaustion pattern
            labels[
                (df['Volume'] > 3.0 * volume_ma20) & 
                (df['Close'].rolling(5).std() > 2 * df['Close'].rolling(20).std())
            ] = 3

            # Climax pattern
            labels[
                (df['Volume'] > 4.0 * volume_ma20) & 
                (abs(df['Close'] - df['Close'].shift(1)) > 3 * df['Close'].rolling(20).std())
            ] = 4

            # Additional patterns...
            return labels

        except Exception as e:
            logger.error(f"Error generating training labels: {str(e)}")
            raise

    def detect_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect volume patterns using multiple ML techniques"""
        if not self.is_trained:
            self.train(df)

        try:
            # Extract features
            features, self.feature_names = self._extract_advanced_features(df)

            # Get pattern predictions with probabilities
            pattern_probs = self.volume_classifier.predict_proba(features)
            patterns = self._interpret_patterns(pattern_probs)

            # Detect anomalies
            anomalies = self.anomaly_detector.predict(features)
            anomaly_scores = self.anomaly_detector.score_samples(features)

            # Cluster volume profiles
            cluster_labels = self.volume_clusterer.predict(features)

            # Analyze recent pattern
            recent_pattern = self._analyze_recent_pattern(features[-20:], pattern_probs[-20:])

            return {
                'timestamp': df.index[-1],
                'patterns': patterns,
                'anomalies': anomalies.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'recent_pattern': recent_pattern,
                'volume_profile': self._get_volume_profile(df, cluster_labels[-1])
            }

        except Exception as e:
            logger.error(f"Error detecting volume patterns: {str(e)}")
            raise

    def _interpret_patterns(self, pattern_probs: np.ndarray) -> List[Dict]:
        """Interpret pattern probabilities into meaningful insights"""
        try:
            patterns = []
            for i, probs in enumerate(pattern_probs[-5:]):  # Last 5 patterns
                pattern_type = self.PATTERN_TYPES[np.argmax(probs)]
                confidence = float(np.max(probs))

                if confidence > 0.6:  # Only include high confidence patterns
                    pattern_metrics = self._calculate_pattern_metrics(pattern_type, confidence)
                    patterns.append({
                        'pattern': pattern_type,
                        'confidence': confidence,
                        'strength': pattern_metrics['strength'],
                        'risk_level': pattern_metrics['risk_level'],
                        'suggested_action': pattern_metrics['suggested_action']
                    })

            return patterns

        except Exception as e:
            logger.error(f"Error interpreting patterns: {str(e)}")
            raise

    def _calculate_pattern_metrics(self, pattern_type: str, confidence: float) -> Dict:
        """Calculate comprehensive pattern metrics"""
        try:
            base_strength = confidence

            # Risk levels based on pattern type
            risk_levels = {
                'BREAKOUT': 'HIGH',
                'CLIMAX': 'VERY_HIGH',
                'EXHAUSTION': 'HIGH',
                'ACCUMULATION': 'MEDIUM',
                'DISTRIBUTION': 'MEDIUM',
                'CONSOLIDATION': 'LOW',
                'TREND_CONTINUATION': 'MEDIUM',
                'REVERSAL': 'HIGH'
            }

            # Suggested actions based on pattern type
            actions = {
                'BREAKOUT': 'Consider entry with tight stop',
                'CLIMAX': 'Prepare for potential reversal',
                'EXHAUSTION': 'Consider taking profits',
                'ACCUMULATION': 'Watch for upside breakout',
                'DISTRIBUTION': 'Watch for downside break',
                'CONSOLIDATION': 'Monitor for breakout direction',
                'TREND_CONTINUATION': 'Hold existing positions',
                'REVERSAL': 'Consider position adjustment'
            }

            return {
                'strength': min(1.0, base_strength * 1.2),
                'risk_level': risk_levels.get(pattern_type, 'MEDIUM'),
                'suggested_action': actions.get(pattern_type, 'Monitor price action')
            }

        except Exception as e:
            logger.error(f"Error calculating pattern metrics: {str(e)}")
            raise

    def _get_volume_profile(self, df: pd.DataFrame, cluster_label: int) -> str:
        """Get enhanced volume profile description"""
        try:
            profiles = {
                0: 'LOW_VOLUME_CONSOLIDATION',
                1: 'NORMAL_TRADING_ACTIVITY',
                2: 'HIGH_VOLUME_BREAKOUT',
                3: 'INSTITUTIONAL_ACCUMULATION',
                4: 'DISTRIBUTION_PATTERN'
            }
            return profiles.get(cluster_label, 'UNKNOWN')

        except Exception as e:
            logger.error(f"Error getting volume profile: {str(e)}")
            raise

    def _analyze_recent_pattern(self, features: np.ndarray, pattern_probs: np.ndarray) -> Dict:
        """Analyze recent volume pattern with enhanced metrics"""
        try:
            # Get SHAP explanations for the pattern
            pattern_explanation = self.explain_volume_pattern(features[-1:])

            # Calculate trend strength and momentum
            recent_patterns = [np.argmax(probs) for probs in pattern_probs]

            # Determine pattern type and confidence
            pattern_type = self._determine_pattern_type(recent_patterns)
            pattern_confidence = float(np.max(pattern_probs[-1]))
            
            return {
                'type': pattern_type,
                'confidence': pattern_confidence,
                'strength': self._calculate_pattern_strength(pattern_probs),
                'drivers': pattern_explanation['driver_attribution'],
                'momentum_quality': self._assess_momentum_quality(features[-20:])
            }

        except Exception as e:
            logger.error(f"Error analyzing recent pattern: {str(e)}")
            raise

    def _determine_pattern_type(self, recent_patterns: List[int]) -> str:
        """Determine the most dominant pattern type"""
        try:
            pattern_counts = {}
            for pattern in recent_patterns:
                pattern_name = self.PATTERN_TYPES.get(pattern, 'UNKNOWN')
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

            if not pattern_counts:
               return "NO_PATTERN_DETECTED"

            return max(pattern_counts, key=pattern_counts.get)

        except Exception as e:
             logger.error(f"Error determining pattern type {str(e)}")
             raise


    def _calculate_pattern_strength(self, pattern_probs: np.ndarray) -> float:
        """Calculate composite strength of the recent patterns"""
        try:
            # Use average probability of detected patterns as pattern strength.
            avg_prob = np.mean(np.max(pattern_probs, axis=1)[-5:]) if len(pattern_probs) > 0 else 0.0
            return float(avg_prob)

        except Exception as e:
            logger.error(f"Error calculating pattern strength {str(e)}")
            raise


    def _assess_momentum_quality(self, recent_features: np.ndarray) -> float:
        """Assess the quality of momentum based on recent features"""
        try:
            # Calculate momentum quality score (0 to 1)
            volume_trend = np.mean(recent_features[:, 11])  # volume_trend feature
            price_volume_corr = np.mean(recent_features[:, 6])  # price_volume_corr feature

            momentum_score = (
                0.4 * (volume_trend + 1) / 2 +  # Normalize to [0,1]
                0.3 * (price_volume_corr + 1) / 2 +  # Normalize to [0,1]
                0.3 * np.mean(recent_features[:, 4] > 0)  # Ratio of positive price changes
            )

            return float(np.clip(momentum_score, 0, 1))

        except Exception as e:
            logger.error(f"Error assessing momentum quality: {str(e)}")
            raise