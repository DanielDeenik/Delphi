import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

class VolumeShiftAnalyzer:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.shap_explainer = None
        
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for volume analysis."""
        df = data.copy()
        
        # Calculate relative volume (RVOL)
        df['RVOL'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price momentum features
        df['price_change'] = df['Close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(5).std()
        
        # Volume momentum
        df['volume_ma5'] = df['Volume'].rolling(5).mean()
        df['volume_ma20'] = df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['volume_ma5'] / df['volume_ma20']
        
        # Price-volume correlation
        df['price_volume_corr'] = (
            df['price_change'] * 
            (df['Volume'] - df['Volume'].rolling(5).mean()) / 
            df['Volume'].rolling(5).std()
        )
        
        return df.dropna()
    
    def train_model(self, data: pd.DataFrame, target_col: str = 'RVOL') -> None:
        """Train the XGBoost model for volume shift attribution."""
        features_df = self._engineer_features(data)
        
        feature_cols = [
            'price_change', 'price_volatility', 
            'volume_trend', 'price_volume_corr'
        ]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Initialize SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
    
    def analyze_volume_shift(self, data: pd.DataFrame) -> Dict:
        """Analyze volume shifts and attribute causes."""
        features_df = self._engineer_features(data)
        
        feature_cols = [
            'price_change', 'price_volatility', 
            'volume_trend', 'price_volume_corr'
        ]
        
        X = features_df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate SHAP values for the latest data point
        shap_values = self.shap_explainer.shap_values(X_scaled[-1:])
        
        # Get feature importance
        feature_importance = dict(zip(feature_cols, np.abs(shap_values[0])))
        
        # Classify the volume pattern
        volume_classification = self._classify_volume_pattern(
            features_df.iloc[-1], predictions[-1]
        )
        
        return {
            'predicted_rvol': float(predictions[-1]),
            'feature_importance': feature_importance,
            'classification': volume_classification,
            'latest_metrics': {
                'rvol': float(features_df['RVOL'].iloc[-1]),
                'volume_trend': float(features_df['volume_trend'].iloc[-1]),
                'price_volume_correlation': float(features_df['price_volume_corr'].iloc[-1])
            }
        }
    
    def _classify_volume_pattern(self, latest_data: pd.Series, predicted_rvol: float) -> str:
        """Classify the volume pattern based on latest data and predictions."""
        if predicted_rvol > 2.0:  # Significant volume increase
            if latest_data['price_change'] > 0:
                if latest_data['price_volume_corr'] > 0.5:
                    return 'BULLISH_ACCUMULATION'
                return 'POTENTIAL_BREAKOUT'
            else:
                if latest_data['price_volume_corr'] < -0.5:
                    return 'BEARISH_DISTRIBUTION'
                return 'POTENTIAL_REVERSAL'
        elif predicted_rvol < 0.5:  # Significant volume decrease
            if latest_data['volume_trend'] < 0.8:
                return 'VOLUME_EXHAUSTION'
            return 'CONSOLIDATION'
        else:
            return 'NEUTRAL'
    
    def get_explanatory_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate human-readable insights from the analysis."""
        analysis = self.analyze_volume_shift(data)
        insights = []
        
        # Volume level insight
        if analysis['predicted_rvol'] > 2.0:
            insights.append(f"Unusual volume detected: {analysis['predicted_rvol']:.1f}x normal levels")
        
        # Pattern classification insight
        pattern_messages = {
            'BULLISH_ACCUMULATION': 'Showing signs of institutional buying with rising prices',
            'BEARISH_DISTRIBUTION': 'Indicates potential distribution with falling prices',
            'POTENTIAL_BREAKOUT': 'Volume surge suggests possible breakout attempt',
            'POTENTIAL_REVERSAL': 'Heavy volume with price weakness signals possible reversal',
            'VOLUME_EXHAUSTION': 'Volume declining, suggesting trend exhaustion',
            'CONSOLIDATION': 'Normal volume levels indicate consolidation',
            'NEUTRAL': 'No significant volume patterns detected'
        }
        insights.append(pattern_messages[analysis['classification']])
        
        # Add major contributing factors
        sorted_factors = sorted(
            analysis['feature_importance'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_factor = sorted_factors[0][0].replace('_', ' ').title()
        insights.append(f"Primary driver: {top_factor}")
        
        return insights
