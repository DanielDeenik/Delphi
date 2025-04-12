"""
Volume analysis strategies.

This module provides strategies for volume analysis.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from delphi.core.models.base import AnalysisStrategy

# Configure logger
logger = logging.getLogger(__name__)

class SimpleVolumeStrategy(AnalysisStrategy):
    """Simple strategy for volume analysis."""
    
    def __init__(self, z_score_threshold: float = 2.0, lookback_period: int = 20, **kwargs):
        """Initialize the simple volume strategy.
        
        Args:
            z_score_threshold: Z-score threshold for volume spikes
            lookback_period: Lookback period for calculating volume statistics
            **kwargs: Additional arguments
        """
        self.z_score_threshold = z_score_threshold
        self.lookback_period = lookback_period
        
        super().__init__(**kwargs)
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze volume patterns in stock data.
        
        Args:
            data: DataFrame with stock data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            df = data.copy()
            
            # Ensure data is sorted by date (oldest first)
            df = df.sort_values('date')
            
            # Calculate volume moving average
            df[f'volume_ma{self.lookback_period}'] = df['volume'].rolling(window=self.lookback_period).mean()
            
            # Calculate volume standard deviation
            df[f'volume_std{self.lookback_period}'] = df['volume'].rolling(window=self.lookback_period).std()
            
            # Calculate volume z-score
            df['volume_z_score'] = (df['volume'] - df[f'volume_ma{self.lookback_period}']) / df[f'volume_std{self.lookback_period}']
            
            # Detect volume spikes
            df['is_volume_spike'] = df['volume_z_score'] > self.z_score_threshold
            
            # Calculate spike strength (how many standard deviations above the mean)
            df['spike_strength'] = df['volume_z_score'].where(df['is_volume_spike'], 0)
            
            # Calculate price change percentage
            df['price_change_pct'] = df['close'].pct_change() * 100
            
            # Generate signals
            df['signal'] = 'NEUTRAL'
            
            # Bullish signal: volume spike with positive price change
            bullish_mask = (df['is_volume_spike']) & (df['price_change_pct'] > 0)
            df.loc[bullish_mask, 'signal'] = 'BULLISH'
            
            # Bearish signal: volume spike with negative price change
            bearish_mask = (df['is_volume_spike']) & (df['price_change_pct'] < 0)
            df.loc[bearish_mask, 'signal'] = 'BEARISH'
            
            # Calculate signal strength (confidence)
            df['signal_strength'] = df['spike_strength'] * abs(df['price_change_pct']) / 100
            
            # Add notes
            df['notes'] = ''
            df.loc[bullish_mask, 'notes'] = 'Bullish volume spike detected'
            df.loc[bearish_mask, 'notes'] = 'Bearish volume spike detected'
            
            # Calculate relative volume
            df['relative_volume_20d'] = df['volume'] / df[f'volume_ma{self.lookback_period}']
            
            # Prepare results
            results = {
                'analysis_df': df,
                'volume_spikes': df[df['is_volume_spike']].to_dict(orient='records'),
                'summary': {
                    'total_spikes': df['is_volume_spike'].sum(),
                    'bullish_spikes': (bullish_mask).sum(),
                    'bearish_spikes': (bearish_mask).sum(),
                    'avg_spike_strength': df.loc[df['is_volume_spike'], 'spike_strength'].mean() if df['is_volume_spike'].any() else 0,
                    'max_spike_strength': df['spike_strength'].max(),
                    'latest_signal': df.iloc[-1]['signal'] if not df.empty else 'NEUTRAL',
                    'latest_signal_strength': df.iloc[-1]['signal_strength'] if not df.empty else 0,
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in simple volume analysis: {str(e)}")
            return {"error": str(e)}


class MLVolumeStrategy(AnalysisStrategy):
    """ML-based strategy for volume analysis."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize the ML-based volume strategy.
        
        Args:
            model_path: Path to the trained model (optional)
            **kwargs: Additional arguments
        """
        self.model_path = model_path
        self.model = None
        
        # Load model if path is provided
        if model_path:
            self._load_model()
        
        super().__init__(**kwargs)
    
    def _load_model(self):
        """Load the trained model."""
        try:
            import joblib
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            DataFrame with features
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            df = data.copy()
            
            # Ensure data is sorted by date (oldest first)
            df = df.sort_values('date')
            
            # Calculate volume moving averages
            for period in [5, 10, 20, 50]:
                df[f'volume_ma{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma{period}']
            
            # Calculate price moving averages
            for period in [5, 10, 20, 50]:
                df[f'close_ma{period}'] = df['close'].rolling(window=period).mean()
                df[f'close_ratio_{period}'] = df['close'] / df[f'close_ma{period}']
            
            # Calculate returns
            for period in [1, 5, 10]:
                df[f'return_{period}d'] = df['close'].pct_change(period)
            
            # Calculate volume changes
            for period in [1, 5, 10]:
                df[f'volume_change_{period}d'] = df['volume'].pct_change(period)
            
            # Calculate volume standard deviation
            for period in [10, 20, 50]:
                df[f'volume_std{period}'] = df['volume'].rolling(window=period).std()
                df[f'volume_z_score_{period}'] = (df['volume'] - df[f'volume_ma{period}']) / df[f'volume_std{period}']
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze volume patterns in stock data using ML.
        
        Args:
            data: DataFrame with stock data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Check if model is loaded
            if self.model is None:
                logger.warning("Model not loaded")
                return {"error": "Model not loaded"}
            
            # Prepare features
            features_df = self._prepare_features(data)
            
            if features_df.empty:
                logger.warning("Failed to prepare features")
                return {"error": "Failed to prepare features"}
            
            # Select feature columns
            feature_cols = [col for col in features_df.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
            
            # Make predictions
            predictions = self.model.predict(features_df[feature_cols])
            probabilities = self.model.predict_proba(features_df[feature_cols])
            
            # Add predictions to DataFrame
            features_df['prediction'] = predictions
            features_df['probability'] = probabilities.max(axis=1)
            
            # Map predictions to signals
            signal_map = {0: 'NEUTRAL', 1: 'BULLISH', 2: 'BEARISH'}
            features_df['signal'] = features_df['prediction'].map(signal_map)
            
            # Calculate signal strength (confidence)
            features_df['signal_strength'] = features_df['probability']
            
            # Add notes
            features_df['notes'] = 'ML-based signal'
            
            # Prepare results
            results = {
                'analysis_df': features_df,
                'signals': features_df[features_df['signal'] != 'NEUTRAL'].to_dict(orient='records'),
                'summary': {
                    'total_signals': (features_df['signal'] != 'NEUTRAL').sum(),
                    'bullish_signals': (features_df['signal'] == 'BULLISH').sum(),
                    'bearish_signals': (features_df['signal'] == 'BEARISH').sum(),
                    'avg_signal_strength': features_df.loc[features_df['signal'] != 'NEUTRAL', 'signal_strength'].mean() if (features_df['signal'] != 'NEUTRAL').any() else 0,
                    'latest_signal': features_df.iloc[-1]['signal'] if not features_df.empty else 'NEUTRAL',
                    'latest_signal_strength': features_df.iloc[-1]['signal_strength'] if not features_df.empty else 0,
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ML volume analysis: {str(e)}")
            return {"error": str(e)}
