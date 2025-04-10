"""
Neural Network Analyzer for stock market prediction.

This module provides deep learning models for analyzing stock market data.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class NeuralNetworkAnalyzer:
    """Neural network models for stock market analysis."""
    
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Available models
        self.available_models = {
            'price_prediction': self._price_prediction_model,
            'trend_analysis': self._trend_analysis_model,
            'volatility_forecast': self._volatility_forecast_model,
            'volume_analysis': self._volume_analysis_model,
            'sentiment_analysis': self._sentiment_analysis_model
        }
        
        # Mock model parameters (in a real implementation, these would be loaded from saved models)
        self._load_model_parameters()
    
    def _load_model_parameters(self):
        """Load model parameters from files or use defaults."""
        # In a real implementation, this would load trained model weights
        # For now, we'll use mock parameters
        
        self.model_params = {
            'price_prediction': {
                'window_size': 20,
                'prediction_horizon': 5,
                'features': ['Close', 'Volume', 'High', 'Low']
            },
            'trend_analysis': {
                'short_window': 10,
                'long_window': 50,
                'threshold': 0.02
            },
            'volatility_forecast': {
                'window_size': 30,
                'confidence_level': 0.95
            },
            'volume_analysis': {
                'window_size': 15,
                'volume_threshold': 1.5
            },
            'sentiment_analysis': {
                'window_size': 10,
                'sentiment_features': ['Close_change', 'Volume_change']
            }
        }
    
    def analyze_ticker(self, ticker: str, data: pd.DataFrame, models: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a ticker using specified models.
        
        Args:
            ticker: The ticker symbol
            data: Historical market data for the ticker
            models: List of models to use (default: all available models)
            
        Returns:
            Dictionary with analysis results
        """
        if data.empty:
            return {'error': f"No data available for {ticker}"}
        
        # Use all models if none specified
        if not models:
            models = list(self.available_models.keys())
        
        # Filter to only include available models
        models = [m for m in models if m in self.available_models]
        
        if not models:
            return {'error': "No valid models specified"}
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Run each model
        results = {'ticker': ticker}
        for model_name in models:
            model_func = self.available_models[model_name]
            model_result = model_func(processed_data)
            results[model_name] = model_result
        
        return results
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for model input."""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index)
        
        # Sort by date (oldest first)
        df = df.sort_index()
        
        # Calculate additional features
        if 'Close' in df.columns:
            # Daily returns
            df['Returns'] = df['Close'].pct_change()
            
            # Log returns
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Volatility (20-day rolling standard deviation of returns)
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Moving averages
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        if 'Volume' in df.columns:
            # Volume changes
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Relative volume (compared to 20-day average)
            df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def _price_prediction_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Price prediction model."""
        # In a real implementation, this would use a trained neural network
        # For now, we'll use a simple mock implementation
        
        params = self.model_params['price_prediction']
        window_size = params['window_size']
        horizon = params['prediction_horizon']
        
        if len(data) < window_size + horizon:
            return {'error': "Not enough data for price prediction"}
        
        # Get the latest closing prices
        latest_prices = data['Close'].values[-window_size:]
        
        # Calculate a simple prediction (in a real model, this would be a neural network prediction)
        # Here we're just using a weighted average of recent prices with some random noise
        weights = np.linspace(0.5, 1.0, window_size)
        weights = weights / weights.sum()
        
        base_prediction = np.sum(latest_prices * weights)
        current_price = latest_prices[-1]
        
        # Add some random variation to simulate model uncertainty
        predictions = []
        for i in range(horizon):
            # Each day's prediction builds on the previous
            if i == 0:
                day_pred = base_prediction * (1 + np.random.normal(0.001, 0.01))
            else:
                day_pred = predictions[-1] * (1 + np.random.normal(0.001, 0.01))
            predictions.append(day_pred)
        
        # Calculate prediction intervals
        lower_bound = [pred * 0.95 for pred in predictions]
        upper_bound = [pred * 1.05 for pred in predictions]
        
        # Calculate expected return
        expected_return = (predictions[-1] / current_price - 1) * 100
        
        return {
            'current_price': current_price,
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'horizon_days': horizon,
            'expected_return': expected_return,
            'confidence': 0.85  # Mock confidence score
        }
    
    def _trend_analysis_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Trend analysis model."""
        params = self.model_params['trend_analysis']
        short_window = params['short_window']
        long_window = params['long_window']
        threshold = params['threshold']
        
        if len(data) < long_window:
            return {'error': "Not enough data for trend analysis"}
        
        # Calculate short and long-term trends
        short_trend = data['Close'].rolling(window=short_window).mean().iloc[-1]
        long_trend = data['Close'].rolling(window=long_window).mean().iloc[-1]
        
        # Determine trend direction
        if short_trend > long_trend * (1 + threshold):
            trend = "Strong Uptrend"
            strength = (short_trend / long_trend - 1) * 10
        elif short_trend > long_trend:
            trend = "Uptrend"
            strength = (short_trend / long_trend - 1) * 10
        elif short_trend < long_trend * (1 - threshold):
            trend = "Strong Downtrend"
            strength = (1 - short_trend / long_trend) * 10
        elif short_trend < long_trend:
            trend = "Downtrend"
            strength = (1 - short_trend / long_trend) * 10
        else:
            trend = "Sideways"
            strength = 0
        
        # Calculate momentum
        momentum = data['Close'].diff(5).iloc[-1] / data['Close'].iloc[-6] * 100
        
        return {
            'trend': trend,
            'strength': min(strength, 10),  # Cap at 10
            'momentum': momentum,
            'short_ma': short_trend,
            'long_ma': long_trend
        }
    
    def _volatility_forecast_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Volatility forecasting model."""
        params = self.model_params['volatility_forecast']
        window_size = params['window_size']
        confidence_level = params['confidence_level']
        
        if len(data) < window_size:
            return {'error': "Not enough data for volatility forecast"}
        
        # Calculate historical volatility
        if 'Volatility' in data.columns:
            hist_volatility = data['Volatility'].iloc[-1]
        else:
            returns = data['Close'].pct_change().dropna()
            hist_volatility = returns.std()
        
        # Forecast future volatility (in a real model, this would use GARCH or similar)
        # Here we're using a simple approach with some random variation
        forecast_volatility = hist_volatility * (1 + np.random.normal(0, 0.1))
        
        # Calculate Value at Risk (VaR)
        current_price = data['Close'].iloc[-1]
        z_score = 1.96  # Approximately 95% confidence level
        var_1day = current_price * forecast_volatility * z_score
        
        return {
            'historical_volatility': hist_volatility,
            'forecast_volatility': forecast_volatility,
            'var_1day': var_1day,
            'var_1day_percent': (var_1day / current_price) * 100,
            'confidence_level': confidence_level
        }
    
    def _volume_analysis_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Volume analysis model."""
        params = self.model_params['volume_analysis']
        window_size = params['window_size']
        volume_threshold = params['volume_threshold']
        
        if len(data) < window_size or 'Volume' not in data.columns:
            return {'error': "Not enough data for volume analysis"}
        
        # Calculate average volume
        avg_volume = data['Volume'].rolling(window=window_size).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        
        # Determine if current volume is unusual
        relative_volume = current_volume / avg_volume
        
        if relative_volume > volume_threshold:
            volume_signal = "Unusually High"
        elif relative_volume < 1/volume_threshold:
            volume_signal = "Unusually Low"
        else:
            volume_signal = "Normal"
        
        # Check for volume confirmation of price movement
        price_change = data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1
        
        if price_change > 0 and relative_volume > 1:
            confirmation = "Confirmed Upward Movement"
        elif price_change < 0 and relative_volume > 1:
            confirmation = "Confirmed Downward Movement"
        elif price_change > 0 and relative_volume < 1:
            confirmation = "Unconfirmed Upward Movement"
        elif price_change < 0 and relative_volume < 1:
            confirmation = "Unconfirmed Downward Movement"
        else:
            confirmation = "Neutral"
        
        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'relative_volume': relative_volume,
            'volume_signal': volume_signal,
            'price_volume_confirmation': confirmation
        }
    
    def _sentiment_analysis_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Sentiment analysis model based on price action."""
        params = self.model_params['sentiment_analysis']
        window_size = params['window_size']
        
        if len(data) < window_size:
            return {'error': "Not enough data for sentiment analysis"}
        
        # Calculate price momentum
        price_momentum = data['Close'].diff(5).iloc[-1] / data['Close'].iloc[-6]
        
        # Calculate volume trend
        if 'Volume' in data.columns:
            volume_trend = data['Volume'].iloc[-1] / data['Volume'].rolling(window=window_size).mean().iloc[-1]
        else:
            volume_trend = 1.0
        
        # Calculate RSI if available
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
        else:
            rsi = 50  # Neutral default
        
        # Combine factors to determine sentiment
        sentiment_score = 0
        
        # Price momentum contribution
        sentiment_score += price_momentum * 50
        
        # RSI contribution
        if rsi > 70:
            sentiment_score += 20  # Overbought
        elif rsi < 30:
            sentiment_score -= 20  # Oversold
        else:
            sentiment_score += (rsi - 50) / 50 * 20  # Scaled contribution
        
        # Volume contribution
        if volume_trend > 1.5 and price_momentum > 0:
            sentiment_score += 15  # High volume upward movement
        elif volume_trend > 1.5 and price_momentum < 0:
            sentiment_score -= 15  # High volume downward movement
        
        # Normalize to -100 to 100 range
        sentiment_score = max(min(sentiment_score, 100), -100)
        
        # Determine sentiment category
        if sentiment_score > 70:
            sentiment = "Very Bullish"
        elif sentiment_score > 30:
            sentiment = "Bullish"
        elif sentiment_score > -30:
            sentiment = "Neutral"
        elif sentiment_score > -70:
            sentiment = "Bearish"
        else:
            sentiment = "Very Bearish"
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'price_momentum': price_momentum,
            'volume_trend': volume_trend,
            'rsi': rsi
        }
