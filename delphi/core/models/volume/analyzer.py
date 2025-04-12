"""
Volume analyzer for stock data.

This module provides a volume analyzer for stock data.
"""
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from delphi.core.models.base import Analyzer, AnalysisStrategy

# Configure logger
logger = logging.getLogger(__name__)

class VolumeAnalyzer(Analyzer):
    """Analyzer for volume patterns in stock data."""
    
    def __init__(self, strategy: Optional[AnalysisStrategy] = None, cache_size: int = 128, **kwargs):
        """Initialize the volume analyzer.
        
        Args:
            strategy: Analysis strategy to use (default: SimpleVolumeStrategy)
            cache_size: Size of the LRU cache for analyze methods
            **kwargs: Additional arguments
        """
        # Set default strategy if not provided
        if strategy is None:
            from delphi.core.models.volume.strategies import SimpleVolumeStrategy
            strategy = SimpleVolumeStrategy()
        
        self.strategy = strategy
        
        super().__init__(cache_size=cache_size, **kwargs)
    
    def _analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze volume patterns in stock data.
        
        Args:
            data: DataFrame with stock data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Validate data
            if data.empty:
                logger.warning("DataFrame is empty")
                return {"error": "DataFrame is empty"}
            
            # Check for required columns
            required_columns = ['date', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                return {"error": f"Missing required columns: {missing_columns}"}
            
            # Use strategy to analyze data
            results = self.strategy.analyze(data, **kwargs)
            
            # Add timestamp
            results['timestamp'] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return {"error": str(e)}
    
    def detect_volume_spikes(self, data: pd.DataFrame, z_score_threshold: float = 2.0, 
                            lookback_period: int = 20) -> pd.DataFrame:
        """Detect volume spikes in stock data.
        
        Args:
            data: DataFrame with stock data
            z_score_threshold: Z-score threshold for volume spikes
            lookback_period: Lookback period for calculating volume statistics
            
        Returns:
            DataFrame with volume spike detection results
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            df = data.copy()
            
            # Ensure data is sorted by date (oldest first)
            df = df.sort_values('date')
            
            # Calculate volume moving average
            df[f'volume_ma{lookback_period}'] = df['volume'].rolling(window=lookback_period).mean()
            
            # Calculate volume standard deviation
            df[f'volume_std{lookback_period}'] = df['volume'].rolling(window=lookback_period).std()
            
            # Calculate volume z-score
            df['volume_z_score'] = (df['volume'] - df[f'volume_ma{lookback_period}']) / df[f'volume_std{lookback_period}']
            
            # Detect volume spikes
            df['is_volume_spike'] = df['volume_z_score'] > z_score_threshold
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting volume spikes: {str(e)}")
            return pd.DataFrame()
    
    def calculate_relative_volume(self, data: pd.DataFrame, periods: List[int] = [5, 20, 50]) -> pd.DataFrame:
        """Calculate relative volume metrics.
        
        Args:
            data: DataFrame with stock data
            periods: List of periods for calculating moving averages
            
        Returns:
            DataFrame with relative volume metrics
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            df = data.copy()
            
            # Ensure data is sorted by date (oldest first)
            df = df.sort_values('date')
            
            # Calculate volume moving averages
            for period in periods:
                df[f'volume_ma{period}'] = df['volume'].rolling(window=period).mean()
                
                # Calculate relative volume (current volume / moving average)
                df[f'relative_volume_{period}d'] = df['volume'] / df[f'volume_ma{period}']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating relative volume: {str(e)}")
            return pd.DataFrame()
    
    def analyze_volume_price_divergence(self, data: pd.DataFrame, lookback_period: int = 20) -> pd.DataFrame:
        """Analyze volume-price divergence.
        
        Args:
            data: DataFrame with stock data
            lookback_period: Lookback period for calculating correlations
            
        Returns:
            DataFrame with volume-price divergence analysis
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            df = data.copy()
            
            # Ensure data is sorted by date (oldest first)
            df = df.sort_values('date')
            
            # Calculate price change
            df['price_change'] = df['close'].diff()
            
            # Calculate volume change
            df['volume_change'] = df['volume'].diff()
            
            # Calculate rolling correlation between price change and volume change
            df['price_volume_correlation'] = df['price_change'].rolling(window=lookback_period).corr(df['volume_change'])
            
            # Detect divergence
            df['is_divergence'] = False
            
            # Positive price change with decreasing volume
            positive_divergence = (df['price_change'] > 0) & (df['volume_change'] < 0)
            df.loc[positive_divergence, 'is_divergence'] = True
            df.loc[positive_divergence, 'divergence_type'] = 'POSITIVE'
            
            # Negative price change with increasing volume
            negative_divergence = (df['price_change'] < 0) & (df['volume_change'] > 0)
            df.loc[negative_divergence, 'is_divergence'] = True
            df.loc[negative_divergence, 'divergence_type'] = 'NEGATIVE'
            
            return df
            
        except Exception as e:
            logger.error(f"Error analyzing volume-price divergence: {str(e)}")
            return pd.DataFrame()
