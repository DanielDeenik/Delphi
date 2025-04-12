"""
Volume footprint analysis module for the Volume Intelligence Trading System.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import numba
from numba import jit

from trading_ai.config import config_manager

# Configure logging
logger = logging.getLogger(__name__)

class VolumeAnalyzer:
    """Analyzer for volume patterns and inefficiencies."""
    
    def __init__(self, ma_periods: Optional[List[int]] = None):
        """Initialize the volume analyzer.
        
        Args:
            ma_periods: List of periods for moving averages
        """
        self.ma_periods = ma_periods or config_manager.system_config.volume_ma_periods
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume metrics for a DataFrame.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with added volume metrics
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame, skipping volume metrics calculation")
                return df
            
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Sort by date (oldest first) if not already sorted
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df = result_df.sort_index()
            elif 'date' in result_df.columns:
                result_df = result_df.sort_values('date')
            
            # Calculate volume moving averages
            for period in self.ma_periods:
                result_df[f'volume_ma{period}'] = result_df['volume'].rolling(window=period).mean()
            
            # Calculate relative volume (ratio to moving averages)
            for period in self.ma_periods:
                result_df[f'relative_volume_{period}d'] = result_df['volume'] / result_df[f'volume_ma{period}']
            
            # Calculate volume Z-score (using 20-day MA and STD)
            result_df['volume_std20'] = result_df['volume'].rolling(window=20).std()
            result_df['volume_z_score'] = (result_df['volume'] - result_df['volume_ma20']) / result_df['volume_std20']
            
            # Identify volume spikes (Z-score > 2.0)
            result_df['is_volume_spike'] = result_df['volume_z_score'] > 2.0
            
            # Calculate spike strength (relative to 20-day MA)
            result_df['spike_strength'] = np.where(
                result_df['is_volume_spike'],
                result_df['volume'] / result_df['volume_ma20'],
                0
            )
            
            # Calculate price change percentage
            result_df['price_change_pct'] = result_df['close'].pct_change() * 100
            
            logger.info("Successfully calculated volume metrics")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating volume metrics: {str(e)}")
            return df
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_volume_profile(prices: np.ndarray, volumes: np.ndarray, num_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate volume profile (volume at price).
        
        Args:
            prices: Array of prices
            volumes: Array of volumes
            num_bins: Number of price bins
            
        Returns:
            Tuple of (price_levels, volume_at_price)
        """
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        # Create price bins
        bin_size = (max_price - min_price) / num_bins
        price_levels = np.linspace(min_price, max_price, num_bins)
        volume_at_price = np.zeros(num_bins)
        
        # Distribute volume across price bins
        for i in range(len(prices)):
            price = prices[i]
            volume = volumes[i]
            
            # Find the bin index
            bin_idx = int((price - min_price) / bin_size)
            
            # Ensure index is within bounds
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            
            # Add volume to the bin
            volume_at_price[bin_idx] += volume
        
        return price_levels, volume_at_price
    
    def calculate_volume_profile(self, df: pd.DataFrame, days: int = 20, num_bins: int = 50) -> pd.DataFrame:
        """Calculate volume profile (volume at price) for recent data.
        
        Args:
            df: DataFrame with price and volume data
            days: Number of recent days to include
            num_bins: Number of price bins
            
        Returns:
            DataFrame with price levels and volume at each level
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame, skipping volume profile calculation")
                return pd.DataFrame()
            
            # Make a copy and sort by date (oldest first)
            if isinstance(df.index, pd.DatetimeIndex):
                temp_df = df.sort_index().tail(days)
            elif 'date' in df.columns:
                temp_df = df.sort_values('date').tail(days)
            else:
                temp_df = df.tail(days)
            
            # Extract prices and volumes
            prices = temp_df['close'].values
            volumes = temp_df['volume'].values
            
            # Calculate volume profile
            price_levels, volume_at_price = self._calculate_volume_profile(prices, volumes, num_bins)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                'price_level': price_levels,
                'volume_at_price': volume_at_price
            })
            
            # Calculate percentage of total volume
            total_volume = np.sum(volume_at_price)
            result_df['volume_percentage'] = (result_df['volume_at_price'] / total_volume) * 100
            
            # Identify high volume nodes (> 2% of total volume)
            result_df['is_high_volume_node'] = result_df['volume_percentage'] > 2.0
            
            # Identify low volume nodes (< 0.5% of total volume)
            result_df['is_low_volume_node'] = result_df['volume_percentage'] < 0.5
            
            logger.info(f"Successfully calculated volume profile with {num_bins} price bins")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return pd.DataFrame()
    
    def detect_volume_inefficiencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect volume inefficiencies in price data.
        
        Args:
            df: DataFrame with price, volume, and volume metrics
            
        Returns:
            DataFrame with added inefficiency signals
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame, skipping inefficiency detection")
                return df
            
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Ensure we have the necessary volume metrics
            if 'volume_z_score' not in result_df.columns:
                result_df = self.calculate_volume_metrics(result_df)
            
            # Initialize signal column
            result_df['signal'] = 'NEUTRAL'
            result_df['confidence'] = 0.0
            result_df['notes'] = ''
            
            # Get ticker direction (buy/short)
            ticker = result_df['symbol'].iloc[0] if 'symbol' in result_df.columns else None
            direction = config_manager.get_ticker_direction(ticker) if ticker else None
            
            # Detect inefficiencies
            for i in range(1, len(result_df)):
                # Current and previous row
                curr = result_df.iloc[i]
                prev = result_df.iloc[i-1]
                
                # Skip if we don't have enough data
                if pd.isna(curr['volume_z_score']) or pd.isna(curr['price_change_pct']):
                    continue
                
                # Volume spike with price change
                if curr['is_volume_spike']:
                    # For buy candidates
                    if direction == 'buy':
                        # Volume spike with positive price change
                        if curr['price_change_pct'] > 0 and curr['volume_z_score'] > 2.5:
                            result_df.at[result_df.index[i], 'signal'] = 'STRONG_BUY'
                            result_df.at[result_df.index[i], 'confidence'] = min(curr['volume_z_score'] / 5, 1.0)
                            result_df.at[result_df.index[i], 'notes'] = (
                                f"Volume spike (Z={curr['volume_z_score']:.2f}) with "
                                f"positive price change ({curr['price_change_pct']:.2f}%)"
                            )
                        
                        # Volume spike with negative price change (potential reversal)
                        elif curr['price_change_pct'] < 0 and curr['volume_z_score'] > 3.0:
                            result_df.at[result_df.index[i], 'signal'] = 'POTENTIAL_REVERSAL'
                            result_df.at[result_df.index[i], 'confidence'] = min(curr['volume_z_score'] / 6, 0.8)
                            result_df.at[result_df.index[i], 'notes'] = (
                                f"Volume spike (Z={curr['volume_z_score']:.2f}) with "
                                f"negative price change ({curr['price_change_pct']:.2f}%) - potential reversal"
                            )
                    
                    # For short candidates
                    elif direction == 'short':
                        # Volume spike with negative price change
                        if curr['price_change_pct'] < 0 and curr['volume_z_score'] > 2.5:
                            result_df.at[result_df.index[i], 'signal'] = 'STRONG_SHORT'
                            result_df.at[result_df.index[i], 'confidence'] = min(curr['volume_z_score'] / 5, 1.0)
                            result_df.at[result_df.index[i], 'notes'] = (
                                f"Volume spike (Z={curr['volume_z_score']:.2f}) with "
                                f"negative price change ({curr['price_change_pct']:.2f}%)"
                            )
                        
                        # Volume spike with positive price change (potential reversal)
                        elif curr['price_change_pct'] > 0 and curr['volume_z_score'] > 3.0:
                            result_df.at[result_df.index[i], 'signal'] = 'POTENTIAL_REVERSAL'
                            result_df.at[result_df.index[i], 'confidence'] = min(curr['volume_z_score'] / 6, 0.8)
                            result_df.at[result_df.index[i], 'notes'] = (
                                f"Volume spike (Z={curr['volume_z_score']:.2f}) with "
                                f"positive price change ({curr['price_change_pct']:.2f}%) - potential reversal"
                            )
            
            logger.info("Successfully detected volume inefficiencies")
            return result_df
            
        except Exception as e:
            logger.error(f"Error detecting volume inefficiencies: {str(e)}")
            return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, volume_profile_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels based on volume profile.
        
        Args:
            df: DataFrame with price and volume data
            volume_profile_df: DataFrame with volume profile data
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if df.empty or volume_profile_df.empty:
                logger.warning("Empty DataFrame, skipping support/resistance calculation")
                return {}
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Find high volume nodes
            high_volume_nodes = volume_profile_df[volume_profile_df['is_high_volume_node']]
            
            # Find support levels (high volume nodes below current price)
            support_levels = high_volume_nodes[high_volume_nodes['price_level'] < current_price]
            support_levels = support_levels.sort_values('price_level', ascending=False)
            
            # Find resistance levels (high volume nodes above current price)
            resistance_levels = high_volume_nodes[high_volume_nodes['price_level'] > current_price]
            resistance_levels = resistance_levels.sort_values('price_level')
            
            # Get nearest support and resistance
            nearest_support = support_levels['price_level'].iloc[0] if not support_levels.empty else None
            nearest_resistance = resistance_levels['price_level'].iloc[0] if not resistance_levels.empty else None
            
            # Find low volume nodes (potential breakout levels)
            low_volume_nodes = volume_profile_df[volume_profile_df['is_low_volume_node']]
            
            # Find low volume nodes near current price
            price_range = 0.05 * current_price  # 5% range
            nearby_low_volume = low_volume_nodes[
                (low_volume_nodes['price_level'] > current_price - price_range) &
                (low_volume_nodes['price_level'] < current_price + price_range)
            ]
            
            # Get potential breakout levels
            breakout_level = nearby_low_volume['price_level'].iloc[0] if not nearby_low_volume.empty else None
            
            # Calculate stop loss and take profit levels
            if nearest_support is not None and nearest_resistance is not None:
                # For buy signals
                buy_stop_loss = nearest_support * 0.99  # 1% below support
                buy_take_profit = nearest_resistance * 1.01  # 1% above resistance
                
                # For short signals
                short_stop_loss = nearest_resistance * 1.01  # 1% above resistance
                short_take_profit = nearest_support * 0.99  # 1% below support
            else:
                # Fallback to percentage-based levels
                buy_stop_loss = current_price * 0.95  # 5% below current price
                buy_take_profit = current_price * 1.10  # 10% above current price
                short_stop_loss = current_price * 1.05  # 5% above current price
                short_take_profit = current_price * 0.90  # 10% below current price
            
            # Return results
            return {
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'potential_breakout': breakout_level,
                'buy_stop_loss': buy_stop_loss,
                'buy_take_profit': buy_take_profit,
                'short_stop_loss': short_stop_loss,
                'short_take_profit': short_take_profit
            }
            
        except Exception as e:
            logger.error(f"Error calculating support and resistance: {str(e)}")
            return {}
