"""
Volume analyzer for detecting volume patterns and inefficiencies.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from volume_trading.config import config

# Configure logging
logger = logging.getLogger(__name__)

class VolumeAnalyzer:
    """Analyzer for volume patterns and inefficiencies."""
    
    def __init__(self, ma_periods: Optional[List[int]] = None, spike_threshold: Optional[float] = None):
        """Initialize the volume analyzer.
        
        Args:
            ma_periods: List of periods for moving averages
            spike_threshold: Z-score threshold for volume spikes
        """
        self.ma_periods = ma_periods or config.get("volume_ma_periods")
        self.spike_threshold = spike_threshold or config.get("volume_spike_threshold")
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume patterns in a DataFrame.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with added volume metrics
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame, skipping analysis")
                return df
            
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Ensure DataFrame is sorted by date (oldest first)
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df = result_df.sort_index(ascending=True)
            
            # Calculate volume moving averages
            for period in self.ma_periods:
                result_df[f"volume_ma{period}"] = result_df["volume"].rolling(window=period).mean()
            
            # Calculate relative volume (ratio to moving averages)
            for period in self.ma_periods:
                result_df[f"rel_volume_{period}d"] = result_df["volume"] / result_df[f"volume_ma{period}"]
            
            # Calculate volume Z-score (using 20-day MA and STD)
            result_df["volume_std20"] = result_df["volume"].rolling(window=20).std()
            result_df["volume_z_score"] = (result_df["volume"] - result_df["volume_ma20"]) / result_df["volume_std20"]
            
            # Identify volume spikes
            result_df["is_volume_spike"] = result_df["volume_z_score"] > self.spike_threshold
            
            # Calculate price change
            result_df["price_change"] = result_df["close"].pct_change() * 100
            
            # Generate signals
            result_df["signal"] = "NEUTRAL"
            result_df["signal_strength"] = 0.0
            result_df["notes"] = ""
            
            # Analyze patterns
            self._analyze_patterns(result_df)
            
            # Sort by date (newest first) before returning
            if isinstance(result_df.index, pd.DatetimeIndex):
                result_df = result_df.sort_index(ascending=False)
            
            logger.info(f"Successfully analyzed volume patterns for {result_df['symbol'].iloc[0]}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return df
    
    def _analyze_patterns(self, df: pd.DataFrame) -> None:
        """Analyze volume patterns and generate signals.
        
        Args:
            df: DataFrame with volume metrics
        """
        # Get ticker and direction
        ticker = df["symbol"].iloc[0]
        tracked_stocks = config.get_tracked_stocks()
        
        direction = None
        for dir_type, tickers in tracked_stocks.items():
            if ticker in tickers:
                direction = dir_type
                break
        
        # Skip if ticker is not in tracked stocks
        if direction is None:
            return
        
        # Analyze each row (except the first one)
        for i in range(1, len(df)):
            # Skip if we don't have enough data
            if pd.isna(df.iloc[i]["volume_z_score"]) or pd.isna(df.iloc[i]["price_change"]):
                continue
            
            # Current and previous row
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Volume spike with price change
            if curr["is_volume_spike"]:
                # For buy candidates
                if direction == "buy":
                    # Volume spike with positive price change
                    if curr["price_change"] > 0:
                        df.loc[df.index[i], "signal"] = "BUY"
                        df.loc[df.index[i], "signal_strength"] = min(curr["volume_z_score"] / 5, 1.0)
                        df.loc[df.index[i], "notes"] = f"Volume spike (Z={curr['volume_z_score']:.2f}) with positive price change ({curr['price_change']:.2f}%)"
                    
                    # Volume spike with negative price change (potential reversal)
                    elif curr["price_change"] < -1.0 and curr["volume_z_score"] > 3.0:
                        df.loc[df.index[i], "signal"] = "POTENTIAL_REVERSAL"
                        df.loc[df.index[i], "signal_strength"] = min(curr["volume_z_score"] / 6, 0.8)
                        df.loc[df.index[i], "notes"] = f"Volume spike (Z={curr['volume_z_score']:.2f}) with negative price change ({curr['price_change']:.2f}%) - potential reversal"
                
                # For short candidates
                elif direction == "short":
                    # Volume spike with negative price change
                    if curr["price_change"] < 0:
                        df.loc[df.index[i], "signal"] = "SHORT"
                        df.loc[df.index[i], "signal_strength"] = min(curr["volume_z_score"] / 5, 1.0)
                        df.loc[df.index[i], "notes"] = f"Volume spike (Z={curr['volume_z_score']:.2f}) with negative price change ({curr['price_change']:.2f}%)"
                    
                    # Volume spike with positive price change (potential reversal)
                    elif curr["price_change"] > 1.0 and curr["volume_z_score"] > 3.0:
                        df.loc[df.index[i], "signal"] = "POTENTIAL_REVERSAL"
                        df.loc[df.index[i], "signal_strength"] = min(curr["volume_z_score"] / 6, 0.8)
                        df.loc[df.index[i], "notes"] = f"Volume spike (Z={curr['volume_z_score']:.2f}) with positive price change ({curr['price_change']:.2f}%) - potential reversal"
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get a summary of volume analysis.
        
        Args:
            df: DataFrame with volume analysis results
            
        Returns:
            Dictionary with summary information
        """
        try:
            if df.empty:
                return {
                    "ticker": "",
                    "direction": "",
                    "latest_close": 0.0,
                    "latest_volume": 0,
                    "latest_volume_z_score": 0.0,
                    "is_volume_spike": False,
                    "latest_signal": "NEUTRAL",
                    "signal_strength": 0.0,
                    "notes": "",
                    "volume_spikes_count": 0,
                    "buy_signals_count": 0,
                    "short_signals_count": 0,
                    "reversal_signals_count": 0
                }
            
            # Get ticker and direction
            ticker = df["symbol"].iloc[0]
            tracked_stocks = config.get_tracked_stocks()
            
            direction = None
            for dir_type, tickers in tracked_stocks.items():
                if ticker in tickers:
                    direction = dir_type
                    break
            
            # Get latest data
            latest = df.iloc[0]  # Assuming sorted by date desc
            
            # Count signals
            volume_spikes_count = df["is_volume_spike"].sum()
            buy_signals_count = (df["signal"] == "BUY").sum()
            short_signals_count = (df["signal"] == "SHORT").sum()
            reversal_signals_count = (df["signal"] == "POTENTIAL_REVERSAL").sum()
            
            # Return summary
            return {
                "ticker": ticker,
                "direction": direction or "",
                "latest_close": latest.get("close", 0.0),
                "latest_volume": latest.get("volume", 0),
                "latest_volume_z_score": latest.get("volume_z_score", 0.0),
                "is_volume_spike": latest.get("is_volume_spike", False),
                "latest_signal": latest.get("signal", "NEUTRAL"),
                "signal_strength": latest.get("signal_strength", 0.0),
                "notes": latest.get("notes", ""),
                "volume_spikes_count": volume_spikes_count,
                "buy_signals_count": buy_signals_count,
                "short_signals_count": short_signals_count,
                "reversal_signals_count": reversal_signals_count
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return {}
