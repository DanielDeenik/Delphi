"""
Volume Plots

This module provides functions for plotting volume patterns.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def plot_volume(df: pd.DataFrame, output_dir: str = 'plots', symbol: str = None, show_obv: bool = True) -> Dict[str, str]:
    """
    Plot volume patterns.
    
    Args:
        df: DataFrame with market data
        output_dir: Directory to save plots to
        symbol: Symbol to use in plot titles and filenames
        show_obv: Whether to show On-Balance Volume plot
        
    Returns:
        Dict[str, str]: Dictionary mapping plot types to file paths
    """
    try:
        if df is None or df.empty:
            logger.error("No data to plot")
            return {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get symbol from DataFrame if not provided
        if symbol is None and 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]
        
        # Use a default symbol if still None
        symbol = symbol or 'UNKNOWN'
        
        # Create volume plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price
        ax1.plot(df.index, df['close'], label='Close Price')
        ax1.set_title(f'{symbol} - Price and Volume')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        # Plot volume
        ax2.bar(df.index, df['volume'], label='Volume')
        if 'volume_ma20' in df.columns:
            ax2.plot(df.index, df['volume_ma20'], color='red', label='20-day MA')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        volume_path = os.path.join(output_dir, f'{symbol}_volume.png')
        plt.savefig(volume_path)
        plt.close()
        
        logger.info(f"Saved volume pattern plot to {volume_path}")
        
        # Create OBV plot if OBV is available and show_obv is True
        obv_path = None
        if show_obv and 'obv' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot price
            ax1.plot(df.index, df['close'], label='Close Price')
            ax1.set_title(f'{symbol} - Price and On-Balance Volume (OBV)')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot OBV
            ax2.plot(df.index, df['obv'], label='OBV')
            if 'obv_ma20' in df.columns:
                ax2.plot(df.index, df['obv_ma20'], color='red', label='20-day MA')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('OBV')
            ax2.grid(True)
            ax2.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            obv_path = os.path.join(output_dir, f'{symbol}_obv.png')
            plt.savefig(obv_path)
            plt.close()
            
            logger.info(f"Saved OBV plot to {obv_path}")
        
        # Return paths
        return {
            "volume": volume_path,
            "obv": obv_path
        }
        
    except Exception as e:
        logger.error(f"Error plotting volume patterns: {str(e)}")
        return {}
