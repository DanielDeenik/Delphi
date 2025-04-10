"""
Correlation Plots

This module provides functions for plotting correlations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def plot_correlation_matrix(corr_df: pd.DataFrame, output_dir: str = 'plots', filename: str = 'correlation_matrix.png') -> str:
    """
    Plot correlation matrix.
    
    Args:
        corr_df: Correlation DataFrame
        output_dir: Directory to save plot to
        filename: Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    try:
        if corr_df is None or corr_df.empty:
            logger.error("No data to plot")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        
        # Add title
        plt.title('Correlation Matrix')
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved correlation matrix to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        return ""

def plot_normalized_prices(price_df: pd.DataFrame, output_dir: str = 'plots', filename: str = 'normalized_prices.png') -> str:
    """
    Plot normalized prices.
    
    Args:
        price_df: DataFrame with price data for multiple symbols
        output_dir: Directory to save plot to
        filename: Filename for the plot
        
    Returns:
        str: Path to the saved plot
    """
    try:
        if price_df is None or price_df.empty:
            logger.error("No data to plot")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Normalize prices to start at 100 for comparison
        normalized_df = price_df.div(price_df.iloc[0]) * 100
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot normalized prices
        for column in normalized_df.columns:
            plt.plot(normalized_df.index, normalized_df[column], label=column)
        
        # Add title and labels
        plt.title('Normalized Price Comparison (Base = 100)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved normalized price comparison to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error plotting normalized prices: {str(e)}")
        return ""
