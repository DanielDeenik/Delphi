#!/usr/bin/env python3
"""
Analyze data imported to BigQuery.
Based on working code from GitHub repository.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from google.cloud import bigquery

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Cloud project ID - replace with your project ID
PROJECT_ID = "delphi-449908"

# BigQuery dataset and table
DATASET_ID = "market_data"
TABLE_ID = "time_series"

# Dictionary of symbols with their names
SYMBOLS_DICT = {
    '^VIX': 'CBOE Volatility Index',
    'SPY': 'SPDR S&P 500 ETF Trust',
    'PLTR': 'Palantir Technologies Inc.'
}

def fetch_data_from_bigquery(days=90):
    """
    Fetch data from BigQuery.
    
    Args:
        days: Number of days to fetch
        
    Returns:
        dict: Dictionary mapping symbols to DataFrames
    """
    try:
        # Initialize BigQuery client
        client = bigquery.Client(project=PROJECT_ID)
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        symbols_str = ", ".join([f"'{symbol}'" for symbol in SYMBOLS_DICT.keys()])
        query = f"""
        SELECT 
            symbol, 
            symbol_name,
            date, 
            open, 
            high, 
            low, 
            close, 
            adjusted_close, 
            volume
        FROM 
            `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
        WHERE 
            symbol IN ({symbols_str})
            AND date >= '{start_date}'
            AND date <= '{end_date}'
        ORDER BY 
            symbol, date
        """
        
        # Execute query
        logger.info(f"Fetching data from BigQuery for {len(SYMBOLS_DICT)} symbols...")
        query_job = client.query(query)
        df = query_job.to_dataframe()
        
        if df.empty:
            logger.error("No data returned from BigQuery")
            return {}
        
        logger.info(f"Retrieved {len(df)} rows from BigQuery")
        
        # Split data by symbol
        data = {}
        for symbol in SYMBOLS_DICT.keys():
            symbol_data = df[df['symbol'] == symbol].copy()
            if not symbol_data.empty:
                # Set date as index
                symbol_data = symbol_data.set_index('date')
                # Sort by date
                symbol_data = symbol_data.sort_index()
                
                data[symbol] = symbol_data
                logger.info(f"Retrieved {len(symbol_data)} rows for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data from BigQuery: {str(e)}")
        return {}

def analyze_volume_patterns(data):
    """
    Analyze volume patterns in the data.
    
    Args:
        data: Dictionary mapping symbols to DataFrames
        
    Returns:
        dict: Analysis results for each symbol
    """
    results = {}
    
    for symbol, df in data.items():
        try:
            logger.info(f"Analyzing volume patterns for {symbol}...")
            
            # Calculate volume moving averages
            df['volume_ma5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ma50'] = df['volume'].rolling(window=50).mean()
            
            # Calculate volume ratios
            df['volume_ratio_5'] = df['volume'] / df['volume_ma5']
            df['volume_ratio_20'] = df['volume'] / df['volume_ma20']
            
            # Identify volume spikes (volume > 2x 20-day average)
            volume_spikes = df[df['volume_ratio_20'] > 2].copy()
            
            # Identify volume drops (volume < 0.5x 20-day average)
            volume_drops = df[df['volume_ratio_20'] < 0.5].copy()
            
            # Calculate price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(periods=5)
            
            # Identify price-volume divergences
            # Price up, volume down
            price_up_volume_down = df[(df['price_change'] > 0) & (df['volume_ratio_20'] < 0.8)].copy()
            
            # Price down, volume up
            price_down_volume_up = df[(df['price_change'] < 0) & (df['volume_ratio_20'] > 1.5)].copy()
            
            # Calculate On-Balance Volume (OBV)
            df['obv'] = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    df['obv'].iloc[i] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    df['obv'].iloc[i] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
                else:
                    df['obv'].iloc[i] = df['obv'].iloc[i-1]
            
            # Calculate OBV moving average
            df['obv_ma20'] = df['obv'].rolling(window=20).mean()
            
            # Identify OBV divergences
            # Price up, OBV down
            obv_bearish_divergence = df[(df['price_change_5d'] > 0) & (df['obv'].diff(5) < 0)].copy()
            
            # Price down, OBV up
            obv_bullish_divergence = df[(df['price_change_5d'] < 0) & (df['obv'].diff(5) > 0)].copy()
            
            # Get recent volume patterns (last 5 days)
            recent_volume = df.iloc[-5:][['close', 'volume', 'volume_ratio_20']].copy() if len(df) >= 5 else pd.DataFrame()
            
            # Store results
            results[symbol] = {
                'data_points': len(df),
                'volume_spikes': len(volume_spikes),
                'volume_drops': len(volume_drops),
                'price_up_volume_down': len(price_up_volume_down),
                'price_down_volume_up': len(price_down_volume_up),
                'obv_bearish_divergence': len(obv_bearish_divergence),
                'obv_bullish_divergence': len(obv_bullish_divergence),
                'recent_volume': recent_volume.to_dict() if not recent_volume.empty else {},
                'avg_volume': df['volume'].mean(),
                'max_volume': df['volume'].max(),
                'min_volume': df['volume'].min(),
                'avg_volume_ratio': df['volume_ratio_20'].mean(),
                'last_close': df['close'].iloc[-1] if not df.empty else None,
                'last_volume': df['volume'].iloc[-1] if not df.empty else None,
                'last_volume_ratio': df['volume_ratio_20'].iloc[-1] if not df.empty and not np.isnan(df['volume_ratio_20'].iloc[-1]) else None,
                'data': df  # Include the DataFrame for plotting
            }
            
            logger.info(f"Analysis completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns for {symbol}: {str(e)}")
            results[symbol] = {'error': str(e)}
    
    return results

def analyze_correlations(data):
    """
    Analyze correlations between symbols.
    
    Args:
        data: Dictionary mapping symbols to DataFrames
        
    Returns:
        dict: Correlation analysis results
    """
    try:
        if len(data) < 2:
            logger.error("Not enough data for correlation analysis")
            return {'error': "Not enough data for correlation analysis"}
        
        # Create DataFrames for price and returns
        price_df = pd.DataFrame()
        return_df = pd.DataFrame()
        volume_df = pd.DataFrame()
        
        # Extract data for each symbol
        for symbol, df in data.items():
            # Add close price
            price_df[symbol] = df['close']
            
            # Add daily returns
            return_df[symbol] = df['close'].pct_change().fillna(0)
            
            # Add volume
            volume_df[symbol] = df['volume']
        
        # Calculate correlations
        price_corr = price_df.corr()
        return_corr = return_df.corr()
        volume_corr = volume_df.corr()
        
        # Calculate VIX-SPY correlation specifically if both exist
        vix_spy_corr = None
        if '^VIX' in return_df.columns and 'SPY' in return_df.columns:
            vix_spy_corr = return_df['^VIX'].corr(return_df['SPY'])
            logger.info(f"VIX-SPY correlation: {vix_spy_corr:.4f}")
        
        # Calculate VIX-PLTR correlation specifically if both exist
        vix_pltr_corr = None
        if '^VIX' in return_df.columns and 'PLTR' in return_df.columns:
            vix_pltr_corr = return_df['^VIX'].corr(return_df['PLTR'])
            logger.info(f"VIX-PLTR correlation: {vix_pltr_corr:.4f}")
        
        # Calculate SPY-PLTR correlation specifically if both exist
        spy_pltr_corr = None
        if 'SPY' in return_df.columns and 'PLTR' in return_df.columns:
            spy_pltr_corr = return_df['SPY'].corr(return_df['PLTR'])
            logger.info(f"SPY-PLTR correlation: {spy_pltr_corr:.4f}")
        
        # Prepare results
        results = {
            'price_correlation': price_corr.to_dict(),
            'return_correlation': return_corr.to_dict(),
            'volume_correlation': volume_corr.to_dict(),
            'key_correlations': {
                'VIX_SPY': vix_spy_corr,
                'VIX_PLTR': vix_pltr_corr,
                'SPY_PLTR': spy_pltr_corr
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing correlations: {str(e)}")
        return {'error': str(e)}

def plot_volume_patterns(results, output_dir='plots'):
    """
    Plot volume patterns.
    
    Args:
        results: Analysis results for each symbol
        output_dir: Directory to save plots to
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, result in results.items():
            try:
                if 'error' in result or 'data' not in result:
                    continue
                
                df = result['data']
                
                logger.info(f"Creating volume pattern plots for {symbol}...")
                
                # Create figure with 2 subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot price
                ax1.plot(df.index, df['close'], label='Close Price')
                ax1.set_title(f'{symbol} - Price and Volume')
                ax1.set_ylabel('Price')
                ax1.grid(True)
                ax1.legend()
                
                # Plot volume
                ax2.bar(df.index, df['volume'], label='Volume')
                ax2.plot(df.index, df['volume_ma20'], color='red', label='20-day MA')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.grid(True)
                ax2.legend()
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                output_path = os.path.join(output_dir, f'{symbol}_volume.png')
                plt.savefig(output_path)
                plt.close()
                
                logger.info(f"Saved volume pattern plot for {symbol} to {output_path}")
                
                # Create OBV plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot price
                ax1.plot(df.index, df['close'], label='Close Price')
                ax1.set_title(f'{symbol} - Price and On-Balance Volume (OBV)')
                ax1.set_ylabel('Price')
                ax1.grid(True)
                ax1.legend()
                
                # Plot OBV
                ax2.plot(df.index, df['obv'], label='OBV')
                ax2.plot(df.index, df['obv_ma20'], color='red', label='20-day MA')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('OBV')
                ax2.grid(True)
                ax2.legend()
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                output_path = os.path.join(output_dir, f'{symbol}_obv.png')
                plt.savefig(output_path)
                plt.close()
                
                logger.info(f"Saved OBV plot for {symbol} to {output_path}")
                
            except Exception as e:
                logger.error(f"Error creating plots for {symbol}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")

def plot_correlations(data, correlation_results, output_dir='plots'):
    """
    Plot correlation analysis results.
    
    Args:
        data: Dictionary mapping symbols to DataFrames
        correlation_results: Correlation analysis results
        output_dir: Directory to save plots to
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot correlation heatmap
        if 'return_correlation' in correlation_results:
            # Convert dictionary to DataFrame
            corr_df = pd.DataFrame(correlation_results['return_correlation'])
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            
            # Add title
            plt.title('Return Correlation Matrix')
            
            # Save figure
            output_path = os.path.join(output_dir, 'return_correlation_heatmap.png')
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Saved correlation heatmap to {output_path}")
        
        # Plot normalized price comparison
        price_df = pd.DataFrame()
        for symbol, df in data.items():
            price_df[symbol] = df['close']
        
        # Normalize prices to start at 100 for comparison
        normalized_df = price_df.div(price_df.iloc[0]) * 100
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot normalized prices
        for symbol in normalized_df.columns:
            plt.plot(normalized_df.index, normalized_df[symbol], label=symbol)
        
        # Add title and labels
        plt.title('Normalized Price Comparison (Base = 100)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.grid(True)
        plt.legend()
        
        # Save figure
        output_path = os.path.join(output_dir, 'normalized_price_comparison.png')
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved normalized price comparison to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating correlation plots: {str(e)}")

def main():
    """Main function."""
    try:
        # Fetch data from BigQuery
        data = fetch_data_from_bigquery(days=90)
        
        if not data:
            logger.error("No data to analyze")
            return False
        
        # Analyze volume patterns
        volume_results = analyze_volume_patterns(data)
        
        # Plot volume patterns
        plot_volume_patterns(volume_results)
        
        # Analyze correlations
        correlation_results = analyze_correlations(data)
        
        # Plot correlations
        plot_correlations(data, correlation_results)
        
        # Print volume analysis summary
        logger.info("\nVolume Analysis Summary:")
        for symbol, result in volume_results.items():
            if 'error' in result:
                logger.error(f"{symbol}: Error - {result['error']}")
                continue
                
            logger.info(f"\n{symbol} Analysis:")
            logger.info(f"  Data points: {result['data_points']}")
            logger.info(f"  Volume spikes: {result['volume_spikes']}")
            logger.info(f"  Volume drops: {result['volume_drops']}")
            logger.info(f"  Price up, volume down: {result['price_up_volume_down']}")
            logger.info(f"  Price down, volume up: {result['price_down_volume_up']}")
            logger.info(f"  OBV bearish divergence: {result['obv_bearish_divergence']}")
            logger.info(f"  OBV bullish divergence: {result['obv_bullish_divergence']}")
            logger.info(f"  Average volume: {result['avg_volume']:.2f}")
            logger.info(f"  Last close: {result['last_close']:.2f}")
            logger.info(f"  Last volume: {result['last_volume']}")
            logger.info(f"  Last volume ratio: {result['last_volume_ratio']:.2f}")
        
        # Print correlation analysis summary
        logger.info("\nCorrelation Analysis Summary:")
        if 'error' in correlation_results:
            logger.error(f"Correlation analysis failed: {correlation_results['error']}")
        else:
            for pair, corr in correlation_results['key_correlations'].items():
                if corr is not None:
                    logger.info(f"  {pair}: {corr:.4f}")
        
        logger.info("\nAnalysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the analysis
    success = main()
    
    if success:
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed. Check the logs for details.")
