"""
Mosaic Theory Evaluator

This module implements the Mosaic Theory approach to stock evaluation,
combining multiple sources of information using NLP and other analysis techniques.
"""

import os
import logging
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class MosaicTheoryEvaluator:
    """
    Evaluates stocks using the Mosaic Theory approach.
    
    Combines technical analysis, fundamental analysis, sentiment analysis,
    and other factors to form a comprehensive evaluation.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the evaluator.
        
        Args:
            db_path: Path to SQLite database (default: data/market_data.db)
        """
        # Set up database path
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            self.db_path = os.path.join(data_dir, 'market_data.db')
        else:
            self.db_path = db_path
        
        # Component weights (can be adjusted)
        self.weights = {
            'technical': 0.35,
            'fundamental': 0.30,
            'sentiment': 0.20,
            'volume': 0.15
        }
        
        # Initialize sub-components
        self._init_components()
    
    def _init_components(self):
        """Initialize analysis components."""
        # In a full implementation, these would be separate classes
        # For now, we'll implement them as methods
        pass
    
    def get_market_data(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """
        Retrieve market data for a symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days of data to retrieve
            
        Returns:
            DataFrame with market data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query data
            cursor.execute('''
            SELECT date, data
            FROM market_data
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
            ''', (symbol, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                logger.warning(f"No data found for {symbol} in the specified date range")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data_list = []
            for date_str, data_json in rows:
                row_data = json.loads(data_json)
                row_data['date'] = pd.to_datetime(date_str)
                data_list.append(row_data)
            
            df = pd.DataFrame(data_list)
            
            # Ensure date is the index
            if 'date' in df.columns:
                df = df.set_index('date')
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dict with technical indicators and scores
        """
        if data.empty:
            return {'error': 'No data available for technical analysis'}
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have the necessary columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {'error': f"Missing required columns: {', '.join(missing_columns)}"}
            
            # Calculate moving averages
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            # Calculate MACD
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
            
            # Get the latest values
            latest = df.iloc[-1]
            
            # Calculate trend signals
            trend_signals = {
                'ma_10_50_cross': 1 if df['MA_10'].iloc[-1] > df['MA_50'].iloc[-1] and df['MA_10'].iloc[-2] <= df['MA_50'].iloc[-2] else
                               -1 if df['MA_10'].iloc[-1] < df['MA_50'].iloc[-1] and df['MA_10'].iloc[-2] >= df['MA_50'].iloc[-2] else 0,
                'ma_50_200_cross': 1 if df['MA_50'].iloc[-1] > df['MA_200'].iloc[-1] and df['MA_50'].iloc[-2] <= df['MA_200'].iloc[-2] else
                                 -1 if df['MA_50'].iloc[-1] < df['MA_200'].iloc[-1] and df['MA_50'].iloc[-2] >= df['MA_200'].iloc[-2] else 0,
                'price_above_ma_200': 1 if latest['Close'] > latest['MA_200'] else -1,
                'macd_signal_cross': 1 if latest['MACD'] > latest['MACD_Signal'] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2] else
                                   -1 if latest['MACD'] < latest['MACD_Signal'] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2] else 0,
                'rsi_overbought': -1 if latest['RSI'] > 70 else 0,
                'rsi_oversold': 1 if latest['RSI'] < 30 else 0,
                'bollinger_breakout_up': 1 if latest['Close'] > latest['BB_Upper'] else 0,
                'bollinger_breakout_down': -1 if latest['Close'] < latest['BB_Lower'] else 0
            }
            
            # Calculate overall trend score (-100 to 100)
            trend_score = sum(trend_signals.values()) * 12.5  # Scale to -100 to 100
            
            # Determine trend direction
            if trend_score > 50:
                trend = "Strong Uptrend"
            elif trend_score > 20:
                trend = "Uptrend"
            elif trend_score > -20:
                trend = "Neutral"
            elif trend_score > -50:
                trend = "Downtrend"
            else:
                trend = "Strong Downtrend"
            
            # Calculate momentum
            momentum = (latest['Close'] / df['Close'].iloc[-10] - 1) * 100
            
            # Calculate volatility
            volatility = df['Close'].pct_change().rolling(window=20).std() * 100
            
            return {
                'trend_score': trend_score,
                'trend': trend,
                'momentum': momentum.iloc[-1],
                'volatility': volatility.iloc[-1],
                'rsi': latest['RSI'],
                'macd': latest['MACD'],
                'macd_signal': latest['MACD_Signal'],
                'bollinger_width': (latest['BB_Upper'] - latest['BB_Lower']) / latest['BB_Middle'] * 100,
                'signals': trend_signals
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume patterns.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dict with volume analysis and scores
        """
        if data.empty:
            return {'error': 'No data available for volume analysis'}
        
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure we have the necessary columns
            if 'Volume' not in df.columns or 'Close' not in df.columns:
                return {'error': "Missing required columns: Volume and/or Close"}
            
            # Calculate volume indicators
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_50']
            
            # Calculate price-volume relationship
            df['Price_Change'] = df['Close'].pct_change()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Identify volume spikes
            df['Volume_Spike'] = (df['Volume'] > df['Volume_MA_10'] * 2).astype(int)
            
            # Get the latest values
            latest = df.iloc[-1]
            
            # Calculate volume signals
            volume_signals = {
                'volume_trend': 1 if latest['Volume_MA_10'] > latest['Volume_MA_50'] else -1,
                'volume_spike': 1 if latest['Volume_Spike'] == 1 else 0,
                'price_volume_confirm': 1 if (latest['Price_Change'] > 0 and latest['Volume_Change'] > 0) or
                                          (latest['Price_Change'] < 0 and latest['Volume_Change'] < 0) else -1
            }
            
            # Calculate overall volume score (-100 to 100)
            volume_score = sum(volume_signals.values()) * 33.33  # Scale to -100 to 100
            
            # Determine volume signal
            if volume_score > 50:
                signal = "Strong Volume Confirmation"
            elif volume_score > 0:
                signal = "Volume Confirmation"
            elif volume_score > -50:
                signal = "Neutral Volume"
            else:
                signal = "Volume Divergence"
            
            # Calculate average volume ratio
            avg_volume_ratio = df['Volume_Ratio'].iloc[-5:].mean()
            
            return {
                'volume_score': volume_score,
                'volume_signal': signal,
                'volume_ratio': latest['Volume_Ratio'],
                'avg_volume_ratio': avg_volume_ratio,
                'volume_trend': 'Increasing' if volume_signals['volume_trend'] > 0 else 'Decreasing',
                'volume_spike_detected': bool(volume_signals['volume_spike']),
                'price_volume_confirmation': bool(volume_signals['price_volume_confirm']),
                'signals': volume_signals
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment for a stock.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with sentiment analysis and scores
        """
        # In a full implementation, this would analyze news, social media, etc.
        # For now, we'll return a mock implementation
        
        try:
            # Mock sentiment data
            import random
            
            # Generate a random sentiment score between -100 and 100
            sentiment_score = random.uniform(-100, 100)
            
            # Determine sentiment category
            if sentiment_score > 50:
                sentiment = "Very Bullish"
            elif sentiment_score > 20:
                sentiment = "Bullish"
            elif sentiment_score > -20:
                sentiment = "Neutral"
            elif sentiment_score > -50:
                sentiment = "Bearish"
            else:
                sentiment = "Very Bearish"
            
            # Mock news sentiment
            news_sentiment = random.uniform(-1, 1)
            
            # Mock social media sentiment
            social_sentiment = random.uniform(-1, 1)
            
            # Mock sentiment change
            sentiment_change = random.uniform(-0.5, 0.5)
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment': sentiment,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'sentiment_change': sentiment_change,
                'sources': {
                    'news_count': random.randint(5, 50),
                    'social_mentions': random.randint(10, 1000),
                    'analyst_ratings': random.randint(3, 15)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze fundamental data for a stock.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with fundamental analysis and scores
        """
        # In a full implementation, this would analyze financial statements, ratios, etc.
        # For now, we'll return a mock implementation
        
        try:
            # Mock fundamental data
            import random
            
            # Generate random fundamental metrics
            pe_ratio = random.uniform(10, 30)
            pb_ratio = random.uniform(1, 5)
            ps_ratio = random.uniform(1, 10)
            debt_equity = random.uniform(0.1, 2)
            
            # Generate random growth rates
            revenue_growth = random.uniform(-0.1, 0.3)
            earnings_growth = random.uniform(-0.2, 0.4)
            
            # Calculate fundamental score
            fundamental_score = 0
            
            # PE ratio contribution
            if pe_ratio < 15:
                fundamental_score += 25
            elif pe_ratio < 25:
                fundamental_score += 10
            else:
                fundamental_score -= 10
            
            # PB ratio contribution
            if pb_ratio < 2:
                fundamental_score += 20
            elif pb_ratio < 4:
                fundamental_score += 5
            else:
                fundamental_score -= 10
            
            # Debt/Equity contribution
            if debt_equity < 0.5:
                fundamental_score += 20
            elif debt_equity < 1:
                fundamental_score += 10
            else:
                fundamental_score -= 10
            
            # Growth contribution
            fundamental_score += revenue_growth * 100
            fundamental_score += earnings_growth * 100
            
            # Cap the score at -100 to 100
            fundamental_score = max(min(fundamental_score, 100), -100)
            
            # Determine fundamental rating
            if fundamental_score > 50:
                rating = "Strong Buy"
            elif fundamental_score > 20:
                rating = "Buy"
            elif fundamental_score > -20:
                rating = "Hold"
            elif fundamental_score > -50:
                rating = "Sell"
            else:
                rating = "Strong Sell"
            
            return {
                'fundamental_score': fundamental_score,
                'rating': rating,
                'metrics': {
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'ps_ratio': ps_ratio,
                    'debt_equity': debt_equity
                },
                'growth': {
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_stock(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """
        Evaluate a stock using the Mosaic Theory approach.
        
        Args:
            symbol: Trading symbol
            days: Number of days of data to analyze
            
        Returns:
            Dict with comprehensive evaluation
        """
        logger.info(f"Evaluating {symbol} using Mosaic Theory")
        
        try:
            # Get market data
            data = self.get_market_data(symbol, days)
            
            if data.empty:
                return {'error': f"No market data available for {symbol}"}
            
            # Perform component analyses
            technical = self._calculate_technical_indicators(data)
            volume = self._analyze_volume(data)
            sentiment = self._analyze_sentiment(symbol)
            fundamental = self._analyze_fundamentals(symbol)
            
            # Check for errors in any component
            components = {
                'technical': technical,
                'volume': volume,
                'sentiment': sentiment,
                'fundamental': fundamental
            }
            
            for name, component in components.items():
                if 'error' in component:
                    logger.warning(f"Error in {name} analysis for {symbol}: {component['error']}")
            
            # Calculate composite score
            composite_score = 0
            valid_weights_sum = 0
            
            for name, component in components.items():
                if 'error' not in component:
                    score_key = f"{name}_score" if name != 'fundamental' else 'fundamental_score'
                    if score_key in component:
                        composite_score += component[score_key] * self.weights[name]
                        valid_weights_sum += self.weights[name]
            
            # Normalize composite score if we have valid components
            if valid_weights_sum > 0:
                composite_score /= valid_weights_sum
            else:
                return {'error': "No valid analysis components available"}
            
            # Determine overall rating
            if composite_score > 70:
                rating = "Strong Buy"
            elif composite_score > 30:
                rating = "Buy"
            elif composite_score > -30:
                rating = "Hold"
            elif composite_score > -70:
                rating = "Sell"
            else:
                rating = "Strong Sell"
            
            # Create evaluation result
            evaluation = {
                'symbol': symbol,
                'composite_score': composite_score,
                'rating': rating,
                'components': components,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Evaluation completed for {symbol}. Rating: {rating}, Score: {composite_score:.2f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def evaluate_multiple(self, symbols: List[str], days: int = 90) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple stocks.
        
        Args:
            symbols: List of trading symbols
            days: Number of days of data to analyze
            
        Returns:
            Dict mapping symbols to their evaluations
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.evaluate_stock(symbol, days)
        
        return results


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize evaluator
    evaluator = MosaicTheoryEvaluator()
    
    # Evaluate a stock
    evaluation = evaluator.evaluate_stock('AAPL')
    
    # Print evaluation
    print(json.dumps(evaluation, indent=2))
