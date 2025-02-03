"""Financial indicators calculation module"""
from typing import Dict, List
import pandas as pd
import numpy as np

class FinancialIndicators:
    """Calculate various financial indicators and technical analysis metrics"""
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate multiple moving averages"""
        for window in windows:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        return df
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_Middle'] = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * num_std)
        return df

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate volume profile indicators"""
        df['Volume_MA'] = df['Volume'].rolling(window=window).mean()
        df['Volume_Std'] = df['Volume'].rolling(window=window).std()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available indicators"""
        df = FinancialIndicators.calculate_moving_averages(df)
        df = FinancialIndicators.calculate_rsi(df)
        df = FinancialIndicators.calculate_macd(df)
        df = FinancialIndicators.calculate_bollinger_bands(df)
        df = FinancialIndicators.calculate_volume_profile(df)
        return df
