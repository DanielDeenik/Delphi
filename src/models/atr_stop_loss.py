
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ATRStopLoss:
    def __init__(self, atr_multiplier: float = 1.5, atr_period: int = 14):
        self.atr_multiplier = atr_multiplier
        self.atr_period = atr_period

    def calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
        
        return float(atr)

    def get_stop_level(self, data: pd.DataFrame, position: str) -> float:
        """Calculate stop level based on ATR"""
        try:
            atr = self.calculate_atr(data)
            current_price = data['Close'].iloc[-1]
            
            if position.upper() == 'LONG':
                return current_price - (self.atr_multiplier * atr)
            elif position.upper() == 'SHORT':
                return current_price + (self.atr_multiplier * atr)
            
            return current_price
            
        except Exception as e:
            logger.error(f"Error calculating stop level: {str(e)}")
            return current_price
