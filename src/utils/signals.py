import numpy as np
import pandas as pd

class SignalGenerator:
    def __init__(self):
        self.signals = []

    def generate_volume_signals(self, df):
        """Generate trading signals based on O'Neil's volume patterns"""
        signals = []

        # Calculate required metrics
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['price_ma'] = df['Close'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()

        # Volume Breakout Signal (150%+ volume)
        volume_breakout = df['Volume'] > 1.5 * df['volume_ma']

        # Failed Breakout (price falls back below breakout level)
        failed_breakout = (df['Close'] < df['Close'].shift(1)) & (df['Close'].shift(1) > df['price_ma'].shift(1))

        # Volume Climax (10%+ price change on 300%+ volume)
        volume_climax = (abs(df['price_change']) > 0.10) & (df['volume_ratio'] > 3)

        # Weak Rally (price up, volume down)
        weak_rally = (df['price_change'] > 0) & (df['volume_change'] < 0)

        # Price-Volume Divergence
        price_up = df['Close'] > df['Close'].shift(1)
        volume_up = df['Volume'] > df['Volume'].shift(1)
        divergence = price_up != volume_up

        # Generate signals using boolean indexing
        dates_with_breakouts = df[volume_breakout].index
        for date in dates_with_breakouts:
            ratio = df.loc[date, 'volume_ratio']
            signals.append({
                'timestamp': date,
                'type': 'VOLUME_BREAKOUT',
                'strength': 'HIGH' if ratio > 2 else 'MEDIUM',
                'message': f"Volume spike {ratio:.1f}x above average"
            })

        dates_with_failed_breakouts = df[failed_breakout].index
        for date in dates_with_failed_breakouts:
            signals.append({
                'timestamp': date,
                'type': 'FAILED_BREAKOUT',
                'strength': 'HIGH',
                'message': "Price fell below breakout level"
            })

        dates_with_climax = df[volume_climax].index
        for date in dates_with_climax:
            ratio = df.loc[date, 'volume_ratio']
            signals.append({
                'timestamp': date,
                'type': 'VOLUME_CLIMAX',
                'strength': 'HIGH',
                'message': f"Possible exhaustion move on {ratio:.1f}x volume"
            })

        dates_with_weak_rally = df[weak_rally].index
        for date in dates_with_weak_rally:
            signals.append({
                'timestamp': date,
                'type': 'WEAK_RALLY',
                'strength': 'MEDIUM',
                'message': "Price up on declining volume"
            })

        dates_with_divergence = df[divergence].index
        for date in dates_with_divergence:
            signals.append({
                'timestamp': date,
                'type': 'PRICE_VOLUME_DIVERGENCE',
                'strength': 'MEDIUM',
                'message': "Price and volume moving in opposite directions"
            })

        return signals[::-1]  # Return most recent signals first