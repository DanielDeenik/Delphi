
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats

class MonteCarloRiskManager:
    def __init__(self, simulations: int = 10000, time_horizon: int = 30):
        self.simulations = simulations
        self.time_horizon = time_horizon
        self.confidence_level = 0.95

    def simulate_price_paths(self, returns: np.ndarray) -> np.ndarray:
        """Generate Monte Carlo price paths"""
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        return np.exp(
            np.cumsum(
                np.random.normal(
                    mu, 
                    sigma, 
                    (self.simulations, self.time_horizon)
                ), 
                axis=1
            )
        )

    def calculate_var_cvar(self, paths: np.ndarray) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        final_values = paths[:, -1]
        var = np.percentile(final_values, (1 - self.confidence_level) * 100)
        cvar = np.mean(final_values[final_values < var])
        return var, cvar

    def optimize_position_size(self, 
                             returns: np.ndarray, 
                             stop_loss: float,
                             max_position: float = 0.1) -> Dict[str, float]:
        """Optimize position size using risk metrics"""
        paths = self.simulate_price_paths(returns)
        var, cvar = self.calculate_var_cvar(paths)
        
        stop_loss_prob = np.mean(paths[:, -1] < stop_loss)
        kelly_fraction = self._calculate_kelly(returns)
        
        # Risk-adjusted position size
        position_size = min(
            kelly_fraction * (1 - stop_loss_prob),
            max_position
        )
        
        return {
            'optimal_size': position_size,
            'var': var,
            'cvar': cvar,
            'stop_loss_prob': stop_loss_prob
        }

    def _calculate_kelly(self, returns: np.ndarray) -> float:
        """Calculate Kelly Criterion fraction"""
        win_prob = np.mean(returns > 0)
        avg_win = np.mean(returns[returns > 0])
        avg_loss = abs(np.mean(returns[returns < 0]))
        
        return (win_prob / avg_loss) - ((1 - win_prob) / avg_win)
