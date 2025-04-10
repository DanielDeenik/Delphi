"""
Configuration Utilities

This module provides utilities for configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env(env_file: str = ".env") -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        bool: Success status
    """
    try:
        # Load environment variables from .env file
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
        return True
    except Exception as e:
        logger.error(f"Error loading environment variables from {env_file}: {str(e)}")
        return False

def get_config() -> Dict[str, Any]:
    """
    Get the configuration.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Default configuration
    config = {
        "database": {
            "use_bigquery": os.environ.get("USE_BIGQUERY", "").lower() == "true",
            "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            "dataset_id": os.environ.get("BIGQUERY_DATASET", "market_data"),
            "table_id": os.environ.get("BIGQUERY_TABLE", "time_series"),
            "location": os.environ.get("BIGQUERY_LOCATION", "US")
        },
        "alpha_vantage": {
            "api_key": os.environ.get("ALPHA_VANTAGE_API_KEY"),
            "key_file": os.environ.get("ALPHA_VANTAGE_KEY_FILE"),
            "rate_limit": int(os.environ.get("ALPHA_VANTAGE_RATE_LIMIT", "5")),
            "rate_limit_period": int(os.environ.get("ALPHA_VANTAGE_RATE_LIMIT_PERIOD", "60"))
        },
        "symbols": {
            "default": ['^VIX', 'SPY', 'PLTR'],
            "names": {
                '^VIX': 'CBOE Volatility Index',
                'SPY': 'SPDR S&P 500 ETF Trust',
                'PLTR': 'Palantir Technologies Inc.'
            }
        },
        "analysis": {
            "default_days": int(os.environ.get("DEFAULT_ANALYSIS_DAYS", "90")),
            "volume_spike_threshold": float(os.environ.get("VOLUME_SPIKE_THRESHOLD", "2.0")),
            "volume_drop_threshold": float(os.environ.get("VOLUME_DROP_THRESHOLD", "0.5"))
        },
        "performance": {
            "use_numba": os.environ.get("USE_NUMBA", "").lower() == "true",
            "max_workers": int(os.environ.get("MAX_WORKERS", "4")),
            "cache_size": int(os.environ.get("CACHE_SIZE", "128"))
        }
    }
    
    # Load configuration from file if available
    config_path = os.environ.get("DELPHI_CONFIG")
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with file values
            _update_dict(config, file_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    return config

def _update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a dictionary recursively.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Dict[str, Any]: Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_dict(d[k], v)
        else:
            d[k] = v
    return d
