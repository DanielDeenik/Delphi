"""
Data sources for Delphi.

This module contains data source implementations.
"""

from delphi.core.data.sources.base import DataSource
from delphi.core.data.sources.alpha_vantage import AlphaVantageClient

# Factory function for creating data sources
def create_data_source(source_type: str, **kwargs):
    """Create a data source.
    
    Args:
        source_type: Type of data source to create
        **kwargs: Additional arguments for the data source
        
    Returns:
        DataSource instance
    """
    if source_type == "alpha_vantage":
        return AlphaVantageClient(**kwargs)
    else:
        raise ValueError(f"Unknown data source type: {source_type}")
