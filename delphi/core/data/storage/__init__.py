"""
Storage services for Delphi.

This module contains storage service implementations.
"""

from delphi.core.data.storage.base import StorageService
from delphi.core.data.storage.bigquery import BigQueryStorage

# Factory function for creating storage services
def create_storage_service(storage_type: str, **kwargs):
    """Create a storage service.
    
    Args:
        storage_type: Type of storage service to create
        **kwargs: Additional arguments for the storage service
        
    Returns:
        StorageService instance
    """
    if storage_type == "bigquery":
        return BigQueryStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
