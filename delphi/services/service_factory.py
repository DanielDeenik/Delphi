"""
Service Factory

This module provides a factory for creating storage services.
"""

import os
import logging
from typing import Dict, Any, Optional

from .storage_service import StorageService
from .bigquery_service import BigQueryService
from ..utils.config import get_config

logger = logging.getLogger(__name__)

def get_storage_service(service_type: str = None, **kwargs) -> StorageService:
    """
    Get a storage service.
    
    Args:
        service_type: Type of storage service (if None, will try to load from config)
        **kwargs: Additional arguments for the storage service
        
    Returns:
        StorageService: Storage service instance
    """
    # Load config
    config = get_config()
    
    # Get service type from config if not provided
    if service_type is None:
        if config and 'database' in config and 'use_bigquery' in config['database']:
            service_type = "bigquery" if config['database']['use_bigquery'] else "sqlite"
        else:
            service_type = os.environ.get("STORAGE_SERVICE_TYPE", "bigquery")
    
    # Create service
    if service_type.lower() == "bigquery":
        return BigQueryService(**kwargs)
    else:
        raise ValueError(f"Unknown storage service type: {service_type}")
