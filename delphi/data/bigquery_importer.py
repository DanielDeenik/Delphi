"""
BigQuery Importer

This module provides a class for importing data to BigQuery.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..services.bigquery_service import BigQueryService
from ..utils.config import get_config

logger = logging.getLogger(__name__)

class BigQueryImporter:
    """
    Class for importing data to BigQuery.
    """
    
    def __init__(self, project_id: str = None, dataset_id: str = None, table_id: str = None):
        """
        Initialize the BigQuery importer.
        
        Args:
            project_id: Google Cloud project ID (if None, will try to load from config)
            dataset_id: BigQuery dataset ID (if None, will try to load from config)
            table_id: BigQuery table ID (if None, will try to load from config)
        """
        # Load config
        config = get_config()
        
        # Set project ID
        self.project_id = project_id
        if self.project_id is None:
            self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if self.project_id is None and config and 'database' in config and 'project_id' in config['database']:
            self.project_id = config['database']['project_id']
        
        # Set dataset ID
        self.dataset_id = dataset_id
        if self.dataset_id is None:
            self.dataset_id = os.environ.get("BIGQUERY_DATASET", "market_data")
        if self.dataset_id is None and config and 'database' in config and 'dataset_id' in config['database']:
            self.dataset_id = config['database']['dataset_id']
        
        # Set table ID
        self.table_id = table_id
        if self.table_id is None:
            self.table_id = os.environ.get("BIGQUERY_TABLE", "time_series")
        if self.table_id is None and config and 'database' in config and 'table_id' in config['database']:
            self.table_id = config['database']['table_id']
        
        if not self.project_id:
            raise ValueError("Google Cloud project ID is required")
        
        # Initialize BigQuery service
        self.service = BigQueryService(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_id=self.table_id
        )
        
        logger.info(f"Initialized BigQuery importer for {self.project_id}.{self.dataset_id}.{self.table_id}")
    
    def import_data(self, data_source, symbols: List[str], symbol_names: Dict[str, str] = None, max_workers: int = 4, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Import data from a data source to BigQuery.
        
        Args:
            data_source: Data source to import from
            symbols: List of symbols to import
            symbol_names: Dictionary mapping symbols to names
            max_workers: Maximum number of workers for parallel processing
            **kwargs: Additional arguments for the data source
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for each symbol
        """
        try:
            # Set up BigQuery
            if not self.service.setup_dataset() or not self.service.setup_table():
                logger.error("Failed to set up BigQuery")
                return {symbol: {"success": False, "message": "Failed to set up BigQuery"} for symbol in symbols}
            
            # Fetch data
            data = data_source.fetch_batch(symbols, max_workers=max_workers, **kwargs)
            
            # Process and import each symbol
            results = {}
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary mapping futures to symbols
                future_to_symbol = {}
                
                for symbol, df in data.items():
                    if df.empty:
                        results[symbol] = {
                            "success": False,
                            "message": "No data returned",
                            "rows": 0
                        }
                        continue
                    
                    # Add symbol name if provided
                    if symbol_names and symbol in symbol_names:
                        df["symbol_name"] = symbol_names[symbol]
                    else:
                        df["symbol_name"] = symbol
                    
                    # Add created_at timestamp
                    df["created_at"] = datetime.now()
                    
                    # Submit task to executor
                    future = executor.submit(self.service.store_market_data, symbol, df)
                    future_to_symbol[future] = symbol
                
                # Process completed futures
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        success = future.result()
                        
                        if success:
                            results[symbol] = {
                                "success": True,
                                "message": f"Successfully imported {len(data[symbol])} rows",
                                "rows": len(data[symbol])
                            }
                        else:
                            results[symbol] = {
                                "success": False,
                                "message": "Failed to store data",
                                "rows": 0
                            }
                    except Exception as e:
                        logger.error(f"Error importing data for {symbol}: {str(e)}")
                        results[symbol] = {
                            "success": False,
                            "message": f"Error: {str(e)}",
                            "rows": 0
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error importing data: {str(e)}")
            return {symbol: {"success": False, "message": f"Error: {str(e)}"} for symbol in symbols}
