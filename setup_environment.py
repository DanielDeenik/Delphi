#!/usr/bin/env python3
"""
Setup environment for data import.

This script sets up the environment for data import by:
1. Checking if the Google Cloud SDK is installed
2. Checking if the Alpha Vantage API key is valid
3. Checking if the BigQuery dataset exists
4. Creating the BigQuery dataset if it doesn't exist
"""
import os
import sys
import logging
import json
import requests
from pathlib import Path
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_google_cloud_sdk():
    """Check if the Google Cloud SDK is installed."""
    try:
        from google.cloud import bigquery
        logger.info("Google Cloud SDK is installed")
        return True
    except ImportError:
        logger.error("Google Cloud SDK is not installed. Please install it with: pip install google-cloud-bigquery")
        return False

def check_alpha_vantage_api_key(api_key):
    """Check if the Alpha Vantage API key is valid."""
    if not api_key:
        logger.error("Alpha Vantage API key is not set")
        return False
    
    # Test the API key with a simple request
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey={api_key}&outputsize=compact"
    try:
        response = requests.get(url)
        data = response.json()
        
        if "Error Message" in data:
            logger.error(f"Alpha Vantage API key is invalid: {data['Error Message']}")
            return False
        
        if "Time Series (Daily)" in data:
            logger.info("Alpha Vantage API key is valid")
            return True
        
        logger.warning(f"Unexpected response from Alpha Vantage: {data}")
        return False
    except Exception as e:
        logger.error(f"Error checking Alpha Vantage API key: {str(e)}")
        return False

def check_bigquery_dataset(project_id, dataset_id):
    """Check if the BigQuery dataset exists."""
    try:
        client = bigquery.Client(project=project_id)
        dataset_ref = client.dataset(dataset_id)
        
        try:
            client.get_dataset(dataset_ref)
            logger.info(f"BigQuery dataset {project_id}.{dataset_id} exists")
            return True
        except NotFound:
            logger.warning(f"BigQuery dataset {project_id}.{dataset_id} does not exist")
            return False
    except Exception as e:
        logger.error(f"Error checking BigQuery dataset: {str(e)}")
        return False

def create_bigquery_dataset(project_id, dataset_id):
    """Create the BigQuery dataset."""
    try:
        client = bigquery.Client(project=project_id)
        dataset_ref = client.dataset(dataset_id)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        
        dataset = client.create_dataset(dataset)
        logger.info(f"Created BigQuery dataset {project_id}.{dataset_id}")
        return True
    except Exception as e:
        logger.error(f"Error creating BigQuery dataset: {str(e)}")
        return False

def setup_environment():
    """Set up the environment for data import."""
    # Check if the Google Cloud SDK is installed
    if not check_google_cloud_sdk():
        return False
    
    # Load configuration
    try:
        config_path = Path("config/system_config.json")
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        project_id = config.get("google_cloud_project")
        dataset_id = config.get("bigquery_dataset")
        api_key = config.get("alpha_vantage_api_key")
        
        if not project_id:
            logger.error("Google Cloud project ID is not set in the configuration")
            return False
        
        if not dataset_id:
            logger.error("BigQuery dataset ID is not set in the configuration")
            return False
        
        if not api_key:
            logger.error("Alpha Vantage API key is not set in the configuration")
            return False
        
        # Check if the Alpha Vantage API key is valid
        if not check_alpha_vantage_api_key(api_key):
            return False
        
        # Check if the BigQuery dataset exists
        if not check_bigquery_dataset(project_id, dataset_id):
            # Create the BigQuery dataset
            if not create_bigquery_dataset(project_id, dataset_id):
                return False
        
        logger.info("Environment setup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_environment():
        logger.info("Environment setup completed successfully")
        sys.exit(0)
    else:
        logger.error("Environment setup failed")
        sys.exit(1)
