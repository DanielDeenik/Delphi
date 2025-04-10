#!/usr/bin/env python
"""
Setup script for InfluxDB Cloud.

This script helps set up environment variables for connecting to InfluxDB Cloud.

Usage:
    python setup_influxdb_cloud.py --url YOUR_INFLUXDB_CLOUD_URL --token YOUR_TOKEN --org YOUR_ORG --bucket YOUR_BUCKET
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up InfluxDB Cloud for Delphi')
    parser.add_argument('--url', required=True, help='InfluxDB Cloud URL (e.g., https://us-west-2-1.aws.cloud2.influxdata.com)')
    parser.add_argument('--token', required=True, help='InfluxDB Cloud API token')
    parser.add_argument('--org', required=True, help='InfluxDB Cloud organization name')
    parser.add_argument('--bucket', default='market_data', help='Bucket name (default: market_data)')
    return parser.parse_args()

def save_env_variables(args):
    """Save InfluxDB configuration to .env file."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Read existing .env file if it exists
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    # Update with InfluxDB variables
    env_vars['INFLUXDB_URL'] = args.url
    env_vars['INFLUXDB_TOKEN'] = args.token
    env_vars['INFLUXDB_ORG'] = args.org
    env_vars['INFLUXDB_BUCKET'] = args.bucket
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"InfluxDB Cloud configuration saved to {env_path}")
    logger.info("You can now use the following environment variables in your application:")
    logger.info(f"  INFLUXDB_URL={env_vars['INFLUXDB_URL']}")
    logger.info(f"  INFLUXDB_TOKEN=<hidden>")
    logger.info(f"  INFLUXDB_ORG={env_vars['INFLUXDB_ORG']}")
    logger.info(f"  INFLUXDB_BUCKET={env_vars['INFLUXDB_BUCKET']}")

def main():
    """Main entry point."""
    args = parse_args()
    save_env_variables(args)
    logger.info("InfluxDB Cloud setup completed successfully")
    logger.info("\nNext steps:")
    logger.info("1. Make sure you have the influxdb-client Python package installed:")
    logger.info("   pip install influxdb-client")
    logger.info("2. Test your connection by running a simple query:")
    logger.info("   python -c \"from src.services.influxdb_storage_service import InfluxDBStorageService; print(InfluxDBStorageService().client.ping())\"")

if __name__ == "__main__":
    main()
