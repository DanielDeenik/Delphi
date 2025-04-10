#!/usr/bin/env python
"""
Setup script for InfluxDB.

This script helps set up InfluxDB for the Delphi project.
It creates the necessary organization, bucket, and token.

Usage:
    python setup_influxdb.py [--host HOST] [--port PORT]
"""

import os
import sys
import argparse
import requests
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up InfluxDB for Delphi')
    parser.add_argument('--host', default='localhost', help='InfluxDB host (default: localhost)')
    parser.add_argument('--port', default='8086', help='InfluxDB port (default: 8086)')
    parser.add_argument('--username', default='admin', help='Admin username (default: admin)')
    parser.add_argument('--password', default='adminpassword', help='Admin password (default: adminpassword)')
    parser.add_argument('--org', default='delphi', help='Organization name (default: delphi)')
    parser.add_argument('--bucket', default='market_data', help='Bucket name (default: market_data)')
    parser.add_argument('--retention', default='30d', help='Data retention period (default: 30d)')
    return parser.parse_args()

def setup_influxdb(args):
    """Set up InfluxDB with the specified configuration."""
    base_url = f"http://{args.host}:{args.port}"
    
    # Step 1: Check if InfluxDB is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            logger.error(f"InfluxDB is not running at {base_url}. Status code: {response.status_code}")
            return False
        logger.info("InfluxDB is running")
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to InfluxDB at {base_url}")
        logger.info("Make sure InfluxDB is running and accessible")
        logger.info("You can start InfluxDB using Docker with:")
        logger.info("docker run -d -p 8086:8086 --name influxdb influxdb:2.7")
        return False
    
    # Step 2: Set up initial user, org, and bucket (if not already set up)
    try:
        setup_response = requests.post(
            f"{base_url}/api/v2/setup",
            json={
                "username": args.username,
                "password": args.password,
                "org": args.org,
                "bucket": args.bucket,
                "retentionPeriodSeconds": parse_retention_period(args.retention)
            }
        )
        
        if setup_response.status_code == 422:
            logger.info("InfluxDB is already set up. Proceeding with existing setup.")
        elif setup_response.status_code == 201:
            setup_data = setup_response.json()
            logger.info(f"InfluxDB setup completed successfully")
            logger.info(f"Organization ID: {setup_data['org']['id']}")
            logger.info(f"Bucket ID: {setup_data['bucket']['id']}")
            logger.info(f"User ID: {setup_data['user']['id']}")
            
            # Save the token to .env file
            token = setup_data['auth']['token']
            save_env_variables(args, token)
            return True
        else:
            logger.error(f"Failed to set up InfluxDB. Status code: {setup_response.status_code}")
            logger.error(f"Response: {setup_response.text}")
            return False
    except Exception as e:
        logger.error(f"Error during setup: {str(e)}")
        return False
    
    # If we get here, InfluxDB was already set up, so we need to get a token
    try:
        # Get authentication token
        auth_response = requests.post(
            f"{base_url}/api/v2/signin",
            auth=(args.username, args.password)
        )
        
        if auth_response.status_code != 204:
            logger.error(f"Authentication failed. Status code: {auth_response.status_code}")
            return False
        
        # Get the session cookie
        session_cookie = auth_response.cookies.get_dict()
        
        # Get organizations
        orgs_response = requests.get(
            f"{base_url}/api/v2/orgs",
            cookies=session_cookie
        )
        
        if orgs_response.status_code != 200:
            logger.error(f"Failed to get organizations. Status code: {orgs_response.status_code}")
            return False
        
        orgs_data = orgs_response.json()
        org_id = None
        
        for org in orgs_data['orgs']:
            if org['name'] == args.org:
                org_id = org['id']
                break
        
        if not org_id:
            logger.error(f"Organization '{args.org}' not found")
            return False
        
        # Create a new token
        token_response = requests.post(
            f"{base_url}/api/v2/authorizations",
            json={
                "description": "Delphi API Token",
                "orgID": org_id,
                "permissions": [
                    {
                        "action": "read",
                        "resource": {
                            "orgID": org_id,
                            "type": "buckets"
                        }
                    },
                    {
                        "action": "write",
                        "resource": {
                            "orgID": org_id,
                            "type": "buckets"
                        }
                    }
                ]
            },
            cookies=session_cookie
        )
        
        if token_response.status_code != 201:
            logger.error(f"Failed to create token. Status code: {token_response.status_code}")
            return False
        
        token_data = token_response.json()
        token = token_data['token']
        
        # Save the token to .env file
        save_env_variables(args, token)
        return True
        
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        return False

def parse_retention_period(retention):
    """Parse retention period string (e.g., '30d', '24h') to seconds."""
    if retention.endswith('d'):
        days = int(retention[:-1])
        return days * 24 * 60 * 60
    elif retention.endswith('h'):
        hours = int(retention[:-1])
        return hours * 60 * 60
    elif retention.endswith('m'):
        minutes = int(retention[:-1])
        return minutes * 60
    elif retention.endswith('s'):
        return int(retention[:-1])
    else:
        # Default to days if no unit specified
        return int(retention) * 24 * 60 * 60

def save_env_variables(args, token):
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
    env_vars['INFLUXDB_URL'] = f"http://{args.host}:{args.port}"
    env_vars['INFLUXDB_TOKEN'] = token
    env_vars['INFLUXDB_ORG'] = args.org
    env_vars['INFLUXDB_BUCKET'] = args.bucket
    env_vars['INFLUXDB_RETENTION_PERIOD'] = args.retention
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"InfluxDB configuration saved to {env_path}")
    logger.info("You can now use the following environment variables in your application:")
    logger.info(f"  INFLUXDB_URL={env_vars['INFLUXDB_URL']}")
    logger.info(f"  INFLUXDB_TOKEN={env_vars['INFLUXDB_TOKEN']}")
    logger.info(f"  INFLUXDB_ORG={env_vars['INFLUXDB_ORG']}")
    logger.info(f"  INFLUXDB_BUCKET={env_vars['INFLUXDB_BUCKET']}")

def main():
    """Main entry point."""
    args = parse_args()
    if setup_influxdb(args):
        logger.info("InfluxDB setup completed successfully")
    else:
        logger.error("InfluxDB setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
