#!/usr/bin/env python
"""
Setup script for environment variables.

This script helps set up environment variables for the Delphi project.

Usage:
    python scripts/setup_env.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    # Get the path to the .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    
    # Read existing .env file if it exists
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    
    # Ask for Alpha Vantage API key if not set
    if 'ALPHA_VANTAGE_API_KEY' not in env_vars:
        print("\nAlpha Vantage API Key")
        print("---------------------")
        print("You need an Alpha Vantage API key to fetch market data.")
        print("You can get a free API key at: https://www.alphavantage.co/support/#api-key")
        api_key = input("Enter your Alpha Vantage API key: ")
        if api_key:
            env_vars['ALPHA_VANTAGE_API_KEY'] = api_key
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in env_vars.items():
            f.write(f"{key}={value}\n")
    
    logger.info(f"Environment variables saved to {env_path}")
    logger.info("You can now run the test script:")
    logger.info("python scripts/test_alpha_vantage.py")
    
    # Print instructions for setting environment variables in the current session
    print("\nTo set environment variables for the current terminal session:")
    for key, value in env_vars.items():
        print(f"export {key}={value}")

if __name__ == "__main__":
    main()
