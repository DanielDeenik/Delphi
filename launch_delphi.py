#!/usr/bin/env python3
"""
Cross-platform launcher for the Delphi Trading Intelligence System.

This script automatically launches the Delphi application and opens the browser.
It works on Windows, macOS, and Linux.
"""
import os
import sys
import time
import json
import logging
import platform
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_banner(message):
    """Print a banner message."""
    print("\n" + "=" * 54)
    print(f"    {message}")
    print("=" * 54 + "\n")

def setup_environment():
    """Set up the environment."""
    print_banner("Delphi Trading Intelligence System Launcher")
    
    logger.info("Setting up environment...")
    print("[1/4] Setting up environment...")
    
    # Create necessary directories
    for directory in ["logs", "status", "templates", "static", "config"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Check if config file exists
    config_file = Path("config/config.json")
    if not config_file.exists():
        logger.info("Creating default configuration...")
        config = {
            "alpha_vantage": {
                "api_key": "IAS7UEKOT0HZW0MY",
                "base_url": "https://www.alphavantage.co/query"
            },
            "google_cloud": {
                "project_id": "delphi-449908",
                "dataset": "stock_data"
            },
            "tickers": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
                "TSLA", "NVDA", "JPM", "V", "JNJ",
                "WMT", "PG", "MA", "UNH", "HD",
                "BAC", "PFE", "CSCO", "DIS", "VZ"
            ]
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    return True

def start_application():
    """Start the Flask application."""
    logger.info("Starting Delphi application on port 3000...")
    print("[2/4] Starting Delphi application on port 3000...")
    
    # Prepare command
    cmd = [sys.executable, "-m", "trading_ai.cli.dashboard_cli", "--port", "3000"]
    
    # Start the application
    try:
        if platform.system() == "Windows":
            # Windows - use subprocess.Popen with CREATE_NEW_CONSOLE
            process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            # Unix-like systems - use subprocess.Popen with stdout/stderr redirection
            log_path = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            with open(log_path, "w") as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=log,
                    text=True
                )
        
        # Save PID to file for later cleanup
        with open(".app.pid", "w") as f:
            f.write(str(process.pid))
        
        logger.info(f"Application started with PID {process.pid}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"[ERROR] Failed to start application: {str(e)}")
        return False

def wait_for_application():
    """Wait for the application to start."""
    logger.info("Waiting for application to start...")
    print("[3/4] Waiting for application to start...")
    
    # Wait for the application to start
    time.sleep(5)
    return True

def open_browser():
    """Open the browser to the multi-tab Colab view."""
    logger.info("Opening browser...")
    print("[4/4] Opening browser...")
    
    # URL to open
    url = "http://localhost:3000/colab/all"
    
    # Try to open the browser
    try:
        webbrowser.open(url)
        logger.info(f"Browser opened to {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to open browser: {str(e)}")
        print(f"[WARNING] Could not open browser automatically.")
        print(f"[WARNING] Please open {url} manually in your browser.")
        return False

def display_info():
    """Display information about the application."""
    print("\n" + "=" * 54)
    print("    Delphi Trading Intelligence System is running")
    print("    Dashboard URL: http://localhost:3000")
    print("    Notebooks URL: http://localhost:3000/colab")
    print("    All Notebooks: http://localhost:3000/colab/all")
    print("=" * 54 + "\n")
    print("Press Ctrl+C to stop the application")
    print("")

def cleanup():
    """Clean up resources."""
    logger.info("Cleaning up resources...")
    
    # Kill the application process if PID file exists
    pid_file = Path(".app.pid")
    if pid_file.exists():
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            # Kill the process
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
            else:
                subprocess.run(["kill", str(pid)], check=False)
            
            logger.info(f"Process with PID {pid} killed")
        except Exception as e:
            logger.error(f"Failed to kill process: {str(e)}")
        
        # Remove PID file
        pid_file.unlink(missing_ok=True)
    
    logger.info("Cleanup complete")
    print("\nDelphi application stopped.")

def main():
    """Main entry point for the script."""
    try:
        # Set up the environment
        if not setup_environment():
            return 1
        
        # Start the application
        if not start_application():
            return 1
        
        # Wait for the application to start
        if not wait_for_application():
            return 1
        
        # Open the browser
        open_browser()
        
        # Display information
        display_info()
        
        # Keep the script running until Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping Delphi application...")
        
        return 0
    
    except KeyboardInterrupt:
        print("\nStopping Delphi application...")
        return 0
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"[ERROR] {str(e)}")
        return 1
    
    finally:
        # Clean up resources
        cleanup()

if __name__ == "__main__":
    sys.exit(main())
