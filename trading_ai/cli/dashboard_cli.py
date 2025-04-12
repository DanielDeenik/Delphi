#!/usr/bin/env python3
"""
Command-line interface for launching the dashboard.
"""
import argparse
import logging
import sys
import os
import subprocess
import webbrowser
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Launch the Delphi Trading Intelligence Dashboard")

    # Dashboard options
    parser.add_argument("--port", type=int, default=8080, help="Port to run the dashboard on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the dashboard on")
    parser.add_argument("--browser", action="store_true", help="Open dashboard in browser")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")

    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()

    try:
        # Get the path to the app.py file
        app_path = Path(__file__).parent.parent.parent / "app.py"

        if not app_path.exists():
            logger.error(f"Dashboard file not found: {app_path}")
            return 1

        # Set environment variables for Flask
        os.environ["FLASK_APP"] = str(app_path)
        if args.debug:
            os.environ["FLASK_ENV"] = "development"
            os.environ["FLASK_DEBUG"] = "1"

        # Launch the dashboard
        logger.info(f"Launching dashboard on {args.host}:{args.port}")

        # Open browser if requested
        if args.browser:
            url = f"http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}"
            webbrowser.open(url)

        # Use subprocess to run Flask
        cmd = [
            sys.executable, "-m", "flask", "run",
            "--host", args.host,
            "--port", str(args.port)
        ]

        # Run the command
        process = subprocess.run(cmd)

        return process.returncode

    except Exception as e:
        logger.error(f"Error launching dashboard: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
