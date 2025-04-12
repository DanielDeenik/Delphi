"""
Script to run the BigQuery import with proper authentication.
"""
import os
import subprocess
import sys
import platform
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def authenticate_gcloud():
    """Authenticate with Google Cloud."""
    try:
        logger.info("Authenticating with Google Cloud...")
        subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
        logger.info("Authentication successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Authentication failed: {str(e)}")
        return False
    except FileNotFoundError:
        logger.error("gcloud command not found. Please install Google Cloud SDK.")
        return False

def run_import(batch_size=3, force_full=False):
    """Run the BigQuery import."""
    try:
        # Check if we're on Windows
        if platform.system() == 'Windows':
            logger.info("Running Windows-compatible import script...")
            script = "import_to_bigquery_windows.py"
        else:
            logger.info("Running standard import script...")
            script = "import_to_bigquery_using_trading_ai.py"

        # Build command
        cmd = [sys.executable, script, f"--batch-size={batch_size}"]
        if force_full:
            cmd.append("--force-full")

        # Run the import script
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        logger.info("Import completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Import failed: {str(e)}")
        return False

def main():
    """Main function."""
    # Skip authentication since it's already done manually
    logger.info("Using existing Google Cloud authentication...")

    # Run the import
    success = run_import(batch_size=3, force_full=False)

    if success:
        print("\nData import completed successfully!")
    else:
        print("\nData import failed. Check the logs for details.")

    return success

if __name__ == "__main__":
    main()
