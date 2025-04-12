#!/bin/bash
echo "Running Time Series Import for Delphi Trading Intelligence System"
echo "============================================================"

# Set environment variables
export PYTHONPATH=$(dirname "$0")
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    exit 1
fi

# Run the import script
python3 run_time_series_import.py "$@"

echo "============================================================"
echo "Import process completed with exit code $?"
