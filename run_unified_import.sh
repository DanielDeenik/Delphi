#!/bin/bash
echo "Running Unified Import for Delphi Trading Intelligence System"
echo "============================================================"

# Set environment variables
export PYTHONPATH=$(dirname "$0")
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    exit 1
fi

# Run the unified import
python3 unified_import.py "$@"

echo "============================================================"
echo "Import process completed with exit code $?"
